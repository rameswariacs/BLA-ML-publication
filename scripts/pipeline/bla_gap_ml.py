#!/usr/bin/env python3
"""
Build bond-length-alternation/conjugation descriptors from DFT JSON records and,
when HOMO-LUMO gaps or interaction energies are available, train regression
models to predict the target.

The script is intentionally geometry-first: the descriptors are not tied to a
specific phenalenyl atom numbering, which makes them better candidates for
transfer to related polyaromatic dimers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import statistics
import tempfile
import warnings
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
INTERMONOMER_CC_SIGMA_CUTOFF = 1.75
COVALENT_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}
SYMBOL_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
}
DEFAULT_PI_ATOM_FILENAMES = (
    "pi_atoms.txt",
    "conjugated_atoms.txt",
    "bla_atoms.txt",
    "pi_atom_indices.txt",
)
PLOT_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#5F6368",
    "light_gray": "#E8EAED",
}
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_range": math.nan,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_count": float(len(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=0)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
    }


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("molecules"), list):
            return data["molecules"]
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    raise ValueError("Could not find a list of molecule records in the JSON file.")


def read_atoms(record: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    atoms = record.get("atoms") or []
    atomic_numbers: list[int] = []
    coords: list[list[float]] = []
    for atom in atoms:
        if "xyz_coordinates" not in atom:
            continue
        atomic_numbers.append(int(atom["atomic_number"]))
        coords.append([float(x) for x in atom["xyz_coordinates"]])
    if not coords:
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no xyz_coordinates.")
    return atomic_numbers, np.asarray(coords, dtype=float)


def read_xyz(path: Path) -> tuple[list[int], np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty.")
    try:
        n_atoms = int(lines[0].strip())
        atom_lines = lines[2 : 2 + n_atoms]
    except ValueError:
        atom_lines = lines
    atomic_numbers: list[int] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0].upper()
        z = SYMBOL_TO_Z.get(symbol)
        if z is None:
            try:
                z = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Unknown element '{parts[0]}' in {path}") from exc
        atomic_numbers.append(z)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coords:
        raise ValueError(f"No atoms found in {path}.")
    return atomic_numbers, np.asarray(coords, dtype=float)


def parse_atom_index_list(text: str, n_atoms: int | None = None) -> list[int]:
    """Parse comma/space separated atom indices, including ranges such as 1-13.

    Lists are treated as 1-based by default, matching chemistry file
    conventions. If any explicit index is 0, the whole list is treated as
    zero-based.
    """
    raw: list[int] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in re.split(r"[\s,;]+", line):
            if not token:
                continue
            match = re.fullmatch(r"(\d+)-(\d+)", token)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                step = 1 if end >= start else -1
                raw.extend(range(start, end + step, step))
            else:
                raw.append(int(token))
    if not raw:
        return []
    zero_based = any(value == 0 for value in raw)
    indices = raw if zero_based else [value - 1 for value in raw]
    unique = sorted(set(indices))
    if n_atoms is not None:
        bad = [idx for idx in unique if idx < 0 or idx >= n_atoms]
        if bad:
            raise ValueError(f"Pi atom index outside atom range 0-{n_atoms - 1}: {bad[:8]}")
    return unique


def read_pi_atom_indices(path: Path, n_atoms: int | None = None) -> list[int]:
    return parse_atom_index_list(path.read_text(errors="replace"), n_atoms=n_atoms)


def resolve_pi_atom_file(root: Path, explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        supplied = Path(explicit)
        if supplied.is_absolute():
            candidates.append(supplied)
        else:
            candidates.append(root / supplied)
    candidates.extend(root / name for name in DEFAULT_PI_ATOM_FILENAMES)
    candidates.extend(root / "Monomer" / name for name in DEFAULT_PI_ATOM_FILENAMES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_sp2_carbon_indices(
    atomic_numbers: list[int], coords: np.ndarray, bonds: list[tuple[int, int, float]]
) -> list[int]:
    neighbors: dict[int, list[int]] = {idx: [] for idx in range(len(atomic_numbers))}
    distances: dict[frozenset[int], float] = {}
    for i, j, d in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        distances[frozenset((i, j))] = d

    sp2_like: list[int] = []
    for idx, z in enumerate(atomic_numbers):
        if z != 6:
            continue
        bonded = neighbors[idx]
        carbon_neighbors = [j for j in bonded if atomic_numbers[j] == 6]
        if len(bonded) >= 4:
            continue
        if len(carbon_neighbors) < 2:
            continue
        cc_lengths = [distances[frozenset((idx, j))] for j in carbon_neighbors]
        if float(np.mean(cc_lengths)) > 1.53:
            continue
        if len(bonded) >= 3:
            vectors = []
            for nbr in bonded:
                vec = coords[nbr] - coords[idx]
                length = norm(vec)
                if length > 0:
                    vectors.append(vec / length)
            if len(vectors) >= 3:
                angles = []
                for i, vi in enumerate(vectors):
                    for vj in vectors[i + 1 :]:
                        angles.append(math.degrees(math.acos(float(np.clip(np.dot(vi, vj), -1.0, 1.0)))))
                if angles and max(angles) < 102.0:
                    continue
        sp2_like.append(idx)
    return sp2_like


def choose_pi_atom_indices(
    atomic_numbers: list[int],
    coords: np.ndarray,
    bonds: list[tuple[int, int, float]],
    pi_atom_indices: list[int] | None = None,
) -> tuple[list[int], str]:
    carbon_set = {idx for idx, z in enumerate(atomic_numbers) if z == 6}
    if pi_atom_indices is not None:
        selected = sorted(set(pi_atom_indices))
        noncarbon = [idx for idx in selected if idx not in carbon_set]
        if noncarbon:
            raise ValueError(f"Pi atom list contains non-carbon atoms: {noncarbon[:8]}")
        return selected, "provided"
    return detect_sp2_carbon_indices(atomic_numbers, coords, bonds), "auto_sp2"


def descriptor_from_atoms(
    atomic_numbers: list[int],
    coords: np.ndarray,
    base: dict[str, Any] | None = None,
    pi_atom_indices: list[int] | None = None,
) -> dict[str, float | str | None]:
    bonds = build_bonds(atomic_numbers, coords)
    all_carbon_nodes, all_cc_edges, _ = carbon_graph(atomic_numbers, bonds)
    pi_nodes, pi_source = choose_pi_atom_indices(atomic_numbers, coords, bonds, pi_atom_indices)
    carbon_nodes, cc_edges, adjacency = carbon_graph(atomic_numbers, bonds, allowed_nodes=pi_nodes)
    labels = fragment_labels(carbon_nodes, cc_edges)
    all_labels = fragment_labels(all_carbon_nodes, all_cc_edges)

    cc_lengths = [d for _, _, d in cc_edges]
    median = statistics.median(cc_lengths) if cc_lengths else math.nan
    short = [d for d in cc_lengths if d <= median] if cc_lengths else []
    long = [d for d in cc_lengths if d > median] if cc_lengths else []
    adjacent_delta = []
    for node, neighbors in adjacency.items():
        lengths = [d for _, d in neighbors]
        for i in range(len(lengths)):
            for j in range(i + 1, len(lengths)):
                adjacent_delta.append(abs(lengths[i] - lengths[j]))

    cycles = fundamental_cycle_lengths(carbon_nodes, cc_edges, adjacency)
    cycle_ranges = [max(c) - min(c) for c in cycles if c]
    cycle_alt = []
    for cyc in cycles:
        if len(cyc) > 1:
            cycle_alt.append(float(np.mean([abs(cyc[i] - cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))])))

    features: dict[str, float | str | None] = dict(base or {})
    features.update(
        {
            "n_atoms": float(len(atomic_numbers)),
            "n_carbons": float(sum(1 for z in atomic_numbers if z == 6)),
            "n_hydrogens": float(sum(1 for z in atomic_numbers if z == 1)),
            "n_detected_bonds": float(len(bonds)),
            "n_all_cc_bonds": float(len(all_cc_edges)),
            "n_cc_bonds": float(len(cc_edges)),
            "n_pi_carbons": float(len(carbon_nodes)),
            "n_sp3_like_carbons": float(len(all_carbon_nodes) - len(carbon_nodes)),
            "n_pi_cc_bonds": float(len(cc_edges)),
            "pi_carbon_fraction": float(len(carbon_nodes) / len(all_carbon_nodes)) if all_carbon_nodes else math.nan,
            "pi_atom_source": pi_source,
            "n_carbon_fragments": float(len(set(labels.values())) if labels else 0),
            "cc_short_mean": float(np.mean(short)) if short else math.nan,
            "cc_long_mean": float(np.mean(long)) if long else math.nan,
            "global_bla_long_minus_short": float(np.mean(long) - np.mean(short)) if short and long else math.nan,
            "global_bla_abs_adjacent_mean": float(np.mean(adjacent_delta)) if adjacent_delta else math.nan,
            "global_bla_abs_adjacent_max": float(np.max(adjacent_delta)) if adjacent_delta else math.nan,
        }
    )
    features.update(stats(cc_lengths, "cc_length"))
    features.update(stats(cycle_ranges, "cycle_bla_range"))
    features.update(stats(cycle_alt, "cycle_adjacent_bla"))

    for frag in sorted(set(labels.values())):
        frag_edges = [(i, j, d) for i, j, d in cc_edges if labels.get(i) == frag and labels.get(j) == frag]
        lengths = [d for _, _, d in frag_edges]
        prefix = f"frag{frag + 1}_cc_length"
        features.update(stats(lengths, prefix))
        if lengths:
            med = statistics.median(lengths)
            f_short = [d for d in lengths if d <= med]
            f_long = [d for d in lengths if d > med]
            features[f"frag{frag + 1}_bla_long_minus_short"] = (
                float(np.mean(f_long) - np.mean(f_short)) if f_short and f_long else math.nan
            )

    features.update(stacking_descriptors(coords, labels if labels else all_labels))
    features.update(carbon_distance_fingerprint(coords, labels if labels else all_labels))
    features.update(intermonomer_cc_sigma_descriptors(atomic_numbers, coords))
    return features


def intermonomer_cc_sigma_descriptors(
    atomic_numbers: list[int], coords: np.ndarray, cutoff: float = INTERMONOMER_CC_SIGMA_CUTOFF
) -> dict[str, float | str]:
    """Audit short C-C contacts between monomers without requiring atom order.

    For normal non-covalent dimers, monomers are identified as the two largest
    covalent fragments. If connectivity cannot separate two fragments, the
    ordered-halves convention is retained as a last-resort audit fallback.
    """
    out: dict[str, float | str] = {
        "intermonomer_cc_sigma_cutoff_ang": float(cutoff),
        "n_intermonomer_cc_sigma_bonds": math.nan,
        "has_intermonomer_cc_sigma_bond": math.nan,
        "min_intermonomer_cc_distance": math.nan,
        "intermonomer_halves_assumed": "no",
    }
    bonds = build_bonds(atomic_numbers, coords)
    nodes = list(range(len(atomic_numbers)))
    labels = fragment_labels(nodes, bonds)
    fragments = sorted(
        ({label for label in labels.values()}),
        key=lambda label: sum(1 for idx, value in labels.items() if value == label),
        reverse=True,
    )
    if len(fragments) >= 2:
        frag_a = {idx for idx, label in labels.items() if label == fragments[0]}
        frag_b = {idx for idx, label in labels.items() if label == fragments[1]}
        first_carbons = [idx for idx in frag_a if atomic_numbers[idx] == 6]
        second_carbons = [idx for idx in frag_b if atomic_numbers[idx] == 6]
        distances = [norm(coords[i] - coords[j]) for i in first_carbons for j in second_carbons]
        if distances:
            n_sigma = sum(1 for distance in distances if distance <= cutoff)
            out.update(
                {
                    "n_intermonomer_cc_sigma_bonds": float(n_sigma),
                    "has_intermonomer_cc_sigma_bond": float(n_sigma > 0),
                    "min_intermonomer_cc_distance": float(min(distances)),
                    "intermonomer_halves_assumed": "connectivity_fragments",
                }
            )
            return out

    bridge = detect_intermonomer_cc_bridge(atomic_numbers, coords, bonds, cutoff)
    if bridge is not None:
        out.update(bridge)
        return out

    if len(atomic_numbers) < 2 or len(atomic_numbers) % 2:
        return out

    midpoint = len(atomic_numbers) // 2
    first = atomic_numbers[:midpoint]
    second = atomic_numbers[midpoint:]
    if sorted(first) != sorted(second):
        return out

    first_carbons = [idx for idx, z in enumerate(first) if z == 6]
    second_carbons = [midpoint + idx for idx, z in enumerate(second) if z == 6]
    distances = [norm(coords[i] - coords[j]) for i in first_carbons for j in second_carbons]
    if not distances:
        return out

    n_sigma = sum(1 for distance in distances if distance <= cutoff)
    out.update(
        {
            "n_intermonomer_cc_sigma_bonds": float(n_sigma),
            "has_intermonomer_cc_sigma_bond": float(n_sigma > 0),
            "min_intermonomer_cc_distance": float(min(distances)),
            "intermonomer_halves_assumed": "ordered_equal_halves",
        }
    )
    return out


def composition_signature(atomic_numbers: list[int], nodes: set[int]) -> Counter[int]:
    return Counter(atomic_numbers[idx] for idx in nodes)


def detect_intermonomer_cc_bridge(
    atomic_numbers: list[int],
    coords: np.ndarray,
    bonds: list[tuple[int, int, float]],
    cutoff: float,
) -> dict[str, float | str] | None:
    """Detect a C-C bridge whose removal recovers two composition-matched monomers."""
    n_atoms = len(atomic_numbers)
    if n_atoms < 2:
        return None

    bridge_hits: list[tuple[int, int, float]] = []
    candidates = [(i, j, d) for i, j, d in bonds if atomic_numbers[i] == 6 and atomic_numbers[j] == 6 and d <= cutoff]
    for i, j, distance in candidates:
        trial_edges = [(a, b, d) for a, b, d in bonds if not ((a == i and b == j) or (a == j and b == i))]
        labels = fragment_labels(list(range(n_atoms)), trial_edges)
        fragments = sorted(
            set(labels.values()),
            key=lambda label: sum(1 for value in labels.values() if value == label),
            reverse=True,
        )
        if len(fragments) < 2:
            continue
        frag_a = {idx for idx, label in labels.items() if label == fragments[0]}
        frag_b = {idx for idx, label in labels.items() if label == fragments[1]}
        other_count = sum(1 for label in labels.values() if label not in fragments[:2])
        if len(frag_a) < 0.25 * n_atoms or len(frag_b) < 0.25 * n_atoms or other_count > 0.1 * n_atoms:
            continue
        if composition_signature(atomic_numbers, frag_a) != composition_signature(atomic_numbers, frag_b):
            continue
        bridge_hits.append((i, j, distance))

    if not bridge_hits:
        return None
    return {
        "n_intermonomer_cc_sigma_bonds": float(len(bridge_hits)),
        "has_intermonomer_cc_sigma_bond": 1.0,
        "min_intermonomer_cc_distance": float(min(hit[2] for hit in bridge_hits)),
        "intermonomer_halves_assumed": "connectivity_bridge_removal",
    }


def bond_cutoff(z1: int, z2: int, scale: float = 1.23, slack: float = 0.08) -> float:
    r1 = COVALENT_RADII.get(z1, 0.77)
    r2 = COVALENT_RADII.get(z2, 0.77)
    return scale * (r1 + r2) + slack


def build_bonds(atomic_numbers: list[int], coords: np.ndarray) -> list[tuple[int, int, float]]:
    bonds: list[tuple[int, int, float]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = norm(coords[i] - coords[j])
            if 0.55 <= d <= bond_cutoff(atomic_numbers[i], atomic_numbers[j]):
                bonds.append((i, j, d))
    return bonds


def carbon_graph(
    atomic_numbers: list[int], bonds: list[tuple[int, int, float]], allowed_nodes: list[int] | None = None
) -> tuple[list[int], list[tuple[int, int, float]], dict[int, list[tuple[int, float]]]]:
    allowed_set = set(allowed_nodes) if allowed_nodes is not None else None
    carbon_indices = [i for i, z in enumerate(atomic_numbers) if z == 6 and (allowed_set is None or i in allowed_set)]
    carbon_set = set(carbon_indices)
    cc_edges = [(i, j, d) for i, j, d in bonds if i in carbon_set and j in carbon_set]
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in carbon_indices}
    for i, j, d in cc_edges:
        adjacency[i].append((j, d))
        adjacency[j].append((i, d))
    return carbon_indices, cc_edges, adjacency


def fragment_labels(nodes: list[int], edges: list[tuple[int, int, float]]) -> dict[int, int]:
    if not nodes:
        return {}
    idx = {node: pos for pos, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for i, j, _ in edges:
        rows.extend([idx[i], idx[j]])
        cols.extend([idx[j], idx[i]])
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(matrix, directed=False, return_labels=True)
    return {node: int(labels[idx[node]]) for node in nodes}


def shortest_tree_path(parent: dict[int, int | None], start: int, end: int) -> list[int]:
    ancestors: dict[int, int] = {}
    node: int | None = start
    depth = 0
    while node is not None:
        ancestors[node] = depth
        node = parent[node]
        depth += 1
    node = end
    tail: list[int] = []
    while node not in ancestors:
        tail.append(node)
        node = parent[node]
        if node is None:
            return []
    lca = node
    head: list[int] = []
    node = start
    while node != lca:
        head.append(node)
        node = parent[node]
    head.append(lca)
    return head + tail[::-1]


def fundamental_cycle_lengths(
    nodes: list[int], edges: list[tuple[int, int, float]], adjacency: dict[int, list[tuple[int, float]]]
) -> list[list[float]]:
    edge_lookup = {frozenset((i, j)): d for i, j, d in edges}
    seen: set[int] = set()
    parent: dict[int, int | None] = {}
    tree_edges: set[frozenset[int]] = set()
    cycles: list[list[float]] = []

    for root in nodes:
        if root in seen:
            continue
        parent[root] = None
        seen.add(root)
        queue: deque[int] = deque([root])
        while queue:
            u = queue.popleft()
            for v, _ in adjacency[u]:
                e = frozenset((u, v))
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    tree_edges.add(e)
                    queue.append(v)
                elif parent[u] != v and e not in tree_edges:
                    path = shortest_tree_path(parent, u, v)
                    if len(path) >= 3:
                        lengths = [edge_lookup[frozenset((path[k], path[k + 1]))] for k in range(len(path) - 1)]
                        lengths.append(edge_lookup[e])
                        signature = tuple(sorted(round(x, 5) for x in lengths))
                        if not any(tuple(sorted(round(y, 5) for y in c)) == signature for c in cycles):
                            cycles.append(lengths)
    return cycles


def plane_normal(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    length = norm(normal)
    if length == 0:
        return None
    return normal / length


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    u = axis - float(np.dot(axis, normal)) * normal
    u = u / norm(u)
    v = np.cross(normal, u)
    v = v / norm(v)
    return u, v


def project_points(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> list[tuple[float, float]]:
    projected = []
    for point in points:
        vec = point - origin
        projected.append((float(np.dot(vec, u)), float(np.dot(vec, v))))
    return projected


def convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set((round(x, 10), round(y, 10)) for x, y in points))
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return total * 0.5


def convex_intersection_area(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> float:
    if len(subject) < 3 or len(clip) < 3:
        return 0.0
    if polygon_signed_area(clip) < 0:
        clip = list(reversed(clip))

    def inside(point: tuple[float, float], edge_a: tuple[float, float], edge_b: tuple[float, float]) -> bool:
        return (edge_b[0] - edge_a[0]) * (point[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0]) >= -1e-10

    def intersection(
        start: tuple[float, float],
        end: tuple[float, float],
        edge_a: tuple[float, float],
        edge_b: tuple[float, float],
    ) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end
        x3, y3 = edge_a
        x4, y4 = edge_b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return end
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (float(px), float(py))

    output = subject
    if polygon_signed_area(output) < 0:
        output = list(reversed(output))
    for idx, edge_a in enumerate(clip):
        edge_b = clip[(idx + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        start = input_poly[-1]
        for end in input_poly:
            if inside(end, edge_a, edge_b):
                if not inside(start, edge_a, edge_b):
                    output.append(intersection(start, end, edge_a, edge_b))
                output.append(end)
            elif inside(start, edge_a, edge_b):
                output.append(intersection(start, end, edge_a, edge_b))
            start = end
    return polygon_area(output)


def projected_pi_overlap_descriptors(pts_a: np.ndarray, pts_b: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    origin = 0.5 * (np.mean(pts_a, axis=0) + np.mean(pts_b, axis=0))
    u, v = plane_basis(normal)
    hull_a = convex_hull_2d(project_points(pts_a, origin, u, v))
    hull_b = convex_hull_2d(project_points(pts_b, origin, u, v))
    area_a = polygon_area(hull_a)
    area_b = polygon_area(hull_b)
    overlap = convex_intersection_area(hull_a, hull_b)
    mean_area = 0.5 * (area_a + area_b)
    min_area = min(area_a, area_b)
    return {
        "pi_projected_area_frag1": float(area_a),
        "pi_projected_area_frag2": float(area_b),
        "pi_projected_area_mean": float(mean_area),
        "pi_projected_area_sum": float(area_a + area_b),
        "pi_projected_area_min": float(min_area),
        "pi_projected_area_max": float(max(area_a, area_b)),
        "pi_projected_area_asymmetry_abs": float(abs(area_a - area_b)),
        "projected_pi_overlap_area": float(overlap),
        "projected_pi_overlap_fraction_min": float(overlap / min_area) if min_area > 0 else math.nan,
        "projected_pi_overlap_fraction_mean": float(overlap / mean_area) if mean_area > 0 else math.nan,
    }


def stacking_descriptors(coords: np.ndarray, labels: dict[int, int]) -> dict[str, float]:
    fragments = sorted(
        set(labels.values()),
        key=lambda label: sum(1 for value in labels.values() if value == label),
        reverse=True,
    )
    if len(fragments) < 2:
        return {}
    frag_nodes = [[node for node, label in labels.items() if label == frag] for frag in fragments[:2]]
    pts_a = coords[frag_nodes[0]]
    pts_b = coords[frag_nodes[1]]
    ca = np.mean(pts_a, axis=0)
    cb = np.mean(pts_b, axis=0)
    center_vec = cb - ca
    center_distance = norm(center_vec)
    na = plane_normal(pts_a)
    nb = plane_normal(pts_b)
    out: dict[str, float] = {"fragment_center_distance": center_distance}
    if na is not None and nb is not None:
        normal_angle = math.degrees(math.acos(float(np.clip(abs(np.dot(na, nb)), -1.0, 1.0))))
        avg_normal = na + (nb if np.dot(na, nb) >= 0 else -nb)
        avg_normal = avg_normal / norm(avg_normal)
        pi_distance = abs(float(np.dot(center_vec, avg_normal)))
        slip = math.sqrt(max(center_distance**2 - pi_distance**2, 0.0))
        out.update(
            {
                "interplane_distance": pi_distance,
                "lateral_slip": slip,
                "plane_normal_angle_deg": normal_angle,
            }
        )
        out.update(projected_pi_overlap_descriptors(pts_a, pts_b, avg_normal))
    return out


def carbon_distance_fingerprint(coords: np.ndarray, labels: dict[int, int]) -> dict[str, float]:
    fragments = sorted(
        set(labels.values()),
        key=lambda label: sum(1 for value in labels.values() if value == label),
        reverse=True,
    )
    if len(fragments) >= 2:
        keep_fragments = set(fragments[:2])
        labels = {node: label for node, label in labels.items() if label in keep_fragments}
    carbon_nodes = sorted(labels)
    out: dict[str, float] = {}
    if len(carbon_nodes) < 2:
        return out

    all_distances = []
    interfragment_distances = []
    for pos, i in enumerate(carbon_nodes):
        for j in carbon_nodes[pos + 1 :]:
            d = norm(coords[i] - coords[j])
            all_distances.append(d)
            if labels.get(i) != labels.get(j):
                interfragment_distances.append(d)

    all_bins = np.linspace(1.0, 9.0, 17)
    inter_bins = np.linspace(2.0, 9.0, 15)
    all_hist, _ = np.histogram(all_distances, bins=all_bins, density=True)
    inter_hist, _ = np.histogram(interfragment_distances, bins=inter_bins, density=True)

    for idx, value in enumerate(np.nan_to_num(all_hist), start=1):
        out[f"carbon_distance_hist_{idx:02d}"] = float(value)
    for idx, value in enumerate(np.nan_to_num(inter_hist), start=1):
        out[f"interfragment_carbon_distance_hist_{idx:02d}"] = float(value)
    out.update(stats(interfragment_distances, "interfragment_carbon_distance"))
    return out


def bla_descriptors(record: dict[str, Any], pi_atom_indices: list[int] | None = None) -> dict[str, float | str | None]:
    atomic_numbers, coords = read_atoms(record)
    features = descriptor_from_atoms(
        atomic_numbers,
        coords,
        {
        "id": record.get("id"),
        "charge": record.get("charge"),
        "multiplicity": record.get("multiplicity"),
        "scf_energy_hartree": record.get("energy"),
        },
        pi_atom_indices=pi_atom_indices,
    )
    features.update(find_gap_features(record, atomic_numbers))
    return features


def recursive_items(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield prefix + key, value
            yield from recursive_items(value, prefix + key + ".")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield f"{prefix}{i}.", value
            yield from recursive_items(value, f"{prefix}{i}.")


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_gap_features(record: dict[str, Any], atomic_numbers: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    scalars: dict[str, float] = {}
    arrays: dict[str, list[float]] = {}

    for path, value in recursive_items(record):
        low = path.lower()
        if isinstance(value, (int, float)):
            f = as_float(value)
            if f is not None:
                scalars[low] = f
        elif isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            arrays[low] = [float(x) for x in value]

    for key, value in scalars.items():
        if any(token in key for token in ["homo_lumo_gap", "homolumogap", "band_gap", "gap_ev"]):
            out["gap_ev"] = value
            return out
        if "gap" in key and "energy_grad" not in key and abs(value) < 100:
            out.setdefault("gap_ev", value)

    homo = next((v for k, v in scalars.items() if "homo" in k and "lumo" not in k), None)
    lumo = next((v for k, v in scalars.items() if "lumo" in k and "homo" not in k), None)
    if homo is not None:
        out["homo_energy_raw"] = homo
    if lumo is not None:
        out["lumo_energy_raw"] = lumo
    if homo is not None and lumo is not None:
        gap = lumo - homo
        out["gap_ev"] = gap * HARTREE_TO_EV if abs(gap) < 2 else gap
        return out

    orbital_keys = [k for k in arrays if "orbital" in k and ("energ" in k or "eigen" in k)]
    if orbital_keys:
        energies = arrays[orbital_keys[0]]
        electron_count = sum(atomic_numbers) - int(record.get("charge") or 0)
        homo_idx = electron_count // 2 - 1
        lumo_idx = homo_idx + 1
        if 0 <= homo_idx < len(energies) and 0 <= lumo_idx < len(energies):
            homo_e = energies[homo_idx]
            lumo_e = energies[lumo_idx]
            out["homo_energy_raw"] = homo_e
            out["lumo_energy_raw"] = lumo_e
            gap = lumo_e - homo_e
            out["gap_ev"] = gap * HARTREE_TO_EV if abs(gap) < 2 else gap
    return out


def merge_target_csv(
    rows: list[dict[str, Any]], target_csv: Path | None, id_column: str, value_column: str, target_column: str
) -> list[dict[str, Any]]:
    if target_csv is None:
        return rows
    targets = pd.read_csv(target_csv)
    if id_column not in targets.columns or value_column not in targets.columns:
        raise ValueError(f"{target_csv} must contain '{id_column}' and '{value_column}' columns.")
    target_map = dict(zip(targets[id_column].astype(str), targets[value_column]))
    merged = []
    for row in rows:
        new_row = dict(row)
        key = str(new_row.get("id"))
        if key in target_map:
            new_row[target_column] = target_map[key]
        merged.append(new_row)
    return merged


def add_interaction_target(
    rows: list[dict[str, Any]],
    mode: str,
    target_column: str,
    monomer_energy_hartree: float | None = None,
    monomer_a_energy_hartree: float | None = None,
    monomer_b_energy_hartree: float | None = None,
) -> list[dict[str, Any]]:
    if mode == "none":
        return rows

    energies = [as_float(row.get("scf_energy_hartree")) for row in rows]
    if any(value is None for value in energies):
        raise ValueError("SCF dimer energies are required to compute interaction targets.")
    dimer_energies = np.asarray([float(value) for value in energies], dtype=float)

    if mode == "monomer":
        if monomer_energy_hartree is not None:
            reference = 2.0 * monomer_energy_hartree
        elif monomer_a_energy_hartree is not None and monomer_b_energy_hartree is not None:
            reference = monomer_a_energy_hartree + monomer_b_energy_hartree
        else:
            raise ValueError(
                "--interaction-mode monomer requires --monomer-energy-hartree "
                "or both --monomer-a-energy-hartree and --monomer-b-energy-hartree."
            )
        target_values = (dimer_energies - reference) * HARTREE_TO_KCAL_MOL
    elif mode == "relative_to_min":
        target_values = (dimer_energies - float(np.min(dimer_energies))) * HARTREE_TO_KCAL_MOL
    elif mode == "relative_to_max_distance":
        distances = np.asarray([as_float(row.get("fragment_center_distance")) for row in rows], dtype=float)
        if np.isnan(distances).all():
            raise ValueError("fragment_center_distance is required for relative_to_max_distance.")
        reference = dimer_energies[int(np.nanargmax(distances))]
        target_values = (dimer_energies - reference) * HARTREE_TO_KCAL_MOL
    else:
        raise ValueError(f"Unknown interaction mode: {mode}")

    out = []
    for row, target_value in zip(rows, target_values):
        new_row = dict(row)
        new_row[target_column] = float(target_value)
        out.append(new_row)
    return out


def extract_final_monomer_energy(json_path: Path) -> float:
    with json_path.open() as handle:
        data = json.load(handle)

    geometries = data.get("Geometries") if isinstance(data, dict) else None
    if isinstance(geometries, list) and geometries:
        final_geometry = geometries[-1]
        if isinstance(final_geometry, dict):
            single_point = final_geometry.get("Single_Point_Data")
            if isinstance(single_point, dict) and as_float(single_point.get("FinalEnergy")) is not None:
                return float(single_point["FinalEnergy"])
            thermo = final_geometry.get("THERMOCHEMISTRY_Energies")
            if isinstance(thermo, list) and thermo and as_float(thermo[-1].get("elEnergy")) is not None:
                return float(thermo[-1]["elEnergy"])
            energy = final_geometry.get("Energy")
            if isinstance(energy, list) and energy:
                total = energy[-1].get("totalEnergy") if isinstance(energy[-1], dict) else None
                if isinstance(total, list) and total and as_float(total[-1]) is not None:
                    return float(total[-1])
                if as_float(total) is not None:
                    return float(total)

    candidates: list[tuple[str, float]] = []
    preferred_tokens = ["finalenergy", "final_energy", "final single point", "elenergy", "totalenergy", "total_energy"]
    for path, value in recursive_items(data):
        if isinstance(value, (int, float)):
            low = path.lower().replace(" ", "")
            if any(token.replace(" ", "") in low for token in preferred_tokens):
                number = float(value)
                if -10000.0 < number < 0.0:
                    candidates.append((path, number))
    if candidates:
        return candidates[-1][1]
    raise ValueError(f"Could not extract a final monomer energy from {json_path}.")


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    columns = sorted({key for row in rows for key in row})
    preferred = ["id", "gap_ev", "homo_energy_raw", "lumo_energy_raw", "scf_energy_hartree"]
    columns = [c for c in preferred if c in columns] + [c for c in columns if c not in preferred]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def make_models(random_state: int) -> dict[str, Any]:
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 41)))]),
    }


def descriptor_groups(columns: list[str]) -> dict[str, list[str]]:
    stacking_terms = {"fragment_center_distance", "interplane_distance", "lateral_slip", "plane_normal_angle_deg"}

    def is_bla(col: str) -> bool:
        return (
            "bla" in col
            or col.startswith("cc_length_")
            or col.startswith("cc_short")
            or col.startswith("cc_long")
            or col.startswith("frag")
            or col.startswith("cycle_")
        ) and not col.endswith("_count")

    def is_distance_fp(col: str) -> bool:
        return col.startswith("carbon_distance_hist_") or col.startswith("interfragment_carbon_distance")

    groups = {
        "stacking_only": [col for col in columns if col in stacking_terms],
        "bla_only": [col for col in columns if is_bla(col)],
        "distance_fingerprint_only": [col for col in columns if is_distance_fp(col)],
        "stacking_plus_bla": [col for col in columns if col in stacking_terms or is_bla(col)],
        "stacking_plus_distance_fp": [col for col in columns if col in stacking_terms or is_distance_fp(col)],
        "stacking_bla_distance_fp": [col for col in columns if col in stacking_terms or is_bla(col) or is_distance_fp(col)],
        "all_nonleaky_descriptors": columns,
    }
    return {name: feats for name, feats in groups.items() if feats}


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_model_comparison(summary: pd.DataFrame, output_prefix: Path) -> None:
    best = summary.sort_values("cv_mae").drop_duplicates("descriptor_group").sort_values("cv_mae")
    label_map = {
        "stacking_plus_distance_fp": "Stacking + distance FP",
        "distance_fingerprint_only": "Distance FP only",
        "stacking_bla_distance_fp": "Stacking + BLA + distance FP",
        "all_nonleaky_descriptors": "All nonleaky descriptors",
        "stacking_only": "Stacking only",
        "stacking_plus_bla": "Stacking + BLA",
        "bla_only": "BLA only",
    }
    labels = [label_map.get(name, name.replace("_", " ")) for name in best["descriptor_group"]]
    y_pos = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.barh(y_pos, best["cv_mae"], color=PLOT_COLORS["blue"], height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated MAE (kcal mol$^{-1}$)")
    ax.set_title("Descriptor Family Comparison")
    ax.grid(axis="x", color=PLOT_COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    for idx, (_, row) in enumerate(best.iterrows()):
        ax.text(
            row["cv_mae"] + 0.035,
            idx,
            f"$R^2$={row['cv_r2']:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color=PLOT_COLORS["gray"],
        )
    ax.set_xlim(0, float(best["cv_mae"].max()) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_descriptor_group_comparison"))


def plot_predicted_vs_actual(y: pd.Series, pred: np.ndarray, title: str, output_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(y, pred, s=22, color=PLOT_COLORS["blue"], alpha=0.82, edgecolor="white", linewidth=0.35)
    lower = float(min(np.min(y), np.min(pred)))
    upper = float(max(np.max(y), np.max(pred)))
    pad = 0.05 * (upper - lower) if upper > lower else 1.0
    ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color=PLOT_COLORS["gray"], linewidth=1.1)
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(lower - pad, upper + pad)
    ax.set_xlabel("DFT interaction energy (kcal mol$^{-1}$)")
    ax.set_ylabel("ML-predicted interaction energy (kcal mol$^{-1}$)")
    ax.set_title(title)
    ax.grid(color=PLOT_COLORS["light_gray"], linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_best_predicted_vs_dft"))


def plot_feature_importance(importance: pd.DataFrame, output_prefix: Path) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.barh(top["feature"], top["importance_mae"], color=PLOT_COLORS["green"])
    ax.set_xlabel("Permutation importance: MAE increase (kcal mol$^{-1}$)")
    ax.set_title("Top Descriptor Contributions")
    ax.grid(axis="x", color=PLOT_COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_feature_importance"))


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, output_prefix: Path) -> None:
    candidate_cols = [
        target_col,
        "global_bla_long_minus_short",
        "global_bla_abs_adjacent_mean",
        "global_bla_abs_adjacent_max",
        "frag1_bla_long_minus_short",
        "frag2_bla_long_minus_short",
        "cycle_adjacent_bla_mean",
        "fragment_center_distance",
        "interplane_distance",
        "lateral_slip",
        "plane_normal_angle_deg",
    ]
    cols = [col for col in candidate_cols if col in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.1, 5.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([c.replace("_", " ") for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title("Descriptor Correlation Map")
    fig.tight_layout()
    save_figure(fig, output_prefix.with_name(output_prefix.name + "_correlation_heatmap"))


def markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_float_dtype(clean[col]):
            clean[col] = clean[col].map(lambda value: f"{value:.3f}")
    headers = [str(col) for col in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in clean.columns) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    output_prefix: Path,
    target_col: str,
    monomer_energy: float | None,
    summary: pd.DataFrame,
    target_stats: pd.Series,
    best_row: pd.Series,
    top_features: pd.DataFrame,
) -> None:
    lines = [
        "# BLA/Geometry Interaction-Energy ML Summary",
        "",
        f"Target column: `{target_col}`",
        f"Samples: {int(best_row['n_samples'])}",
        f"Best descriptor group: `{best_row['descriptor_group']}`",
        f"Best model: `{best_row['model']}`",
        "",
    ]
    if monomer_energy is not None:
        lines.append(f"Monomer energy used: `{monomer_energy:.15f}` Hartree")
        lines.append("")
    lines.extend(
        [
            "## Target Range",
            "",
            f"- Minimum: {target_stats['min']:.3f} kcal/mol",
            f"- Mean: {target_stats['mean']:.3f} kcal/mol",
            f"- Maximum: {target_stats['max']:.3f} kcal/mol",
            "",
            "## Best Cross-Validated Performance",
            "",
            f"- MAE: {best_row['cv_mae']:.3f} kcal/mol",
            f"- RMSE: {best_row['cv_rmse']:.3f} kcal/mol",
            f"- R2: {best_row['cv_r2']:.3f}",
            "",
            "## Descriptor Group Leaderboard",
            "",
            markdown_table(
                summary.sort_values("cv_mae")
                .drop_duplicates("descriptor_group")[["descriptor_group", "model", "cv_mae", "cv_rmse", "cv_r2", "n_features"]]
            ),
            "",
            "## Top Features",
            "",
            markdown_table(top_features.head(15)),
            "",
            "Generated figures are saved as both `.png` and `.pdf` for manuscript use.",
        ]
    )
    output_prefix.with_name(output_prefix.name + "_analysis_summary.md").write_text("\n".join(lines))


def train_models(
    feature_csv: Path, target_col: str, output_prefix: Path, random_state: int, monomer_energy: float | None = None
) -> None:
    df = pd.read_csv(feature_csv)
    if target_col not in df.columns or df[target_col].notna().sum() < 20:
        print(f"No usable '{target_col}' target found; wrote descriptors only.")
        return

    configure_plot_style()
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.dropna(axis=1, how="all")
    y = numeric[target_col]
    leakage_columns = {
        "scf_energy_hartree",
        "homo_energy_raw",
        "lumo_energy_raw",
        "gap_ev",
        "interaction_energy_kcal_mol",
        "interaction_energy_relative_kcal_mol",
    }
    drop_columns = [col for col in leakage_columns | {target_col} if col in numeric.columns]
    x = numeric.drop(columns=drop_columns)
    keep = y.notna()
    x = x.loc[keep]
    y = y.loc[keep]
    x = x.fillna(x.median(numeric_only=True))
    df_model = df.loc[keep].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    cv = KFold(n_splits=min(5, len(x)), shuffle=True, random_state=random_state)
    groups = descriptor_groups(list(x.columns))
    rows = []
    report: dict[str, Any] = {"target": target_col, "n_samples": int(len(x)), "descriptor_groups": {}}
    best: dict[str, Any] = {}
    best_score = float("inf")

    for group_name, features in groups.items():
        xg = x[features]
        xg_train = x_train[features]
        xg_test = x_test[features]
        report["descriptor_groups"][group_name] = {"n_features": len(features), "models": {}}
        for model_name, model in make_models(random_state).items():
            pred_cv = cross_val_predict(model, xg, y, cv=cv)
            model.fit(xg_train, y_train)
            pred_test = model.predict(xg_test)
            cv_metrics = metric_dict(y, pred_cv)
            test_metrics = metric_dict(y_test, pred_test)
            result = {
                "descriptor_group": group_name,
                "model": model_name,
                "n_samples": int(len(xg)),
                "n_features": int(len(features)),
                "cv_mae": cv_metrics["mae"],
                "cv_rmse": cv_metrics["rmse"],
                "cv_r2": cv_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
            rows.append(result)
            report["descriptor_groups"][group_name]["models"][model_name] = result
            if result["cv_mae"] < best_score:
                best_score = result["cv_mae"]
                best = {
                    "descriptor_group": group_name,
                    "model_name": model_name,
                    "features": features,
                    "cv_pred": pred_cv,
                    "model": model,
                    "result": result,
                }

    summary = pd.DataFrame(rows).sort_values(["cv_mae", "test_mae"])
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_model_summary.csv"), index=False)
    assert best
    best_model = best["model"]
    best_features = best["features"]
    best_model.fit(x[best_features], y)
    report["best_model"] = best["model_name"]
    report["best_descriptor_group"] = best["descriptor_group"]
    report["best_metrics"] = best["result"]

    perm = permutation_importance(
        best_model,
        x[best_features],
        y,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    importance = (
        pd.DataFrame({"feature": best_features, "importance_mae": perm.importances_mean})
        .sort_values("importance_mae", ascending=False)
        .head(30)
    )
    importance.to_csv(output_prefix.with_name(output_prefix.name + "_feature_importance.csv"), index=False)

    with output_prefix.with_name(output_prefix.name + "_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": best_model,
                "features": list(best_features),
                "target": target_col,
                "descriptor_group": best["descriptor_group"],
            },
            handle,
        )
    with output_prefix.with_name(output_prefix.name + "_model_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    predictions = pd.DataFrame(
        {
            "id": df_model["id"] if "id" in df_model.columns else np.arange(len(y)),
            "dft_target": y,
            "cv_prediction": best["cv_pred"],
            "residual": y.to_numpy() - best["cv_pred"],
        }
    )
    predictions.to_csv(output_prefix.with_name(output_prefix.name + "_best_cv_predictions.csv"), index=False)
    plot_model_comparison(summary, output_prefix)
    plot_predicted_vs_actual(
        y,
        best["cv_pred"],
        f"{best['descriptor_group']} / {best['model_name']}",
        output_prefix,
    )
    plot_feature_importance(importance, output_prefix)
    plot_correlation_heatmap(df_model, target_col, output_prefix)
    write_markdown_summary(
        output_prefix,
        target_col,
        monomer_energy,
        summary,
        y.describe(),
        pd.Series(best["result"]),
        importance,
    )

    print(f"Best descriptor group: {best['descriptor_group']}")
    print(f"Best model: {best['model_name']}")
    print(json.dumps(best["result"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="PLY_all_molecules.json", help="Input DFT JSON file.")
    parser.add_argument("--features", default="PLY_BLA_descriptors.csv", help="Output descriptor CSV.")
    parser.add_argument("--target", default="gap_ev", help="Target column for ML training.")
    parser.add_argument("--target-csv", default=None, help="Optional CSV containing externally computed HOMO-LUMO gaps.")
    parser.add_argument("--target-id-column", default="id", help="ID column in --target-csv.")
    parser.add_argument("--target-value-column", default="gap_ev", help="Gap-value column in --target-csv.")
    parser.add_argument(
        "--interaction-mode",
        choices=["none", "monomer", "relative_to_min", "relative_to_max_distance"],
        default="none",
        help=(
            "Create an interaction-energy target from dimer SCF energies. "
            "'monomer' gives absolute E_int; the relative modes are useful when isolated monomer energies are unavailable."
        ),
    )
    parser.add_argument("--monomer-energy-hartree", type=float, default=None, help="Isolated monomer SCF energy for identical dimers.")
    parser.add_argument("--monomer-json", default=None, help="ORCA-style monomer JSON; final optimized electronic energy is extracted.")
    parser.add_argument("--monomer-a-energy-hartree", type=float, default=None, help="Isolated monomer A SCF energy.")
    parser.add_argument("--monomer-b-energy-hartree", type=float, default=None, help="Isolated monomer B SCF energy.")
    parser.add_argument(
        "--pi-atoms",
        default=None,
        help=(
            "Optional text file containing conjugated carbon atom indices for BLA. "
            "Supports comma/space separated values and ranges such as 1-13. "
            "If omitted, sp2-like carbons are detected automatically."
        ),
    )
    parser.add_argument("--model-prefix", default="PLY_BLA_gap", help="Prefix for model/report outputs.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open() as handle:
        data = json.load(handle)
    records = flatten_records(data)

    pi_atom_indices = read_pi_atom_indices(Path(args.pi_atoms)) if args.pi_atoms else None

    rows = []
    for record in records:
        rows.append(bla_descriptors(record, pi_atom_indices=pi_atom_indices))
    rows = merge_target_csv(
        rows,
        Path(args.target_csv) if args.target_csv else None,
        args.target_id_column,
        args.target_value_column,
        args.target,
    )
    monomer_energy = args.monomer_energy_hartree
    if args.monomer_json:
        monomer_energy = extract_final_monomer_energy(Path(args.monomer_json))
        print(f"Extracted monomer energy: {monomer_energy:.15f} Hartree from {args.monomer_json}")

    rows = add_interaction_target(
        rows,
        args.interaction_mode,
        args.target,
        monomer_energy,
        args.monomer_a_energy_hartree,
        args.monomer_b_energy_hartree,
    )

    features_path = Path(args.features)
    write_csv(rows, features_path)
    print(f"Wrote {len(rows)} descriptor rows to {features_path}")
    print(f"Records with target '{args.target}': {sum(1 for row in rows if row.get(args.target) is not None)}")
    train_models(features_path, args.target, Path(args.model_prefix), args.random_state, monomer_energy)


if __name__ == "__main__":
    main()
