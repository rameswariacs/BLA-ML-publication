#!/usr/bin/env python3
"""Make a combined descriptor-distribution figure for the BLA-ML SI.

The figure is written as editable SVG using only pandas/numpy and standard
Python string output. This avoids local font-cache issues while keeping the
graphic fully editable in Illustrator, Inkscape, or PowerPoint.
"""

from __future__ import annotations

from pathlib import Path
import html

import numpy as np
import pandas as pd


CSV_PATH = Path(
    "/Users/ram/Documents/BLA-ML/LATEST-FOLDERS/WITHOUT-BS-FILTER/"
    "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/"
    "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv"
)
OUTDIR = Path(
    "/Users/ram/Documents/Pancake-Bond-Search/Script/outputs/"
    "key_descriptor_distributions"
)

SYSTEM_ORDER = [
    "phenalenyl",
    "olympicenyl",
    "fluorenyl",
    "CPBP",
    "phenalenyl_olympicenyl",
]
SYSTEM_LABELS = ["Ph", "Oly", "Flu", "CPBP", "Ph-Oly"]
SYSTEM_NAMES = {
    "phenalenyl": "Phenalenyl",
    "olympicenyl": "Olympicenyl",
    "fluorenyl": "Fluorenyl",
    "CPBP": "CPBP",
    "phenalenyl_olympicenyl": "Phenalenyl-olympicenyl",
}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A3D9A"]

PANELS = [
    {
        "letter": "a",
        "column": "stack_plane_normal_angle_deg",
        "title": "Plane alignment",
        "ylabel": "&#952;<tspan baseline-shift=\"sub\" font-size=\"70%\">plane</tspan> (deg)",
        "ylim": (0.0, 90.0),
        "tick_step": 30.0,
        "ref_lines": [(10.0, "10 deg")],
    },
    {
        "letter": "b",
        "column": "contact_min_cc_distance",
        "title": "Closest contact",
        "ylabel": "d<tspan baseline-shift=\"sub\" font-size=\"70%\">min</tspan> (&#197;)",
        "ylim": (2.2, 5.5),
        "tick_step": 0.8,
        "ref_lines": [(3.4, "3.4 A")],
    },
    {
        "letter": "c",
        "column": "mean_top10_closest_c_distances",
        "title": "Short-contact network",
        "ylabel": "d<tspan baseline-shift=\"sub\" font-size=\"70%\">10</tspan> (&#197;)",
        "ylim": (2.6, 6.2),
        "tick_step": 0.9,
        "ref_lines": [(3.4, "3.4 A")],
    },
    {
        "letter": "d",
        "column": "projected_pi_overlap_fraction_min",
        "title": "Normalized overlap",
        "ylabel": "f<tspan baseline-shift=\"sub\" font-size=\"70%\">ov,min</tspan>",
        "ylim": (0.0, 1.05),
        "tick_step": 0.25,
        "ref_lines": [],
    },
    {
        "letter": "e",
        "column": "pi_projected_area_mean",
        "title": "Mean projected pi-area",
        "ylabel": "S&#772;<tspan baseline-shift=\"sub\" font-size=\"70%\">&#960;</tspan> (&#197;<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>)",
        "ylim": None,
        "tick_step": None,
        "ref_lines": [],
    },
]


def kde(values: np.ndarray, grid: np.ndarray, ymin: float, ymax: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.zeros_like(grid)
    std = float(np.std(values, ddof=1))
    bandwidth = max((ymax - ymin) / 120.0, 1.06 * std * n ** (-1 / 5))
    if bandwidth == 0:
        bandwidth = (ymax - ymin) / 120.0
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (
        n * bandwidth * np.sqrt(2 * np.pi)
    )
    return density


def fmt_tick(value: float, step: float | None) -> str:
    if step is None:
        return f"{value:.0f}"
    if step < 0.3:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if step < 1:
        return f"{value:.1f}"
    return f"{value:.0f}"


def panel_limits(values_by_system: list[np.ndarray], panel: dict[str, object]) -> tuple[float, float, float]:
    if panel["ylim"] is not None:
        ymin, ymax = panel["ylim"]
        return float(ymin), float(ymax), float(panel["tick_step"])
    all_values = np.concatenate(values_by_system)
    ymin = 0.0
    ymax = float(np.ceil(np.percentile(all_values, 99.5) / 10.0) * 10.0)
    ymax = max(ymax, float(np.ceil(np.max(all_values) / 10.0) * 10.0))
    return ymin, ymax, 20.0


def add_text(svg: list[str], x: float, y: float, text: str, **attrs: object) -> None:
    attr = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    svg.append(f"<text x=\"{x:.2f}\" y=\"{y:.2f}\" {attr}>{text}</text>")


def make_figure(df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 900
    margin_x, margin_y = 82, 84
    gap_x, gap_y = 62, 92
    panel_w = (width - 2 * margin_x - 2 * gap_x) / 3
    panel_h = 300
    top_y = 100
    bottom_y = top_y + panel_h + gap_y
    panel_positions = [
        (margin_x, top_y),
        (margin_x + panel_w + gap_x, top_y),
        (margin_x + 2 * (panel_w + gap_x), top_y),
        (margin_x + 0.5 * (panel_w + gap_x), bottom_y),
        (margin_x + 1.5 * (panel_w + gap_x), bottom_y),
    ]

    rng = np.random.default_rng(42)
    summary_rows: list[dict[str, object]] = []
    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        "<style>"
        "text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}"
        ".title{font-size:26px;font-weight:700;}"
        ".paneltitle{font-size:20px;font-weight:700;}"
        ".letter{font-size:18px;font-weight:700;}"
        ".tick{font-size:15px;}"
        ".axis{font-size:18px;}"
        ".sys{font-size:15px;font-weight:600;}"
        ".note{font-size:15px;fill:#374151;}"
        "</style>"
    )
    add_text(
        svg,
        width / 2,
        42,
        "Key structural-descriptor distributions across radical-dimer systems",
        **{"text-anchor": "middle", "class": "title"},
    )

    for panel, (x0, y0) in zip(PANELS, panel_positions):
        values_by_system = [
            df.loc[df["system"].eq(system), str(panel["column"])].dropna().to_numpy(dtype=float)
            for system in SYSTEM_ORDER
        ]
        ymin, ymax, tick_step = panel_limits(values_by_system, panel)

        def sx(index: int) -> float:
            return x0 + 55 + (index + 0.5) * (panel_w - 88) / len(SYSTEM_ORDER)

        def sy(value: float) -> float:
            return y0 + (ymax - value) / (ymax - ymin) * panel_h

        add_text(svg, x0, y0 - 30, str(panel["letter"]), **{"class": "letter"})
        add_text(
            svg,
            x0 + panel_w / 2,
            y0 - 30,
            html.escape(str(panel["title"])),
            **{"text-anchor": "middle", "class": "paneltitle"},
        )

        tick_start = np.ceil(ymin / tick_step) * tick_step
        for tick in np.arange(tick_start, ymax + 0.5 * tick_step, tick_step):
            yy = sy(float(tick))
            svg.append(
                f'<line x1="{x0+46:.2f}" x2="{x0+panel_w:.2f}" y1="{yy:.2f}" '
                f'y2="{yy:.2f}" stroke="#E5E7EB" stroke-width="1"/>'
            )
            add_text(
                svg,
                x0 + 38,
                yy + 5,
                fmt_tick(float(tick), tick_step),
                **{"text-anchor": "end", "class": "tick"},
            )

        for ref_value, ref_label in panel["ref_lines"]:
            if ymin <= ref_value <= ymax:
                yy = sy(ref_value)
                svg.append(
                    f'<line x1="{x0+46:.2f}" x2="{x0+panel_w:.2f}" y1="{yy:.2f}" '
                    f'y2="{yy:.2f}" stroke="#555" stroke-width="1.1" stroke-dasharray="7 5"/>'
                )
                add_text(
                    svg,
                    x0 + panel_w - 4,
                    yy - 7,
                    html.escape(ref_label),
                    **{"text-anchor": "end", "class": "tick"},
                )

        svg.append(
            f'<line x1="{x0+46:.2f}" x2="{x0+46:.2f}" y1="{y0:.2f}" '
            f'y2="{y0+panel_h:.2f}" stroke="#333" stroke-width="1.2"/>'
        )
        svg.append(
            f'<line x1="{x0+46:.2f}" x2="{x0+panel_w:.2f}" y1="{y0+panel_h:.2f}" '
            f'y2="{y0+panel_h:.2f}" stroke="#333" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text transform="translate({x0+8:.2f} {y0+panel_h/2:.2f}) rotate(-90)" '
            f'text-anchor="middle" class="axis">{panel["ylabel"]}</text>'
        )

        grid = np.linspace(ymin, ymax, 260)
        max_width = 34
        for index, (system, label, color, values) in enumerate(
            zip(SYSTEM_ORDER, SYSTEM_LABELS, COLORS, values_by_system)
        ):
            x_center = sx(index)
            density = kde(values, grid, ymin, ymax)
            if density.max() > 0:
                density = density / density.max() * max_width
            left_points = [(x_center - d, sy(g)) for d, g in zip(density, grid)]
            right_points = [
                (x_center + d, sy(g)) for d, g in zip(density[::-1], grid[::-1])
            ]
            points = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in left_points + right_points
            )
            svg.append(
                f'<polygon points="{points}" fill="{color}" fill-opacity="0.38" '
                f'stroke="#222" stroke-width="0.9"/>'
            )

            q05, q25, med, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
            box_w = 20
            svg.append(
                f'<line x1="{x_center:.2f}" x2="{x_center:.2f}" y1="{sy(q05):.2f}" '
                f'y2="{sy(q95):.2f}" stroke="#222" stroke-width="1"/>'
            )
            svg.append(
                f'<rect x="{x_center-box_w/2:.2f}" y="{sy(q75):.2f}" '
                f'width="{box_w}" height="{sy(q25)-sy(q75):.2f}" fill="{color}" '
                f'fill-opacity="0.88" stroke="#222" stroke-width="0.9"/>'
            )
            svg.append(
                f'<line x1="{x_center-box_w/2:.2f}" x2="{x_center+box_w/2:.2f}" '
                f'y1="{sy(med):.2f}" y2="{sy(med):.2f}" stroke="#000" stroke-width="1.8"/>'
            )
            sample_size = min(len(values), 90)
            sample = values[rng.choice(len(values), size=sample_size, replace=False)]
            for value in sample:
                jitter = float(rng.normal(0, 3.0))
                svg.append(
                    f'<circle cx="{x_center+jitter:.2f}" cy="{sy(value):.2f}" '
                    f'r="1.65" fill="{color}" fill-opacity="0.16"/>'
                )
            add_text(
                svg,
                x_center,
                y0 + panel_h + 25,
                html.escape(label),
                **{"text-anchor": "middle", "class": "sys"},
            )

            summary_rows.append(
                {
                    "descriptor": panel["column"],
                    "system": system,
                    "system_name": SYSTEM_NAMES[system],
                    "n": len(values),
                    "mean": round(float(np.mean(values)), 6),
                    "median": round(float(np.median(values)), 6),
                    "p05": round(float(np.percentile(values, 5)), 6),
                    "q25": round(float(np.percentile(values, 25)), 6),
                    "q75": round(float(np.percentile(values, 75)), 6),
                    "p95": round(float(np.percentile(values, 95)), 6),
                    "min": round(float(np.min(values)), 6),
                    "max": round(float(np.max(values)), 6),
                }
            )

    legend_y = height - 28
    legend_x = 320
    for i, (label, color) in enumerate(zip(SYSTEM_NAMES.values(), COLORS)):
        x = legend_x + i * 175
        svg.append(f'<circle cx="{x:.2f}" cy="{legend_y-5:.2f}" r="7" fill="{color}"/>')
        add_text(svg, x + 12, legend_y, html.escape(label), **{"class": "note"})

    svg.append("</svg>")

    figure_path = OUTDIR / "combined_key_descriptor_distributions.svg"
    summary_path = OUTDIR / "combined_key_descriptor_distribution_summary.csv"
    figure_path.write_text("\n".join(svg))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(figure_path)
    print(summary_path)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    missing = [panel["column"] for panel in PANELS if panel["column"] not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    make_figure(df)


if __name__ == "__main__":
    main()
