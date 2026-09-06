#!/usr/bin/env python3
"""
Three reviewer-requested analyses in one job.

  A.  Leave-one-system-out validation, with a descriptor-space coverage
      diagnostic that explains the result                        (Reviewer 2.1)
  B.  Regularized linear (Ridge) baseline                        (Reviewer 2.2)
  C.  Permutation importance for ExtRa Trees, individual and grouped
                                                                 (Reviewer 2.3, 2.4)

All three reuse the same descriptors, the same partitions and the same model
settings as the manuscript, so the numbers are directly comparable to published
values. Nothing existing is modified; output goes to a new directory.

-------------------------------------------------------------------------------
A. LEAVE-ONE-SYSTEM-OUT
-------------------------------------------------------------------------------
Each radical family is held out in turn, the models are fitted on the remaining
four, and performance is measured on the held-out family. Three references are
reported alongside the model:

  R2_model        conventional R^2 against the held-out family's own mean
  R2_global_mean  what you get by predicting the training-set global mean;
                  this is the honest baseline for an unseen family, because
                  its own mean is by definition unknown
  R2_oracle_mean  what you would get if the held-out family's mean WERE known;
                  the ceiling attributable to knowing the family offset

The gap between R2_model and R2_oracle_mean separates two different failures:
inability to place the family on the right absolute scale, versus failure of the
geometry-property relationship itself to transfer.

A coverage diagnostic is computed first, because it largely determines the
outcome. Tree ensembles cannot predict outside the range of their training
targets by construction, so a family lying outside the training descriptor range
is not merely difficult but unreachable.

-------------------------------------------------------------------------------
B. RIDGE BASELINE
-------------------------------------------------------------------------------
RidgeCV over a broad alpha grid, inside the same imputation and scaling pipeline,
evaluated with the manuscript's five-fold CV and its 70/15/15 test split. The
alpha grid is searched so the linear model is not handicapped by an arbitrary
regularization choice.

-------------------------------------------------------------------------------
C. EXTRA TREES PERMUTATION IMPORTANCE
-------------------------------------------------------------------------------
Permutation importance on held-out data, so both model families can be compared
on the same model-agnostic metric. Also computed with the three correlated
contact-count descriptors permuted together, since permuting one member of a
correlated group lets the model recover the information from its neighbours and
understates the group's true contribution.

-------------------------------------------------------------------------------
INPUT (read-only), relative to the run directory
-------------------------------------------------------------------------------
  ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv
  unified_19descriptor_no_s2_filter_manifest.json
  train_validate_test/extra_trees/extra_trees_train_validate_test_predictions.csv
      (supplies the 70/15/15 membership by structure id)

OUTPUT -> revision_analyses/
  loso_coverage_diagnostic.csv     descriptor- and target-range coverage
  loso_results.csv                 per family, target and model
  ridge_baseline.csv               Ridge vs ExtRa Trees vs MLP
  extra_trees_permutation_importance.csv
  extra_trees_permutation_grouped.csv
  REVISION_REPORT.md
  fig_loso.png/.pdf, fig_ridge_baseline.png/.pdf

Usage
  python revision_analyses.py                    # auto-locates the run directory
  python revision_analyses.py --run-dir /path/to/FODFT_..._164211
  python revision_analyses.py --skip mlp         # ExtRa Trees only, much faster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUN_NAME = "FODFT_4Target_NO_S2_FILTER_Run_20260707_164211"
RANDOM_STATE = 42
N_REPEATS = 30

# Order follows Table 2 of the manuscript.
TARGETS = [
    "target_fodft_coupling_abs_ev",
    "target_frontier_gap_ev",
    "target_interaction_energy_kcal_mol",
    "target_somo_somo_elst_corrected_signed_kcal_mol",
]
LABEL = {
    "target_fodft_coupling_abs_ev": r"$|t_{FO}|$",
    "target_frontier_gap_ev": r"$\Delta E_{frontier}$",
    "target_interaction_energy_kcal_mol": r"$\Delta E_{int}$",
    "target_somo_somo_elst_corrected_signed_kcal_mol": r"$\Delta E^{corr}_{SOMO-SOMO}$",
}
CONTACT_COUNT_GROUP = [
    "n_interfragment_c_contacts_3p4",
    "n_interfragment_c_contacts_3p6",
    "n_interfragment_c_contacts_4p0",
]


def find_run_dir(start: Path) -> Path:
    for d in [start, *start.parents]:
        if d.name == RUN_NAME:
            return d
        if (d / RUN_NAME).is_dir():
            return d / RUN_NAME
    raise FileNotFoundError(
        f"Could not locate '{RUN_NAME}' from {start}. Pass --run-dir explicitly.")


# ---------------------------------------------------------------------------
# models, matching the manuscript exactly
# ---------------------------------------------------------------------------

def extra_trees() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, criterion="squared_error",
            max_depth=None, max_features=1.0, bootstrap=False,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])


# The MLP architecture selected by the one-standard-error rule DIFFERS BY
# TARGET. Taken from Y_RANDOMIZATION_SUMMARY.csv / the selected_test rows of
# mlp_train_validate_test_metrics.csv. Using a single architecture for all four
# targets does not reproduce the published models.
MLP_CONFIG = {
    "target_fodft_coupling_abs_ev":                   ((128, 64, 32),     1e-4),
    "target_frontier_gap_ev":                         ((64, 64, 32, 16),  1e-4),
    "target_interaction_energy_kcal_mol":             ((128, 64, 32),     1e-3),
    "target_somo_somo_elst_corrected_signed_kcal_mol":((128, 64, 32),     1e-3),
}


def mlp(target: str) -> TransformedTargetRegressor:
    layers, alpha = MLP_CONFIG[target]
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("x_scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=layers, activation="relu", solver="adam",
            alpha=alpha, batch_size=64, learning_rate_init=1e-3, max_iter=900,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            tol=1e-5, random_state=RANDOM_STATE)),
    ])
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def ridge() -> TransformedTargetRegressor:
    """
    Regularized linear baseline. Standardization is essential for Ridge, and the
    alpha grid is searched so the linear model is not handicapped by an
    arbitrary choice of penalty.
    """
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("x_scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 41))),
    ])
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def build(fam: str, target: str):
    """Return an unfitted model. Only the MLP needs the target, to pick its
    per-target architecture; the others ignore it."""
    if fam == "mlp":
        return mlp(target)
    return {"extra_trees": extra_trees, "ridge": ridge}[fam]()


# ---------------------------------------------------------------------------

def load(run: Path):
    features = json.loads((run / "unified_19descriptor_no_s2_filter_manifest.json")
                          .read_text())["features19"]
    data = pd.read_csv(run / "ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv")
    preds = pd.read_csv(run / "train_validate_test" / "extra_trees"
                        / "extra_trees_train_validate_test_predictions.csv")
    split = (preds[preds["target"].eq(TARGETS[0])][["id", "split"]].drop_duplicates("id"))
    merged = data.merge(split, on="id", validate="one_to_one")
    if len(merged) != len(data):
        raise RuntimeError(f"merge changed row count: {len(data)} -> {len(merged)}")
    return merged, features


# ---------------------------------------------------------------------------
# A. leave-one-system-out
# ---------------------------------------------------------------------------

def coverage_diagnostic(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for s in sorted(df["system"].unique()):
        te, tr = df[df["system"].eq(s)], df[df["system"].ne(s)]
        lo, hi = tr[features].min(), tr[features].max()
        outside = ((te[features] < lo) | (te[features] > hi)).any(axis=1)
        no_overlap = [f for f in features
                      if te[f].min() > hi[f] or te[f].max() < lo[f]]
        row = {"held_out_system": s, "n": len(te),
               "frac_rows_outside_descriptor_range": float(outside.mean()),
               "n_descriptors_no_overlap": len(no_overlap),
               "descriptors_no_overlap": "|".join(no_overlap)}
        for t in TARGETS:
            a, b = tr[t].min(), tr[t].max()
            row[f"frac_target_outside__{t.replace('target_','')}"] = \
                float(((te[t] < a) | (te[t] > b)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_loso(df, features, families) -> pd.DataFrame:
    rows = []
    for s in sorted(df["system"].unique()):
        te, tr = df[df["system"].eq(s)], df[df["system"].ne(s)]
        for target in TARGETS:
            y_tr, y_te = tr[target].to_numpy(), te[target].to_numpy()
            # References that need no model.
            r2_global = r2_score(y_te, np.full_like(y_te, y_tr.mean()))
            r2_oracle = r2_score(y_te, np.full_like(y_te, y_te.mean()))  # == 0 by construction
            for fam in families:
                m = build(fam, target).fit(tr[features], y_tr)
                p = m.predict(te[features])
                rows.append({
                    "held_out_system": s, "target": target, "model_family": fam,
                    "n_train": len(tr), "n_test": len(te),
                    "r2": r2_score(y_te, p), "mae": mean_absolute_error(y_te, p),
                    "r2_predicting_global_training_mean": r2_global,
                    "r2_predicting_heldout_own_mean": r2_oracle,
                    "beats_global_mean": bool(r2_score(y_te, p) > r2_global),
                })
            print(f"    [LOSO] {s:<24} {target.replace('target_',''):<42} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. Ridge baseline
# ---------------------------------------------------------------------------

def run_ridge_baseline(df, features, families) -> pd.DataFrame:
    from sklearn.model_selection import KFold, cross_val_predict
    tr = df["split"].eq("train")
    te = df["split"].eq("test")
    rows = []
    for target in TARGETS:
        y_all = df[target].to_numpy()
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for fam in families:
            # five-fold CV over the whole dataset, as in the manuscript
            oof = cross_val_predict(build(fam, target), df[features], y_all, cv=cv)
            # independent 70/15/15 test subset
            m = build(fam, target).fit(df.loc[tr, features], df.loc[tr, target])
            p = m.predict(df.loc[te, features])
            rows.append({
                "target": target, "model_family": fam,
                "cv_r2": r2_score(y_all, oof), "cv_mae": mean_absolute_error(y_all, oof),
                "test_r2": r2_score(df.loc[te, target], p),
                "test_mae": mean_absolute_error(df.loc[te, target], p),
            })
            print(f"    [ridge-baseline] {target.replace('target_',''):<42} {fam:<12} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. ExtraTrees permutation importance
# ---------------------------------------------------------------------------

def grouped_permutation(model, x_te, y_te, members, rng, n_repeats):
    base = r2_score(y_te, model.predict(x_te))
    drops = []
    for _ in range(n_repeats):
        shuffled = x_te.copy()
        order = rng.permutation(len(shuffled))
        for col in members:                       # same row order for every member,
            shuffled[col] = shuffled[col].to_numpy()[order]   # preserving within-group correlation
        drops.append(base - r2_score(y_te, model.predict(shuffled)))
    return float(np.mean(drops)), float(np.std(drops))


def run_et_permutation(df, features):
    tr, te = df["split"].eq("train"), df["split"].eq("test")
    rng = np.random.default_rng(RANDOM_STATE)
    ind, grp = [], []
    for target in TARGETS:
        m = extra_trees().fit(df.loc[tr, features], df.loc[tr, target])
        x_te, y_te = df.loc[te, features], df.loc[te, target]
        res = permutation_importance(m, x_te, y_te, n_repeats=N_REPEATS,
                                     random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
        for f, mu, sd in zip(features, res.importances_mean, res.importances_std):
            ind.append({"target": target, "model_family": "extra_trees",
                        "importance_type": "permutation_r2_drop",
                        "feature": f, "importance": mu, "importance_std": sd})
        mu, sd = grouped_permutation(m, x_te, y_te, CONTACT_COUNT_GROUP, rng, N_REPEATS)
        grp.append({"target": target, "group": "contact_count_block",
                    "members": "|".join(CONTACT_COUNT_GROUP),
                    "grouped_importance_mean": mu, "grouped_importance_std": sd,
                    "sum_of_individual": float(sum(
                        res.importances_mean[features.index(f)] for f in CONTACT_COUNT_GROUP))})
        print(f"    [ET-permutation] {target.replace('target_',''):<42} done")
    return pd.DataFrame(ind), pd.DataFrame(grp)


# ---------------------------------------------------------------------------

def figures(loso: pd.DataFrame, ridge_df: pd.DataFrame, out: Path):
    fams = sorted(loso["model_family"].unique())
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(3.6 * len(TARGETS), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    systems = sorted(loso["held_out_system"].unique())
    for ax, t in zip(axes, TARGETS):
        b = loso[loso["target"].eq(t)]
        x = np.arange(len(systems))
        w = 0.8 / max(len(fams), 1)
        for i, fam in enumerate(fams):
            v = [b[(b.held_out_system == s) & (b.model_family == fam)]["r2"].iloc[0]
                 for s in systems]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, label=fam, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("phenalenyl_olympicenyl", "PLY-OLY") for s in systems],
                           rotation=40, ha="right", fontsize=8)
        ax.set_title(LABEL[t], fontsize=11)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$R^2$ on the held-out family")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(fams), loc="lower center", bbox_to_anchor=(0.5, -0.10),
               frameon=False, fontsize=9)
    fig.suptitle("Leave-one-system-out: transfer to an unseen radical family", fontsize=12)
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig_loso.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fams2 = ["ridge", "extra_trees", "mlp"]
    fams2 = [f for f in fams2 if f in set(ridge_df["model_family"])]
    x = np.arange(len(TARGETS)); w = 0.8 / len(fams2)
    for i, fam in enumerate(fams2):
        v = [ridge_df[(ridge_df.target == t) & (ridge_df.model_family == fam)]["test_r2"].iloc[0]
             for t in TARGETS]
        ax.bar(x + i * w - 0.4 + w / 2, v, w, label=fam, edgecolor="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[t] for t in TARGETS])
    ax.set_ylabel(r"test-set $R^2$"); ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=9)
    ax.set_title("Linear baseline against the nonlinear models (70/15/15 test subset)")
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig_ridge_baseline.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def report(cov, loso, ridge_df, grp, out: Path):
    L = ["# Revision analyses", "",
         "## A. Descriptor-space coverage when each family is held out", "",
         "| Held out | n | % rows outside training range | descriptors with no overlap |",
         "|---|---|---|---|"]
    for _, r in cov.iterrows():
        L.append(f"| {r.held_out_system} | {int(r.n)} | "
                 f"{100*r['frac_rows_outside_descriptor_range']:.1f}% | "
                 f"{int(r.n_descriptors_no_overlap)} |")
    L += ["", "Tree ensembles cannot predict outside the range of their training targets,",
          "so a family lying wholly outside the training descriptor range is unreachable",
          "rather than merely difficult.", "",
          "## B. Leave-one-system-out", "",
          "| Held out | Target | Model | R2 | MAE | R2 of global-mean prediction | beats it |",
          "|---|---|---|---|---|---|---|"]
    for _, r in loso.iterrows():
        L.append(f"| {r.held_out_system} | {r.target.replace('target_','')} | {r.model_family} "
                 f"| {r.r2:.3f} | {r.mae:.4f} | {r.r2_predicting_global_training_mean:.3f} "
                 f"| {'yes' if r.beats_global_mean else 'NO'} |")
    L += ["", "## C. Ridge baseline", "",
          "| Target | Model | CV R2 | CV MAE | Test R2 | Test MAE |", "|---|---|---|---|---|---|"]
    for _, r in ridge_df.iterrows():
        L.append(f"| {r.target.replace('target_','')} | {r.model_family} | {r.cv_r2:.3f} "
                 f"| {r.cv_mae:.4f} | {r.test_r2:.3f} | {r.test_mae:.4f} |")
    L += ["", "## D. Grouped permutation importance, contact-count block (ExtRa Trees)", "",
          "| Target | Group total | Sum of individual | ratio |", "|---|---|---|---|"]
    for _, r in grp.iterrows():
        ratio = r.grouped_importance_mean / max(r.sum_of_individual, 1e-12)
        L.append(f"| {r.target.replace('target_','')} | {r.grouped_importance_mean:.4f} "
                 f"| {r.sum_of_individual:.4f} | {ratio:.2f} |")
    L += ["", "A group total larger than the sum of its parts means the individual",
          "importances were masked by correlation between the members.", ""]
    (out / "REVISION_REPORT.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="model families to skip, e.g. --skip mlp")
    a = ap.parse_args()

    run = a.run_dir.resolve() if a.run_dir else find_run_dir(HERE)
    out = a.out_dir.resolve() if a.out_dir else (HERE / "revision_analyses")
    out.mkdir(parents=True, exist_ok=True)
    print(f"run directory : {run}\noutput        : {out}\n")

    df, features = load(run)
    families = [f for f in ["extra_trees", "mlp"] if f not in a.skip]
    print(f"{len(df)} structures, {len(features)} descriptors, "
          f"{df.system.nunique()} families, models: {families}\n")

    print("A. coverage diagnostic + leave-one-system-out")
    cov = coverage_diagnostic(df, features)
    cov.to_csv(out / "loso_coverage_diagnostic.csv", index=False)
    loso = run_loso(df, features, families)
    loso.to_csv(out / "loso_results.csv", index=False)

    print("\nB. Ridge baseline")
    ridge_df = run_ridge_baseline(df, features, families + ["ridge"])
    ridge_df.to_csv(out / "ridge_baseline.csv", index=False)

    print("\nC. ExtRa Trees permutation importance")
    ind, grp = run_et_permutation(df, features)
    ind.to_csv(out / "extra_trees_permutation_importance.csv", index=False)
    grp.to_csv(out / "extra_trees_permutation_grouped.csv", index=False)

    figures(loso, ridge_df, out)
    report(cov, loso, ridge_df, grp, out)

    print("\n--- LOSO summary, ExtRa Trees ---")
    b = loso[loso.model_family.eq("extra_trees")]
    for t in TARGETS:
        v = b[b.target.eq(t)]
        print(f"  {t.replace('target_',''):<42} R2 " +
              " ".join(f"{x:6.2f}" for x in v.r2) +
              f"   ({(~v.beats_global_mean).sum()}/{len(v)} fail to beat the global mean)")
    print(f"\nAll outputs written to: {out}")


if __name__ == "__main__":
    main()
