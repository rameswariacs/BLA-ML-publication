# Geometry-Based Machine Learning for pi-Stacked Radical Dimers

This repository contains the processed data, Python scripts, validation outputs, and publication figures associated with a descriptor-based machine-learning study of pi-stacked radical dimers.

The workflow relates 19 geometry-derived descriptors to DFT-derived electronic properties using Extra Trees regression and descriptor-based multilayer perceptron (MLP) models.

## Contents

- `data/processed/`: final processed descriptor/target tables used for modeling.
- `data/diagnostics/`: diagnostic tables, including final broken-symmetry `<S^2>` audit values.
- `scripts/pipeline/`: main no-`<S^2>`-filter workflow scripts.
- `scripts/fodft/`: fragment-orbital coupling extraction and quality-control scripts.
- `scripts/validation/`: Y-randomization, geometry-clustered validation, and model-complexity scripts.
- `scripts/feature_importance/`: feature-importance plotting scripts.
- `scripts/plotting/`: descriptor and target distribution plotting scripts.
- `results/`: CSV/JSON/Markdown outputs used to prepare the manuscript and SI tables.
- `figures/`: publication and supporting-information figures in vector/PDF/PNG formats.
- `docs/`: descriptor definitions and reproducibility notes.

## Main Data File

The main modeling table is:

```text
data/processed/ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv
```

It contains 2,582 pi-stacked radical-dimer geometries represented by 19 structural descriptors and four manuscript target properties:

- fragment-orbital SOMO-SOMO coupling, `target_fodft_coupling_abs_ev`
- interaction energy, `target_interaction_energy_kcal_mol`
- spin-conserving frontier gap, `target_frontier_gap_ev`
- electrostatically corrected SOMO-SOMO interaction energy, `target_somo_somo_elst_corrected_signed_kcal_mol`

## Reproducibility Notes

The repository is designed to reproduce the machine-learning analyses from processed descriptor/target tables. It does not include raw ORCA scratch files, `.gbw` files, density files, or large model pickle files.

The main workflow scripts were originally run on an HPC system and contain absolute HPC paths. To rerun locally, update the path constants in the scripts or use the processed CSV files directly with the validation and plotting scripts.

## Software

The analysis used Python with NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, and joblib. ORCA output and ORCA utility-generated JSON files were used for fragment-orbital coupling analysis. See `requirements.txt` and the Supporting Information for details.

## Licensing

Code in this repository is released under the MIT License. Processed data, descriptor tables, and figures are released under CC BY 4.0 unless otherwise noted. See `LICENSE` and `LICENSE-DATA`.

## Citation

If you use this workflow, data, or figures, please cite the associated manuscript when available. A provisional `CITATION.cff` file is included and should be updated with the final publication details.
