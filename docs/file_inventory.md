# File Inventory

## Data

- `data/processed/ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv`: main manuscript modeling table.
- `data/processed/ALL_DIMERS_19descriptors_7targets_NO_S2_FILTER.csv`: extended target table retained for provenance.
- `data/processed/fodft_couplings_all.csv`: fragment-orbital coupling results.
- `data/diagnostics/BS_S2_ALL_STRUCTURE_AUDIT_NO_FILTER.csv`: final broken-symmetry `<S^2>` audit values.

## Scripts

- `scripts/pipeline/`: main no-filter ML pipeline.
- `scripts/fodft/`: FO-DFT coupling extraction and audit scripts.
- `scripts/validation/`: validation and model-complexity analyses.
- `scripts/feature_importance/`: feature-importance plotting.
- `scripts/plotting/`: target, descriptor, and diagnostic plotting.

## Results

- `results/model_metrics/`: cross-validation, train/validation/test, feature-importance, and individual-system metrics.
- `results/y_randomization/`: 500-permutation Y-randomization control outputs.
- `results/geometry_clustered_validation/`: geometry-clustered external validation outputs.
- `results/complexity/`: Extra Trees and MLP model-complexity sensitivity outputs.
- `results/fodft_qc/`: FO-DFT coupling quality-control summaries.
- `results/diagnostics/`: descriptor and target distribution summaries.

## Figures

- `figures/main/`: main model correlation/parity figures.
- `figures/feature_importance/`: feature-importance figures.
- `figures/validation/`: Y-randomization and geometry-clustered validation figures.
- `figures/complexity/`: model-complexity and convergence figures.
- `figures/descriptor_distributions/`: descriptor-distribution figures.
- `figures/supporting/`: additional SI figures.
