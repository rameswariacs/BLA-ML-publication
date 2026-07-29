# Geometry-Clustered External Validation

## Purpose

This independent analysis tests whether the reported performance survives a split in which structurally similar geometries are kept together. The authoritative dataset, 19 descriptors, BS-state filter, target definitions, and manuscript model settings were not modified.

## Split construction

- Input structures: 2582.
- Clustering variables: the same 19 structural descriptors used by the regression models.
- Clustering was performed separately within each molecular system after median imputation and descriptor standardization.
- Ward agglomerative clustering used approximately one cluster per 25 structures.
- Complete clusters were assigned to train, validation, or test; no geometry cluster occurs in more than one subset.
- Assignment targeted 70/15/15 within every represented system.
- Target values were not used to construct clusters or assign subsets.

## Model protocol

- Extra Trees: 500 trees, minimum leaf size 2, all 19 descriptors, random seed 42.
- MLP: 19-128-64-32-1, alpha = 1e-3, ReLU, Adam, standardized X and y, early stopping, random seed 42.
- The model settings were fixed before this analysis and were not retuned against the clustered test set.
- A cluster-grouped five-fold CV was performed within the combined training-validation pool.
- Final models were fitted to the combined training-validation pool and evaluated once on the cluster-held-out test set.
- A random 70/15/15 split using the same fixed models was rerun as an apples-to-apples reference.

## Split balance

| System | Train | Validation | Test |
|---|---:|---:|---:|
| CPBP | 555 | 119 | 119 |
| fluorenyl | 257 | 55 | 55 |
| olympicenyl | 372 | 80 | 80 |
| phenalenyl | 176 | 38 | 38 |
| phenalenyl_olympicenyl | 446 | 96 | 96 |

## Descriptor-space separation

| Scheme | Mean nearest-development distance | Median distance | Fraction below development 5th-percentile threshold |
|---|---:|---:|---:|
| random | 0.6256 | 0.5232 | 0.054 |
| geometry_clustered | 1.2481 | 1.1171 | 0.000 |

## Cluster-grouped cross-validation

| Model | Target | R2 | MAE | RMSE |
|---|---|---:|---:|---:|
| extra_trees | |t_FO| | 0.8289 | 0.0551 | 0.0873 |
| extra_trees | Frontier gap | 0.9461 | 0.0582 | 0.0824 |
| extra_trees | Interaction energy | 0.8725 | 1.2573 | 1.7646 |
| extra_trees | Corrected SOMO-SOMO | 0.8289 | 0.9447 | 1.9276 |
| mlp | |t_FO| | 0.8060 | 0.0629 | 0.0930 |
| mlp | Frontier gap | 0.9322 | 0.0663 | 0.0924 |
| mlp | Interaction energy | 0.8607 | 1.2918 | 1.8448 |
| mlp | Corrected SOMO-SOMO | 0.8039 | 1.1112 | 2.0637 |

## Final external-test comparison

| Model | Target | Split | R2 | MAE | RMSE | 95% CI for R2 |
|---|---|---|---:|---:|---:|---:|
| extra_trees | |t_FO| | geometry_clustered | 0.8434 | 0.0644 | 0.1051 | [0.7989, 0.8771] |
| extra_trees | |t_FO| | random | 0.9565 | 0.0336 | 0.0475 | [0.9417, 0.9671] |
| extra_trees | Frontier gap | geometry_clustered | 0.9071 | 0.0646 | 0.1240 | [0.8565, 0.9529] |
| extra_trees | Frontier gap | random | 0.9770 | 0.0364 | 0.0541 | [0.9663, 0.9838] |
| extra_trees | Interaction energy | geometry_clustered | 0.8195 | 1.3282 | 1.9476 | [0.7690, 0.8636] |
| extra_trees | Interaction energy | random | 0.9437 | 0.8023 | 1.1809 | [0.9255, 0.9580] |
| extra_trees | Corrected SOMO-SOMO | geometry_clustered | 0.8166 | 1.2956 | 2.9415 | [0.7934, 0.8459] |
| extra_trees | Corrected SOMO-SOMO | random | 0.9643 | 0.5458 | 1.0087 | [0.9451, 0.9766] |
| mlp | |t_FO| | geometry_clustered | 0.8789 | 0.0704 | 0.0924 | [0.8449, 0.9028] |
| mlp | |t_FO| | random | 0.9424 | 0.0409 | 0.0547 | [0.9253, 0.9546] |
| mlp | Frontier gap | geometry_clustered | 0.9195 | 0.0690 | 0.1154 | [0.8866, 0.9482] |
| mlp | Frontier gap | random | 0.9713 | 0.0439 | 0.0604 | [0.9623, 0.9780] |
| mlp | Interaction energy | geometry_clustered | 0.8247 | 1.3628 | 1.9191 | [0.7783, 0.8640] |
| mlp | Interaction energy | random | 0.9362 | 0.9055 | 1.2564 | [0.9183, 0.9505] |
| mlp | Corrected SOMO-SOMO | geometry_clustered | 0.8899 | 1.2107 | 2.2785 | [0.8571, 0.9119] |
| mlp | Corrected SOMO-SOMO | random | 0.9698 | 0.5671 | 0.9280 | [0.9530, 0.9800] |

## Interpretation guide

The clustered split is intentionally more demanding than a random structure-level split. A reduction in R2 is expected if neighboring scan geometries previously appeared across subsets. The central questions are whether performance remains chemically useful, whether both model families show the same qualitative behavior, and whether the cluster-held-out test points are demonstrably farther from the development set in descriptor space.

This validation addresses interpolation across withheld geometric regions within the represented molecular systems. It is not a substitute for leave-one-system-out transfer and does not establish prediction for an entirely new molecular family.

Aggregate R2 values combine systems that occupy different target ranges. The accompanying `individual_system_external_test_metrics.csv` should therefore be consulted together with MAE and RMSE. Within-system R2 can be unstable when a held-out cluster spans only a narrow range of DFT values. The `worst_external_test_residuals.csv` file identifies localized extrapolation failures rather than concealing them in the aggregate correlation.
