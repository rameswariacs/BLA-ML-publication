# Revision analyses

## A. Descriptor-space coverage when each family is held out

| Held out | n | % rows outside training range | descriptors with no overlap |
|---|---|---|---|
| CPBP | 793 | 100.0% | 1 |
| fluorenyl | 367 | 14.2% | 0 |
| olympicenyl | 532 | 0.9% | 0 |
| phenalenyl | 252 | 43.7% | 0 |
| phenalenyl_olympicenyl | 638 | 100.0% | 1 |

Tree ensembles cannot predict outside the range of their training targets,
so a family lying wholly outside the training descriptor range is unreachable
rather than merely difficult.

## B. Leave-one-system-out

| Held out | Target | Model | R2 | MAE | R2 of global-mean prediction | beats it |
|---|---|---|---|---|---|---|
| CPBP | fodft_coupling_abs_ev | extra_trees | 0.548 | 0.0988 | -0.000 | yes |
| CPBP | fodft_coupling_abs_ev | mlp | -5.369 | 0.4622 | -0.000 | NO |
| CPBP | frontier_gap_ev | extra_trees | -76.531 | 0.5451 | -109.531 | yes |
| CPBP | frontier_gap_ev | mlp | -1.537 | 0.0757 | -109.531 | yes |
| CPBP | interaction_energy_kcal_mol | extra_trees | -3.171 | 5.7097 | -6.877 | yes |
| CPBP | interaction_energy_kcal_mol | mlp | -0.911 | 2.7017 | -6.877 | yes |
| CPBP | somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.373 | 2.2554 | -0.030 | yes |
| CPBP | somo_somo_elst_corrected_signed_kcal_mol | mlp | -1.074 | 3.8855 | -0.030 | NO |
| fluorenyl | fodft_coupling_abs_ev | extra_trees | 0.885 | 0.0440 | -0.092 | yes |
| fluorenyl | fodft_coupling_abs_ev | mlp | 0.656 | 0.0821 | -0.092 | yes |
| fluorenyl | frontier_gap_ev | extra_trees | -7.923 | 0.3750 | -23.244 | yes |
| fluorenyl | frontier_gap_ev | mlp | -10.568 | 0.4150 | -23.244 | yes |
| fluorenyl | interaction_energy_kcal_mol | extra_trees | 0.456 | 1.4984 | -3.643 | yes |
| fluorenyl | interaction_energy_kcal_mol | mlp | -0.421 | 2.1287 | -3.643 | yes |
| fluorenyl | somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.674 | 0.9979 | -0.079 | yes |
| fluorenyl | somo_somo_elst_corrected_signed_kcal_mol | mlp | 0.615 | 1.5477 | -0.079 | yes |
| olympicenyl | fodft_coupling_abs_ev | extra_trees | 0.873 | 0.0581 | -0.006 | yes |
| olympicenyl | fodft_coupling_abs_ev | mlp | 0.143 | 0.1512 | -0.006 | yes |
| olympicenyl | frontier_gap_ev | extra_trees | -2.064 | 0.1177 | -0.019 | NO |
| olympicenyl | frontier_gap_ev | mlp | -18.268 | 0.3371 | -0.019 | NO |
| olympicenyl | interaction_energy_kcal_mol | extra_trees | 0.335 | 2.0562 | -0.012 | yes |
| olympicenyl | interaction_energy_kcal_mol | mlp | 0.417 | 1.8655 | -0.012 | yes |
| olympicenyl | somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.896 | 0.9725 | -0.002 | yes |
| olympicenyl | somo_somo_elst_corrected_signed_kcal_mol | mlp | -0.627 | 4.1239 | -0.002 | NO |
| phenalenyl | fodft_coupling_abs_ev | extra_trees | 0.775 | 0.0906 | -0.131 | yes |
| phenalenyl | fodft_coupling_abs_ev | mlp | 0.745 | 0.1000 | -0.131 | yes |
| phenalenyl | frontier_gap_ev | extra_trees | -0.395 | 0.1639 | -4.412 | yes |
| phenalenyl | frontier_gap_ev | mlp | 0.197 | 0.1045 | -4.412 | yes |
| phenalenyl | interaction_energy_kcal_mol | extra_trees | 0.476 | 1.7926 | -1.491 | yes |
| phenalenyl | interaction_energy_kcal_mol | mlp | 0.259 | 2.4578 | -1.491 | yes |
| phenalenyl | somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.681 | 2.3096 | -0.134 | yes |
| phenalenyl | somo_somo_elst_corrected_signed_kcal_mol | mlp | 0.509 | 2.8472 | -0.134 | yes |
| phenalenyl_olympicenyl | fodft_coupling_abs_ev | extra_trees | 0.858 | 0.0550 | -0.022 | yes |
| phenalenyl_olympicenyl | fodft_coupling_abs_ev | mlp | -84.480 | 1.7907 | -0.022 | NO |
| phenalenyl_olympicenyl | frontier_gap_ev | extra_trees | -1.525 | 0.1671 | -1.293 | NO |
| phenalenyl_olympicenyl | frontier_gap_ev | mlp | -428.278 | 2.1551 | -1.293 | NO |
| phenalenyl_olympicenyl | interaction_energy_kcal_mol | extra_trees | 0.647 | 1.4335 | -1.193 | yes |
| phenalenyl_olympicenyl | interaction_energy_kcal_mol | mlp | -1338.348 | 99.8540 | -1.193 | NO |
| phenalenyl_olympicenyl | somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.831 | 1.0426 | -0.003 | yes |
| phenalenyl_olympicenyl | somo_somo_elst_corrected_signed_kcal_mol | mlp | -289.582 | 64.3528 | -0.003 | NO |

## C. Ridge baseline

| Target | Model | CV R2 | CV MAE | Test R2 | Test MAE |
|---|---|---|---|---|---|
| fodft_coupling_abs_ev | extra_trees | 0.941 | 0.0354 | 0.957 | 0.0343 |
| fodft_coupling_abs_ev | mlp | 0.758 | 0.0485 | 0.937 | 0.0419 |
| fodft_coupling_abs_ev | ridge | 0.728 | 0.0851 | 0.755 | 0.0854 |
| frontier_gap_ev | extra_trees | 0.977 | 0.0360 | 0.977 | 0.0374 |
| frontier_gap_ev | mlp | 0.968 | 0.0462 | 0.961 | 0.0484 |
| frontier_gap_ev | ridge | 0.932 | 0.0667 | 0.934 | 0.0661 |
| interaction_energy_kcal_mol | extra_trees | 0.946 | 0.7722 | 0.941 | 0.8328 |
| interaction_energy_kcal_mol | mlp | 0.932 | 0.8748 | 0.933 | 0.9019 |
| interaction_energy_kcal_mol | ridge | 0.771 | 1.7412 | 0.761 | 1.7744 |
| somo_somo_elst_corrected_signed_kcal_mol | extra_trees | 0.959 | 0.5193 | 0.965 | 0.5501 |
| somo_somo_elst_corrected_signed_kcal_mol | mlp | 0.915 | 0.7780 | 0.964 | 0.6434 |
| somo_somo_elst_corrected_signed_kcal_mol | ridge | 0.621 | 2.0502 | 0.668 | 2.0732 |

## D. Grouped permutation importance, contact-count block (ExtRa Trees)

| Target | Group total | Sum of individual | ratio |
|---|---|---|---|
| fodft_coupling_abs_ev | 0.2402 | 0.1612 | 1.49 |
| frontier_gap_ev | 0.0582 | 0.0291 | 2.00 |
| interaction_energy_kcal_mol | 0.1159 | 0.0886 | 1.31 |
| somo_somo_elst_corrected_signed_kcal_mol | 0.2343 | 0.1798 | 1.30 |

A group total larger than the sum of its parts means the individual
importances were masked by correlation between the members.
