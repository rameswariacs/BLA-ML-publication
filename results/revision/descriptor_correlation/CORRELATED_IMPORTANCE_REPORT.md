# Correlated descriptors and feature importance

Descriptor pairs with |rho| > 0.90: **4** of 171 possible pairs.
Pairs with |rho| > 0.80: **11**.

## Strongly correlated pairs

| Descriptor 1 | Descriptor 2 | Spearman rho |
|---|---|---|
| n_interfragment_c_contacts_3p6 | n_interfragment_c_contacts_4p0 | 0.976 |
| n_interfragment_c_contacts_3p4 | n_interfragment_c_contacts_3p6 | 0.956 |
| n_interfragment_c_contacts_3p4 | n_interfragment_c_contacts_4p0 | 0.933 |
| contact_min_cc_distance | mean_top10_closest_c_distances | 0.927 |

## Importance inside the contact-count group

Three descriptors carrying nearly the same information (|rho| = 0.93-0.98). Shares are within-group fractions; an equal split would be 0.333 each.

| Target | Method | 3.4 A | 3.6 A | 4.0 A | max/min |
|---|---|---|---|---|---|
| fodft_coupling_abs_ev | ExtRa Trees (impurity) | 0.727 | 0.187 | 0.086 | 8.5x |
| fodft_coupling_abs_ev | MLP (permutation) | 0.564 | 0.260 | 0.176 | 3.2x |
| frontier_gap_ev | ExtRa Trees (impurity) | 0.346 | 0.277 | 0.377 | 1.4x |
| frontier_gap_ev | MLP (permutation) | 0.322 | 0.234 | 0.444 | 1.9x |
| interaction_energy_kcal_mol | ExtRa Trees (impurity) | 0.234 | 0.257 | 0.509 | 2.2x |
| interaction_energy_kcal_mol | MLP (permutation) | 0.296 | 0.291 | 0.413 | 1.4x |
| somo_somo_elst_corrected_signed_kcal_mol | ExtRa Trees (impurity) | 0.796 | 0.122 | 0.082 | 9.7x |
| somo_somo_elst_corrected_signed_kcal_mol | MLP (permutation) | 0.505 | 0.241 | 0.253 | 2.1x |

## Group-level importance

| Target | Group | ExtRa Trees | MLP |
|---|---|---|---|
| fodft_coupling_abs_ev | Global pi-surface | 0.093 | 0.267 |
| fodft_coupling_abs_ev | Stacking | 0.055 | 0.147 |
| fodft_coupling_abs_ev | Local Contacts | 0.832 | 0.477 |
| fodft_coupling_abs_ev | BLA & Asymmetry | 0.020 | 0.108 |
| frontier_gap_ev | Global pi-surface | 0.744 | 0.490 |
| frontier_gap_ev | Stacking | 0.021 | 0.144 |
| frontier_gap_ev | Local Contacts | 0.070 | 0.269 |
| frontier_gap_ev | BLA & Asymmetry | 0.164 | 0.097 |
| interaction_energy_kcal_mol | Global pi-surface | 0.575 | 0.240 |
| interaction_energy_kcal_mol | Stacking | 0.068 | 0.211 |
| interaction_energy_kcal_mol | Local Contacts | 0.252 | 0.441 |
| interaction_energy_kcal_mol | BLA & Asymmetry | 0.106 | 0.109 |
| somo_somo_elst_corrected_signed_kcal_mol | Global pi-surface | 0.117 | 0.290 |
| somo_somo_elst_corrected_signed_kcal_mol | Stacking | 0.073 | 0.176 |
| somo_somo_elst_corrected_signed_kcal_mol | Local Contacts | 0.784 | 0.464 |
| somo_somo_elst_corrected_signed_kcal_mol | BLA & Asymmetry | 0.025 | 0.070 |

## Reading this

Impurity importance concentrates the shared contribution of correlated
descriptors on whichever member the trees select first, so within-group
ranking is not interpretable. Permutation importance distributes it more
evenly. The GROUP TOTAL and the group-level table are stable between the
two methods, so conclusions stated at group level are robust.
