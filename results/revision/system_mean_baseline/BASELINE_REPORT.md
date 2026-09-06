# System-mean baseline analysis

The baseline predicts each structure's **molecular-system mean** and uses
none of the 19 geometric descriptors. Means are taken from training data only.

The baseline is identical for both model families because it never uses a model.

## five fold cv

| Target | Baseline R² | ExtRa Trees R² | ET gain | MLP R² | MLP gain |
|---|---|---|---|---|---|
| fodft_coupling_abs_ev | 0.027 | 0.941 | +0.914 | 0.933 | +0.906 |
| frontier_gap_ev | 0.908 | 0.977 | +0.069 | 0.969 | +0.061 |
| interaction_energy_kcal_mol | 0.617 | 0.946 | +0.329 | 0.934 | +0.317 |
| somo_somo_elst_corrected_signed_kcal_mol | 0.033 | 0.959 | +0.926 | 0.958 | +0.924 |

## train test 70 15 15

| Target | Baseline R² | ExtRa Trees R² | ET gain | MLP R² | MLP gain |
|---|---|---|---|---|---|
| fodft_coupling_abs_ev | -0.007 | 0.957 | +0.963 | 0.952 | +0.959 |
| frontier_gap_ev | 0.911 | 0.977 | +0.066 | 0.966 | +0.054 |
| interaction_energy_kcal_mol | 0.595 | 0.944 | +0.349 | 0.936 | +0.341 |
| somo_somo_elst_corrected_signed_kcal_mol | 0.003 | 0.964 | +0.960 | 0.970 | +0.967 |

## How to read this

A **low baseline** means the property is not determined by which radical family
the dimer belongs to, so the model's accuracy comes from packing geometry.
A **high baseline** means most of the model's R² was available without using any
descriptor, and the remaining gain is the true geometric contribution.
