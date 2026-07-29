# Extra Trees Complexity-Sensitivity Analysis

## Test-independent protocol

The existing system-stratified 70/15/15 split was reconstructed with random seed 42. The 15% test subset was excluded from all hyperparameter comparisons. The combined training and validation pool was assessed by fixed five-fold system-stratified cross-validation. Median imputation was fitted independently within every fold.

The grid contained 7 tree counts (50, 100, 200, 300, 500, 750, and 1000) and 5 minimum leaf sizes (1, 2, 3, 5, and 10), giving 35 configurations and 175 fits per target. Four manuscript targets produced 700 cross-validation fits.

Selection used the one-standard-error rule. Configurations with mean CV MAE no greater than the minimum mean MAE plus its standard error were considered statistically competitive. Because minimum leaf size controls model flexibility more directly than ensemble size, the largest eligible leaf size was preferred, followed by the smallest eligible number of trees.

## Selected settings

| Target | Best mean-MAE setting | One-SE selected setting | CV MAE (selected) | CV R2 (selected) |
|---|---|---|---:|---:|
| |t_FO| | 1000 trees, leaf 1 | 100 trees, leaf 1 | 0.0361598 | 0.9403 |
| Spin-conserving frontier gap | 500 trees, leaf 1 | 100 trees, leaf 1 | 0.0359518 | 0.9793 |
| Interaction energy | 1000 trees, leaf 1 | 50 trees, leaf 1 | 0.764586 | 0.9457 |
| Corrected SOMO-SOMO energy | 200 trees, leaf 1 | 100 trees, leaf 1 | 0.518379 | 0.9633 |

## Position of the original 500-tree setting in the CV grid

| Target | Original CV MAE | Best CV MAE | Difference from best | Within one SE? |
|---|---:|---:|---:|---|
| |t_FO| | 0.0367419 | 0.0355449 | 0.00119701 | No |
| Spin-conserving frontier gap | 0.0371609 | 0.0356823 | 0.00147855 | No |
| Interaction energy | 0.786136 | 0.755123 | 0.0310133 | No |
| Corrected SOMO-SOMO energy | 0.541398 | 0.515967 | 0.0254304 | No |

## Final held-out test comparison

| Target | Configuration | Trees | Leaf size | Test R2 | Test MAE | Test RMSE |
|---|---|---:|---:|---:|---:|---:|
| |t_FO| | One-SE selected | 100 | 1 | 0.9582 | 0.032478 | 0.0465354 |
| |t_FO| | Original saved model | 500 | 2 | 0.9565 | 0.0336075 | 0.0474892 |
| Spin-conserving frontier gap | One-SE selected | 100 | 1 | 0.9782 | 0.0349782 | 0.052681 |
| Spin-conserving frontier gap | Original saved model | 500 | 2 | 0.9770 | 0.0363726 | 0.054062 |
| Interaction energy | One-SE selected | 50 | 1 | 0.9457 | 0.778106 | 1.15967 |
| Interaction energy | Original saved model | 500 | 2 | 0.9436 | 0.801965 | 1.18108 |
| Corrected SOMO-SOMO energy | One-SE selected | 100 | 1 | 0.9661 | 0.516743 | 0.982937 |
| Corrected SOMO-SOMO energy | Original saved model | 500 | 2 | 0.9636 | 0.547022 | 1.01843 |

The independent test metrics were calculated only after completion of the cross-validation-based selection and were not used to choose the settings.

## Interpretation

Tree-count curves reached a practical plateau well below 500 trees. Thus, 500 trees is not uniquely required, but it is a conservative convergence choice that reduces Monte Carlo variation at modest computational cost. Minimum leaf size had a substantially larger effect than tree count. Leaf sizes of 5 and 10 consistently underfit, whereas leaf sizes of 1 and 2 provided the strongest validation performance. The original 500-tree, two-sample-leaf model therefore lies in the stable high-performing region of the grid, although the strict one-SE rule selects smaller target-specific forests.
