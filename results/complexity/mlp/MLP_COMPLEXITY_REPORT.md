# MLP Architecture and Regularization Sensitivity Analysis

## Test-independent protocol

The established system-stratified 70/15/15 split was reconstructed using random seed 42. The 15% test subset was excluded from architecture and regularization selection. The combined training and validation pool was evaluated with fixed five-fold, system-stratified cross-validation.

Seven hidden-layer architectures were combined with three L2 regularization strengths, giving 21 configurations per target and 420 cross-validation fits for the four manuscript targets. Every fit used the manuscript preprocessing and optimization pipeline: median imputation, standardized descriptors, standardized target values, ReLU hidden units, linear output, Adam optimization, batch size 64, initial learning rate 0.001, and early stopping with a 15% internal validation fraction and 30-epoch patience.

Selection followed the one-standard-error rule. Among configurations with mean CV MAE no greater than the best mean MAE plus its standard error, the network with the fewest trainable parameters was chosen. Ties favored stronger L2 regularization.

## Candidate network sizes

| Hidden-layer architecture | Weights | Biases | Total trainable parameters |
|---|---:|---:|---:|
| 16 | 320 | 17 | 337 |
| 32 | 640 | 33 | 673 |
| 32-16 | 1136 | 49 | 1185 |
| 64-32 | 3296 | 97 | 3393 |
| 64-32-16 | 3792 | 113 | 3905 |
| 128-64-32 | 12704 | 225 | 12929 |
| 64-64-32-16 | 7888 | 177 | 8065 |

## Selected architectures

| Target | Lowest-MAE configuration | One-SE configuration | Parameters | CV MAE | CV R2 |
|---|---|---|---:|---:|---:|
| |t_FO| | 128-64-32, alpha=1e-02 | 128-64-32, alpha=1e-02 | 12929 | 0.0407248 | 0.9336 |
| Spin-conserving frontier gap | 128-64-32, alpha=1e-02 | 128-64-32, alpha=1e-02 | 12929 | 0.0449774 | 0.9700 |
| Interaction energy | 128-64-32, alpha=1e-03 | 128-64-32, alpha=1e-02 | 12929 | 0.858226 | 0.9339 |
| Corrected SOMO-SOMO energy | 128-64-32, alpha=1e-04 | 128-64-32, alpha=1e-02 | 12929 | 0.624973 | 0.9510 |

## Position of the manuscript MLP in the CV grid

The manuscript model uses hidden layers 128-64-32 with alpha = 1e-3 (12,929 trainable parameters).

| Target | Manuscript CV MAE | Best CV MAE | Difference | Within one SE? |
|---|---:|---:|---:|---|
| |t_FO| | 0.0416993 | 0.0407248 | 0.000974471 | Yes |
| Spin-conserving frontier gap | 0.0458722 | 0.0449774 | 0.000894837 | No |
| Interaction energy | 0.85647 | 0.85647 | 0 | Yes |
| Corrected SOMO-SOMO energy | 0.635869 | 0.620032 | 0.0158368 | Yes |

## Final held-out test comparison

| Target | Configuration | Architecture | Parameters | Test R2 | Test MAE | Test RMSE |
|---|---|---|---:|---:|---:|---:|
| |t_FO| | One-SE selected | 128-64-32 | 12929 | 0.9438 | 0.0403141 | 0.053976 |
| |t_FO| | Original saved model | mlp_128_64_32_a1e-4 | 12929 | 0.9517 | 0.037272 | 0.0500578 |
| Spin-conserving frontier gap | One-SE selected | 128-64-32 | 12929 | 0.9724 | 0.040793 | 0.0593157 |
| Spin-conserving frontier gap | Original saved model | mlp_64_64_32_16_a1e-4 | 8065 | 0.9657 | 0.048469 | 0.0660557 |
| Interaction energy | One-SE selected | 128-64-32 | 12929 | 0.9411 | 0.862975 | 1.20779 |
| Interaction energy | Original saved model | mlp_128_64_32_a1e-3 | 12929 | 0.9362 | 0.905481 | 1.25638 |
| Corrected SOMO-SOMO energy | One-SE selected | 128-64-32 | 12929 | 0.9678 | 0.549295 | 0.956861 |
| Corrected SOMO-SOMO energy | Original saved model | mlp_128_64_32_a1e-3 | 12929 | 0.9698 | 0.567082 | 0.928035 |

The independent test set was evaluated only after completion of the cross-validation-based selection and did not influence architecture choice.

## Training-size learning curves

| Target | MAE at 25% training size | MAE at 100% training size | Relative decrease |
|---|---:|---:|---:|
| |t_FO| | 0.0865556 | 0.0407248 | 52.9% |
| Spin-conserving frontier gap | 0.068549 | 0.0449774 | 34.4% |
| Interaction energy | 1.37988 | 0.858226 | 37.8% |
| Corrected SOMO-SOMO energy | 1.88326 | 0.624973 | 66.8% |

## Interpretation

This analysis distinguishes numerical parameter count from effective model complexity. L2 regularization and early stopping constrain all networks, while the one-standard-error rule tests whether smaller architectures retain statistically comparable validation performance. Training-size learning curves provide an additional check that predictive error decreases as more structures become available rather than being sustained by a small subset.

The manuscript 128-64-32 network is the lowest-MAE CV configuration for |t_FO| and corrected SOMO-SOMO energy and remains within one standard error of the best result for interaction energy. For the frontier gap, its mean CV MAE exceeds the strict one-SE threshold by only a small absolute amount. The sensitivity analysis therefore does not establish that 12,929 parameters are necessary for every target, but it shows that the reported architecture occupies the stable, high-performing region of the complexity grid. Retaining one common regularized architecture across the four manuscript targets is consequently defensible, provided it is described as a validated common architecture rather than the unique optimum for every property.
