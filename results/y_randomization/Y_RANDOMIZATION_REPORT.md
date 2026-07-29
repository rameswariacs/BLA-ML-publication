# Y-Randomization Control

The 19-descriptor matrix and original manuscript split were kept unchanged.
For each target and model, the target vector was globally permuted 500 times.
The manuscript-selected estimator was refitted on the original combined
train+validation rows and evaluated on the original test rows.

| Model | Target | Real R2 | Mean shuffled R2 | SD | Best shuffled R2 | Empirical p |
|---|---|---:|---:|---:|---:|---:|
| Extra Trees | |t_FO| | 0.9565 | -0.1315 | 0.0443 | -0.0079 | 0.001996 |
| Descriptor-based MLP | |t_FO| | 0.9517 | -0.0197 | 0.0156 | 0.0181 | 0.001996 |
| Extra Trees | Interaction energy | 0.9436 | -0.1368 | 0.0441 | 0.0090 | 0.001996 |
| Descriptor-based MLP | Interaction energy | 0.9362 | -0.0188 | 0.0163 | 0.0182 | 0.001996 |
| Extra Trees | Spin-conserving frontier gap | 0.9770 | -0.1401 | 0.0425 | -0.0222 | 0.001996 |
| Descriptor-based MLP | Spin-conserving frontier gap | 0.9657 | -0.0190 | 0.0162 | 0.0186 | 0.001996 |
| Extra Trees | Corrected SOMO–SOMO energy | 0.9636 | -0.1224 | 0.0558 | 0.0171 | 0.001996 |
| Descriptor-based MLP | Corrected SOMO–SOMO energy | 0.9698 | -0.0192 | 0.0172 | 0.0290 | 0.001996 |

Empirical p = (number of shuffled R2 values greater than or equal to the
real-target R2 + 1) / (500 + 1). The minimum attainable value is 1/501.
