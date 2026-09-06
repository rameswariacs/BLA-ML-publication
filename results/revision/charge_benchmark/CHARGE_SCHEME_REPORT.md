# Charge-scheme benchmark for the electrostatic correction

22 structures, four population analyses per state.

## Absolute interfragment Coulomb energy, by scheme

These are expected to differ; the schemes are not comparable in magnitude.

| Scheme | mean E_elst(BS) | mean E_elst(T) | correlation BS vs T |
|---|---|---|---|
| mulliken | 6.299 | 5.591 | 0.986919 |
| loewdin | 1.708 | 1.316 | 0.948194 |
| hirshfeld | 0.171 | 0.044 | 0.665521 |
| chelpg | 1.990 | 1.699 | 0.963111 |

## The quantity that enters Eq. 7: dE_elst = E_elst(BS) - E_elst(T)

| Scheme | mean dE_elst | mean abs deviation from Mulliken | max abs deviation |
|---|---|---|---|
| mulliken | +0.7077 | 0.0045 | 0.0103 |
| loewdin | +0.3916 | 0.3283 | 1.0585 |
| hirshfeld | +0.1276 | 0.5939 | 1.9477 |
| chelpg | +0.2914 | 0.4189 | 1.2601 |

All values in kcal mol^-1. For reference the model's own test-set MAE
on this target is 0.550 kcal mol^-1, so a deviation well below that is
immaterial to the reported results.
