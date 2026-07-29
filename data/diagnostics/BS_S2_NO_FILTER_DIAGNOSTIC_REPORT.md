# Broken-Symmetry <S^2> Diagnostic Report: No-Filter Workflow

## Rule used for this run

- No structure is removed by an <S^2> criterion in this workflow.
- The historical diagnostic window `0.10 <= final <S^2> <= 1.80` is retained only to label low- or high-<S^2> structures.
- The value is the last `Expectation value of <S**2>` printed in the original GuessMix broken-symmetry ORCA output.
- Model fitting, cross-validation, feature analysis, and train/validation/test splitting use all cleaned modeling-eligible structures.

## Dataset accounting

- Structures before diagnostic labeling: **2582**
- Structures used for modeling: **2582**
- Low-<S^2> structures included: **52**
- High-<S^2> structures included: **0**

| System | Included | Low-<S^2> included | High-<S^2> included |
|---|---:|---:|---:|
| CPBP | 793 | 0 | 0 |
| fluorenyl | 363 | 4 | 0 |
| olympicenyl | 513 | 19 | 0 |
| phenalenyl | 235 | 17 | 0 |
| phenalenyl_olympicenyl | 626 | 12 | 0 |

## Low/high-<S^2> diagnostic structures included

| Structure | System | <S^2> | Minimum C...C distance (A) | Raw E_SOMO-SOMO (kcal/mol) | Reason |
|---|---|---:|---:|---:|---|
| fluorenyl_orca_0 | fluorenyl | 0.000000 | 2.4173 | -28.3933 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| fluorenyl_orca_1 | fluorenyl | 0.000000 | 2.4703 | -24.0098 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| fluorenyl_orca_2 | fluorenyl | 0.000000 | 2.3822 | -20.4985 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| fluorenyl_orca_7 | fluorenyl | 0.000000 | 2.3142 | -23.2775 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_15 | olympicenyl | 0.000000 | 2.8085 | -24.1462 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_16 | olympicenyl | 0.000000 | 2.7431 | -24.8090 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_17 | olympicenyl | 0.000000 | 2.7787 | -21.9195 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_18 | olympicenyl | 0.000000 | 2.7445 | -21.8184 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_19 | olympicenyl | 0.000000 | 2.7002 | -21.3165 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_20 | olympicenyl | 0.000000 | 2.7746 | -19.9781 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_21 | olympicenyl | 0.000000 | 2.6727 | -20.8133 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_22 | olympicenyl | 0.000000 | 2.6712 | -19.9054 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_23 | olympicenyl | 0.000000 | 2.5618 | -25.9497 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_26 | olympicenyl | 0.000000 | 2.5445 | -26.2201 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_28 | olympicenyl | 0.000000 | 2.4563 | -27.5886 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_30 | olympicenyl | 0.000000 | 2.5765 | -20.3539 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_31 | olympicenyl | 0.000000 | 2.4665 | -25.5736 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_34 | olympicenyl | 0.000000 | 2.4168 | -27.1638 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_37 | olympicenyl | 0.000000 | 2.5701 | -24.7175 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_40 | olympicenyl | 0.000000 | 2.6000 | -20.3267 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_42 | olympicenyl | 0.000000 | 2.5805 | -20.7445 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_43 | olympicenyl | 0.074586 | 2.6429 | -17.5425 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| olympicenyl_orca_83 | olympicenyl | 0.000000 | 2.5231 | -21.0522 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_0 | phenalenyl | 0.000000 | 2.7311 | -36.7612 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_1 | phenalenyl | 0.000000 | 2.6624 | -37.1593 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_2 | phenalenyl | 0.000000 | 2.6514 | -34.9656 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_3 | phenalenyl | 0.000000 | 2.5927 | -37.2059 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_4 | phenalenyl | 0.000000 | 2.2308 | -47.5817 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_5 | phenalenyl | 0.000000 | 2.1629 | -51.0452 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_7 | phenalenyl | 0.000000 | 2.5462 | -25.7715 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_8 | phenalenyl | 0.000000 | 2.4908 | -24.0735 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_10 | phenalenyl | 0.000000 | 2.4918 | -22.9675 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_12 | phenalenyl | 0.001183 | 2.5438 | -20.7331 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_16 | phenalenyl | 0.000001 | 2.4841 | -21.4230 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_19 | phenalenyl | 0.000590 | 2.4636 | -20.7151 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_22 | phenalenyl | 0.000000 | 2.1536 | -35.1424 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_25 | phenalenyl | 0.000000 | 2.1812 | -26.4401 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_27 | phenalenyl | 0.000000 | 2.2312 | -22.5885 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_29 | phenalenyl | 0.000000 | 2.2614 | -20.2216 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_orca_60 | phenalenyl | 0.064931 | 2.3934 | -18.5862 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_0 | phenalenyl_olympicenyl | 0.000000 | 2.4125 | -31.7101 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_1 | phenalenyl_olympicenyl | 0.000000 | 2.4928 | -29.4109 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_2 | phenalenyl_olympicenyl | 0.000000 | 2.5653 | -26.6920 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_3 | phenalenyl_olympicenyl | 0.000000 | 2.5243 | -27.9529 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_4 | phenalenyl_olympicenyl | 0.000000 | 2.1500 | -36.0085 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_5 | phenalenyl_olympicenyl | 0.000000 | 2.1362 | -42.1347 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_6 | phenalenyl_olympicenyl | 0.000000 | 2.1594 | -33.5434 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_7 | phenalenyl_olympicenyl | 0.000000 | 2.1834 | -31.5447 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_8 | phenalenyl_olympicenyl | 0.000000 | 2.1869 | -30.1761 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_9 | phenalenyl_olympicenyl | 0.000000 | 2.2330 | -25.3922 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_11 | phenalenyl_olympicenyl | 0.000000 | 2.2664 | -21.9455 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |
| phenalenyl_olympicenyl_orca_12 | phenalenyl_olympicenyl | 0.000000 | 2.3049 | -18.4575 | diagnostic only: final BS <S^2> below 0.10; retained in no-S2-filter workflow |

## Reproducibility files

- `BS_S2_ALL_STRUCTURE_AUDIT_NO_FILTER.csv`: every structure and its diagnostic label.
- `BS_S2_LOW_OR_HIGH_DIAGNOSTIC_STRUCTURES_INCLUDED.csv`: low/high-<S^2> structures retained in the model input.
- `ALL_DIMERS_19descriptors_7targets_NO_S2_FILTER.csv`: exact modeling input.
- `ALL_DIMERS_19descriptors_7targets_BS_filtered.csv`: compatibility copy of the same no-filter modeling input.
