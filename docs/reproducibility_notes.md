# Reproducibility Notes

This repository provides a clean publication-facing copy of the latest no-`<S^2>`-filter workflow.

## Included

- Final processed descriptor/target tables.
- Main workflow scripts and supporting validation scripts.
- Model metrics, validation summaries, feature-importance summaries, and diagnostic tables.
- Publication and SI figures.

## Not Included

- Raw ORCA output archives.
- ORCA `.gbw`, `.densities`, scratch, or temporary files.
- Large trained-model pickle files.
- Slurm logs and transient HPC output.

## Practical Rerun Strategy

The most direct rerun path is to use:

```text
data/processed/ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv
```

as the input table for the model, validation, feature-importance, and plotting scripts.

Some original scripts contain absolute paths from the HPC run. These should be edited if the workflow is rerun in a different environment.
