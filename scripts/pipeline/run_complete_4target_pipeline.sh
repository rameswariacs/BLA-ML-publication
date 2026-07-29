#!/bin/bash
set -euo pipefail

ROOT=/home/rb1820/BLA-ML
CONTROL=$ROOT/FODFT_All5Dimers_FragmentCalculations
PIPELINE=$CONTROL/complete_4target_pipeline_NO_S2_FILTER
PANELC_PYTHON=${PANELC_PYTHON:-/home/rb1820/.conda/envs/panelc/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/FODFT_4Target_NO_S2_FILTER_Run_$(date +%Y%m%d_%H%M%S)}
SCRATCH_BASE=${SCRATCH_BASE:-/home/scratch/rb1820/ML-scratch}
REUSE_FODFT_CSV=${REUSE_FODFT_CSV:-$ROOT/FODFT_4Target_Complete_Run_20260629_211000/fodft_couplings/fodft_couplings_all.csv}
MANIFEST=$PIPELINE/all5_fragment_jobs_manifest.csv
CORE_SCRIPT=$PIPELINE/scripts/unified_19descriptor_bs_filtered_elstfix_pipeline.py

module purge
module load gcc
module load openmpi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TMPDIR="$SCRATCH_BASE/tmp_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$SCRATCH_BASE"

mkdir -p "$RESULTS_ROOT"/{logs,preflight,fodft_couplings,scripts}
cp "$0" "$RESULTS_ROOT/scripts/run_complete_4target_pipeline.sh"
cp "$PIPELINE"/scripts/*.py "$RESULTS_ROOT/scripts/"

echo "[$(date)] Results root: $RESULTS_ROOT"
echo "$RESULTS_ROOT" > "$PIPELINE/latest_results_path.txt"

if [[ -n "$REUSE_FODFT_CSV" && -s "$REUSE_FODFT_CSV" ]]; then
  echo "[$(date)] Reusing existing FO-DFT coupling CSV to minimize scratch and disk use: $REUSE_FODFT_CSV"
  cp "$REUSE_FODFT_CSV" "$RESULTS_ROOT/fodft_couplings/fodft_couplings_all.csv"
  {
    echo "Reused FO-DFT coupling CSV"
    echo "source=$REUSE_FODFT_CSV"
    echo "copied_to=$RESULTS_ROOT/fodft_couplings/fodft_couplings_all.csv"
  } | tee "$RESULTS_ROOT/logs/02_fodft_couplings_reused.log"
else
  echo "[$(date)] Preflight"
  "$PANELC_PYTHON" "$PIPELINE/scripts/preflight_fodft_inputs.py" \
    --fragment-root "$CONTROL" \
    --manifest "$MANIFEST" \
    --out-dir "$RESULTS_ROOT/preflight" \
    --strict | tee "$RESULTS_ROOT/logs/01_preflight.log"

  echo "[$(date)] FO-DFT coupling projection"
  "$PANELC_PYTHON" "$PIPELINE/scripts/calculate_fodft_couplings_all.py" \
    --manifest "$MANIFEST" \
    --out-dir "$RESULTS_ROOT/fodft_couplings" \
    --scratch-base "$SCRATCH_BASE" | tee "$RESULTS_ROOT/logs/02_fodft_couplings.log"
fi

echo "[$(date)] Core four-target Extra Trees + MLP workflow"
"$PANELC_PYTHON" "$PIPELINE/scripts/run_core_4target_fodft.py" \
  --core-script "$CORE_SCRIPT" \
  --fodft-csv "$RESULTS_ROOT/fodft_couplings/fodft_couplings_all.csv" \
  --out-dir "$RESULTS_ROOT" | tee "$RESULTS_ROOT/logs/03_core_models.log"

echo "[$(date)] FO-DFT numerical and chemical quality-control analysis"
"$PANELC_PYTHON" "$PIPELINE/scripts/analyze_fodft_coupling_qc.py" \
  --fodft-csv "$RESULTS_ROOT/fodft_couplings/fodft_couplings_all.csv" \
  --descriptor-csv "$RESULTS_ROOT/ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv" \
  --out-dir "$RESULTS_ROOT/supporting_analyses/FODFT_Coupling_QualityControl" \
  | tee "$RESULTS_ROOT/logs/03b_fodft_quality_control.log"

echo "[$(date)] Prepare supporting analysis scripts"
"$PANELC_PYTHON" "$PIPELINE/scripts/prepare_and_run_followups.py" \
  --final-dir "$RESULTS_ROOT" \
  --python "$PANELC_PYTHON" \
  --mode prepare | tee "$RESULTS_ROOT/logs/04_prepare_followups.log"

echo "[$(date)] Lightweight supporting analyses: top-7 plot, clustered validation, complexity analyses"
"$PANELC_PYTHON" "$PIPELINE/scripts/prepare_and_run_followups.py" \
  --final-dir "$RESULTS_ROOT" \
  --python "$PANELC_PYTHON" \
  --mode run-light | tee "$RESULTS_ROOT/logs/05_followups_light.log"

cat > "$RESULTS_ROOT/RUN_COMPLETE_NEXT_STEPS.md" <<EOF
# Complete four-target FO-DFT run, no S2 filter

Finished: $(date)

Main folder: \`$RESULTS_ROOT\`

Main dataset:
- \`ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv\`
- Compatibility copy: \`ALL_DIMERS_19descriptors_4targets_FODFT_BS_filtered.csv\`

Main model outputs:
- \`extra_trees/\`
- \`mlp/\`
- \`train_validate_test/\`

FO-DFT target outputs:
- \`fodft_couplings/fodft_couplings_all.csv\`
- \`fodft_couplings/fodft_coupling_summary.json\`

Supporting analyses prepared under:
- \`supporting_analyses/\`

Y-randomization is intentionally prepared but not launched inside this single job because 500 permutations x 8 tasks is long. Submit the copied array script from:
\`$RESULTS_ROOT/supporting_analyses/YRandomization_500x_FODFT_4Targets\`

No <S^2> filter is applied in this run. The legacy filenames containing \`BS_filtered\` are present only for compatibility with archived helper scripts. The canonical FO-DFT dataset is the \`NO_S2_FILTER\` 4-target file above.
EOF

echo "[$(date)] Complete. Results: $RESULTS_ROOT"
rm -rf "$TMPDIR"
