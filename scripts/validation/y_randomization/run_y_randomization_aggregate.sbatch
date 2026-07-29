#!/bin/bash
#SBATCH --job-name=yperm_noS2_agg
#SBATCH --partition=base1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets/logs/yperm_aggregate_%j.out
#SBATCH --error=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets/logs/yperm_aggregate_%j.err

set -euo pipefail

PY=/home/rb1820/.conda/envs/panelc/bin/python
OUT=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets
SCRATCH_BASE=/home/scratch/rb1820/ML-scratch

mkdir -p "$OUT/logs" "$SCRATCH_BASE"
export TMPDIR=$SCRATCH_BASE/tmp_yperm_noS2_aggregate_${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

cd "$OUT"
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} \
  "$PY" aggregate_y_randomization.py
"$PY" make_zoomed_y_randomization_plots.py || true
