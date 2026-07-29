#!/bin/bash
#SBATCH --job-name=yperm_noS2
#SBATCH --partition=base1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --array=0-7%4
#SBATCH --output=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets/logs/yperm_%A_%a.out
#SBATCH --error=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets/logs/yperm_%A_%a.err

set -euo pipefail

PY=/home/rb1820/.conda/envs/panelc/bin/python
OUT=/home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/supporting_analyses/YRandomization_500x_FODFT_4Targets
SCRATCH_BASE=/home/scratch/rb1820/ML-scratch

mkdir -p "$OUT/logs" "$OUT/results_by_task" "$SCRATCH_BASE"
export TMPDIR=$SCRATCH_BASE/tmp_yperm_noS2_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$OUT"
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} \
  "$PY" y_randomization_control.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --n-permutations 500
