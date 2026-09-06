#!/bin/bash
#LAYOUT: threads
#SBATCH --job-name=baseline
#SBATCH --partition=base1
#SBATCH --nodes=1                  # single node, always
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # analysis is I/O bound, 2 cores is plenty
#SBATCH --mem=8G                   # actual use is well under 2G; 8G is headroom
#SBATCH --time=00:20:00            # runs in seconds, 20 min is generous
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ---------------------------------------------------------------------------
# System-mean baseline analysis.
#
# This job REFITS NOTHING. It reads prediction files already produced by the
# manuscript pipeline and computes, for each target, how well a model that
# knows only the radical family (and no geometry) would score. That number is
# the reference against which the descriptor-based models should be judged.
#
# Submit from the directory containing baseline_analysis.py:
#     mkdir -p logs
#     sbatch run_baseline.sbatch
#
# It is small enough to run directly on the login node if you prefer:
#     python baseline_analysis.py
#
# No scratch space is used, so nothing is written to /home/scratch and there is
# nothing to clean up afterwards. Nothing existing is modified or deleted; all
# output goes into a new baseline_analysis/ subdirectory.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

# --- environment -----------------------------------------------------------
# The 'usepanelc' alias written out in full. Aliases are not expanded in
# non-interactive shells, so `usepanelc` would fail here.
module purge
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate panelc
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "python: $(which python)"

# Pin thread pools to the allocation. numpy will otherwise start one thread per
# core on the whole 96-core node inside a 2-core allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python -c "import numpy, pandas, matplotlib; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)"

# --- run -------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python baseline_analysis.py

echo "Finished at $(date)"
echo "Results in: $(pwd)/baseline_analysis/"
