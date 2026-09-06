#!/bin/bash
#LAYOUT: threads
#SBATCH --job-name=revision
#SBATCH --partition=base1
#SBATCH --nodes=1                  # single node, always
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # ExtraTrees uses n_jobs=-1; MLP is the slow part
#SBATCH --mem=16G                  # expected use well under this
#SBATCH --time=03:00:00            # expect under an hour; headroom for the MLP fits
#SBATCH --output=logs/revision_%j.out
#SBATCH --error=logs/revision_%j.err

# ---------------------------------------------------------------------------
# Three reviewer-requested analyses in one job:
#   A  leave-one-system-out + descriptor-coverage diagnostic   (Reviewer 2.1)
#   B  Ridge linear baseline                                   (Reviewer 2.2)
#   C  ExtraTrees permutation importance, individual + grouped (Reviewer 2.3, 2.4)
#
# Submit from anywhere at or below the run directory; the script walks up to
# find it. For example:
#
#     cd /home/rb1820/BLA-ML/FODFT_4Target_NO_S2_FILTER_Run_20260707_164211/Revision
#     mkdir -p logs
#     sbatch run_revision.sbatch
#
# Roughly 60 model fits. Nothing existing is modified; output goes to a new
# revision_analyses/ directory. No scratch space is used.
#
# If you want a fast first look, ExtraTrees alone finishes in a few minutes:
#     python revision_analyses.py --skip mlp
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- locate the run directory, mirroring what the script does --------------
RUNNAME=FODFT_4Target_NO_S2_FILTER_Run_20260707_164211
RUNDIR=""; d="$(pwd)"
while [[ "$d" != "/" ]]; do
    if [[ "$(basename "$d")" == "${RUNNAME}" ]]; then RUNDIR="$d"; break; fi
    if [[ -d "$d/${RUNNAME}" ]]; then RUNDIR="$d/${RUNNAME}"; break; fi
    d="$(dirname "$d")"
done
[[ -n "${RUNDIR}" ]] || { echo "ERROR: cannot find ${RUNNAME} from $(pwd)" >&2; exit 1; }
echo "Run directory: ${RUNDIR}"

for f in \
    "${RUNDIR}/ALL_DIMERS_19descriptors_4targets_FODFT_NO_S2_FILTER.csv" \
    "${RUNDIR}/unified_19descriptor_no_s2_filter_manifest.json" \
    "${RUNDIR}/train_validate_test/extra_trees/extra_trees_train_validate_test_predictions.csv"
do
    [[ -f "$f" ]] || { echo "ERROR: missing input $f" >&2; exit 1; }
done
echo "All three inputs found."

# --- environment -----------------------------------------------------------
# The 'usepanelc' alias written out in full. Aliases are not expanded in
# non-interactive shells, so calling usepanelc here would fail.
module purge
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate panelc
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "python: $(which python)"

# Pin thread pools to the allocation. Without this, numpy and the ExtraTrees
# n_jobs=-1 each try to use all 96 cores of the node inside a 16-core slot.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python -c "import sklearn, numpy, pandas, matplotlib; \
print('sklearn', sklearn.__version__, '| numpy', numpy.__version__, \
      '| pandas', pandas.__version__)"

# --- run -------------------------------------------------------------------
python revision_analyses.py

echo "Finished at $(date)"
echo "Results in: $(pwd)/revision_analyses/"
echo
echo "Record what it actually used:"
echo "  sacct -j ${SLURM_JOB_ID:-<jobid>} --format=JobID,State,ExitCode,MaxRSS,Elapsed"
