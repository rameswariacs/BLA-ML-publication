#!/bin/bash
#LAYOUT: tasks
#SBATCH --job-name=chargebm
#SBATCH --partition=base1
#SBATCH --array=1-44%8             # 44 single points, 8 at a time (8 x 20 = 160 cores)
#SBATCH --nodes=1                  # single node, always
#SBATCH --ntasks=20                # BS uses nprocs 20, triplet 16 (author's own
                                   # production settings); 20 covers both
#SBATCH --cpus-per-task=1
#SBATCH --mem=76G                  # ~3.8 GB/core x 20
#SBATCH --time=12:00:00
#SBATCH --output=logs/chargebm_%A_%a.out
#SBATCH --error=logs/chargebm_%A_%a.err

# ---------------------------------------------------------------------------
# Charge-scheme benchmark for the electrostatic correction (Reviewer 1.4).
#
# Runs one ORCA single point per array task. Each job produces Mulliken,
# Loewdin, Hirshfeld and CHELPG charges in the same output, so all four schemes
# come from one run.
#
# Setup:
#     python generate_charge_inputs.py --xyz-root /home/rb1820/BLA-ML
#     mkdir -p logs
#     sbatch run_charge_benchmark.sbatch
#
# Scratch goes to /home/scratch/<user>/<jobid>_<taskid> and is removed when the
# task ends. Set CLEAN_SCRATCH=0 below to retain it for debugging.
# ---------------------------------------------------------------------------

set -euo pipefail

BENCH="${SLURM_SUBMIT_DIR:-$(pwd)}/charge_benchmark"
JOBLIST="${BENCH}/job_list.txt"
[[ -f "${JOBLIST}" ]] || { echo "ERROR: ${JOBLIST} not found. Run generate_charge_inputs.py first." >&2; exit 1; }

JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBLIST}")
[[ -n "${JOB}" ]] || { echo "No job for array index ${SLURM_ARRAY_TASK_ID}"; exit 0; }
echo "Task ${SLURM_ARRAY_TASK_ID}: ${JOB} on $(hostname) at $(date)"

# --- modules ---------------------------------------------------------------
# ORCA is linked against a specific OpenMPI; 5.0.x and 6.x both expect 4.1.x.
module purge
module load openmpi/4.1.8
module load orca/6.1.1

# --- scratch ---------------------------------------------------------------
SCRATCH_ROOT="/home/scratch/${USER}"
SCRATCH="${SCRATCH_ROOT}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
[[ -d /home/scratch ]] || { echo "ERROR: /home/scratch missing on $(hostname)" >&2; exit 1; }
mkdir -p "${SCRATCH}"

CLEAN_SCRATCH=1
cleanup() {
    if (( CLEAN_SCRATCH )) \
       && [[ -n "${SCRATCH}" ]] \
       && [[ "${SCRATCH}" == "${SCRATCH_ROOT}/"* ]] \
       && [[ "${SCRATCH}" != "${SCRATCH_ROOT}" ]] \
       && [[ -d "${SCRATCH}" ]]; then
        rm -rf "${SCRATCH}"
    else
        echo "Scratch retained at ${SCRATCH} -- remove it manually."
    fi
}
trap cleanup EXIT

cp "${BENCH}/inputs/${JOB}.inp" "${SCRATCH}/"
cd "${SCRATCH}"

# --- run -------------------------------------------------------------------
# ORCA launches its own MPI: call it by absolute path, never via srun/mpirun.
ORCA_EXE=$(which orca)
STATUS=0
"${ORCA_EXE}" "${JOB}.inp" > "${JOB}.out" 2>&1 || STATUS=$?

OUTDIR="${BENCH}/outputs"
mkdir -p "${OUTDIR}"
cp "${JOB}.out" "${OUTDIR}/" 2>/dev/null || true

# ORCA can exit 0 on a failed SCF, so check the text.
if grep -q "ORCA TERMINATED NORMALLY" "${JOB}.out"; then
    echo "OK: ${JOB}"
else
    echo "FAILED: ${JOB} (see ${OUTDIR}/${JOB}.out)" >&2
    exit 1
fi
exit ${STATUS}
