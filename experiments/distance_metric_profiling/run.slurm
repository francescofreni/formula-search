#!/bin/bash -l
#SBATCH --job-name=distance-runtime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
run_root="${1:?Use profile.py submit to create the saved experiment}"
python_bin="${2:?Expected Python executable}"
module load StdEnv/2023 gcc/12.3
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=(srun "$python_bin" -u "$run_root/source/profile.py" worker "$run_root")
printf 'Effective command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
