#!/bin/bash -l
# Resource limits and array indices are provided by experiments/run.py submit.
set -euo pipefail
run_root="${1:?Expected experiment result folder}"
python_bin="${2:?Expected Python executable}"
case_id="${SLURM_ARRAY_TASK_ID:?Expected array task ID}"

# Install before module setup so bootstrap failures reach the result table.
finish() {
    local exit_code=$?
    trap - EXIT
    if (( exit_code != 0 )); then
        local reporter="$python_bin"
        [[ -x "$reporter" ]] || reporter=python3
        "$reporter" -S "$run_root/source/experiments/run.py" record-failure \
            "$run_root" --case "$case_id" --exit-code "$exit_code" || \
            printf 'Could not record worker failure (exit %s).\n' "$exit_code" >&2
    fi
    exit "$exit_code"
}
trap finish EXIT

module load StdEnv/2023 gcc/12.3
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=(srun "$python_bin" "$run_root/source/experiments/run.py" worker
    "$run_root" --case "$case_id")
printf 'Effective command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
