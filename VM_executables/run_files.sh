#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${WORKDIR}/logs"

RUN_SECONDS=300        # 5 minutes
IDLE_SECONDS=30        # 30 seconds (between runs)
INITIAL_IDLE=10        # 30 seconds (before first run)

mkdir -p "${LOGDIR}"

ts() { date +"%Y-%m-%d_%H-%M-%S"; }

run_one() {
  local name="$1"
  shift
  local cmd=( "$@" )

  local start
  start="$(ts)"
  local logfile="${LOGDIR}/${start}__${name}.log"

  echo "============================================================" | tee -a "${logfile}"
  echo "[${start}] START: ${name}" | tee -a "${logfile}"
  echo "CMD: ${cmd[*]}" | tee -a "${logfile}"
  echo "============================================================" | tee -a "${logfile}"

  set +e
  timeout --signal=SIGINT --kill-after=5 "${RUN_SECONDS}" "${cmd[@]}" >>"${logfile}" 2>&1
  rc=$?
  set -e

  local end
  end="$(ts)"
  echo "------------------------------------------------------------" | tee -a "${logfile}"
  echo "[${end}] END: ${name} (exit=${rc})" | tee -a "${logfile}"
  echo "------------------------------------------------------------" | tee -a "${logfile}"

  echo "[${end}] IDLE ${IDLE_SECONDS}s..." | tee -a "${logfile}"
  sleep "${IDLE_SECONDS}"
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v timeout >/dev/null 2>&1 || { echo "ERROR: timeout not found"; exit 1; }

cd "${WORKDIR}"

# -----------------------------
# Initial idle (baseline)
# -----------------------------
echo "[INFO] Initial idle for ${INITIAL_IDLE}s..."
sleep "${INITIAL_IDLE}"

# -----------------------------
# Workload files
# -----------------------------
MEM_A1="mem_stream.py"
MEM_A2="mem_pointer_chase.py"
MEM_A3="mem_alloc_touch_pages.py"

IO_B1="io_seq_fsync.py"
IO_B2="io_rand_rw.py"
IO_B3="io_many_files.py"

A1_ARGS=( --mb 128 --seconds "${RUN_SECONDS}" )
A2_ARGS=( --mb 1024 --seconds "${RUN_SECONDS}" --seed 123 )   # safer than 2048
A3_ARGS=( --objects 2000 --object-kb 256 --sleep-ms 20 --seconds "${RUN_SECONDS}" )

B1_ARGS=( --seconds "${RUN_SECONDS}" --kb 4096 --fsync-wait 1 --path "io_seq.bin" )
B2_ARGS=( --seconds "${RUN_SECONDS}" --file-mb 2048 --block-kb 64 --write-ratio 0.5 --path "io_rand.bin" --seed 123 )
B3_ARGS=( --seconds "${RUN_SECONDS}" --files-per-batch 500 --payload-bytes 1024 --seed 123 )


# -----------------------------
# Run sequence
# -----------------------------
run_one "A1_mem_stream"         "${PYTHON_BIN}" "${MEM_A1}" "${A1_ARGS[@]}"
run_one "A2_mem_pointer_chase"  "${PYTHON_BIN}" "${MEM_A2}" "${A2_ARGS[@]}"
run_one "A3_mem_alloc_touch"    "${PYTHON_BIN}" "${MEM_A3}" "${A3_ARGS[@]}"

run_one "B1_io_seq_fsync"       "${PYTHON_BIN}" "${IO_B1}"  "${B1_ARGS[@]}"
run_one "B2_io_rand_rw"         "${PYTHON_BIN}" "${IO_B2}"  "${B2_ARGS[@]}"
run_one "B3_io_many_files"      "${PYTHON_BIN}" "${IO_B3}"  "${B3_ARGS[@]}"

echo "[DONE] All workloads completed. Logs in: ${LOGDIR}"
