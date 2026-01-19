#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
SUMMARY="${RUN_DIR}/summary.tsv"

if [ ! -f "${SUMMARY}" ]; then
  echo "Missing summary.tsv in ${RUN_DIR}" >&2
  exit 1
fi

OUT="${RUN_DIR}/results.tsv"
echo -e "backend\tthreads\tbatch_size\tvisits\tnum_positions\tboard_size\tvisits_per_sec\tgpu_util_avg\tgpu_util_max\tgpu_mem_avg\tgpu_mem_max\tlog_path" > "${OUT}"

tail -n +2 "${SUMMARY}" | while IFS=$'\t' read -r backend threads batch_size visits num_positions board_size visits_per_sec gpu_util_avg gpu_util_max gpu_mem_avg gpu_mem_max log_path; do
  if [ ! -f "${log_path}" ]; then
    echo "Missing log file: ${log_path}" >&2
    exit 1
  fi

  if [ -z "${visits_per_sec}" ]; then
    echo "Missing visits/s in summary for ${log_path}" >&2
    exit 1
  fi

  echo -e "${backend}\t${threads}\t${batch_size}\t${visits}\t${num_positions}\t${board_size}\t${visits_per_sec}\t${gpu_util_avg}\t${gpu_util_max}\t${gpu_mem_avg}\t${gpu_mem_max}\t${log_path}" >> "${OUT}"
done

echo "Wrote ${OUT}"
