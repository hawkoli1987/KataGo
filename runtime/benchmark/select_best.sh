#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <results.tsv> [out_dir]" >&2
  exit 1
fi

RESULTS="$1"
OUT_DIR="${2:-$(dirname "${RESULTS}")}"

if [ ! -f "${RESULTS}" ]; then
  echo "Missing results.tsv: ${RESULTS}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

BEST_TSV="${OUT_DIR}/best.tsv"
BEST_YAML="${OUT_DIR}/best_config.yaml"

echo -e "category\tbackend\tthreads\tbatch_size\tvisits\tnum_positions\tboard_size\tvisits_per_sec\tgpu_util_avg\tgpu_util_max\tgpu_mem_avg\tgpu_mem_max\tlog_path" > "${BEST_TSV}"

awk -F'\t' 'NR==1{next}
{
  v=$7+0
  if ($1=="cuda" && v>cuda_v) {cuda_v=v; cuda=$0}
  if ($1=="trt" && v>trt_v) {trt_v=v; trt=$0}
  if (v>best_v) {best_v=v; best=$0}
}
END{
  if (best!="") print "overall\t" best;
  if (cuda!="") print "cuda\t" cuda;
  if (trt!="") print "trt\t" trt;
}' "${RESULTS}" >> "${BEST_TSV}"

{
  echo "best:"
  tail -n +2 "${BEST_TSV}" | while IFS=$'\t' read -r category backend threads batch_size visits num_positions board_size visits_per_sec gpu_util_avg gpu_util_max gpu_mem_avg gpu_mem_max log_path; do
    echo "  ${category}:"
    echo "    backend: \"${backend}\""
    echo "    threads: ${threads}"
    echo "    batch_size: ${batch_size}"
    echo "    visits: ${visits}"
    echo "    num_positions: ${num_positions}"
    echo "    board_size: ${board_size}"
    echo "    visits_per_sec: ${visits_per_sec}"
    echo "    gpu_util_avg: ${gpu_util_avg}"
    echo "    gpu_util_max: ${gpu_util_max}"
    echo "    gpu_mem_avg: ${gpu_mem_avg}"
    echo "    gpu_mem_max: ${gpu_mem_max}"
    echo "    log_path: \"${log_path}\""
  done
} > "${BEST_YAML}"

echo "Wrote ${BEST_TSV}"
echo "Wrote ${BEST_YAML}"
