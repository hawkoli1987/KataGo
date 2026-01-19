#!/bin/bash
set -euo pipefail

ROOT_DIR="/scratch/Projects/SPEC-SF-AISG/source_files/KataGo"
CONFIG_YAML="${ROOT_DIR}/runtime/config.yaml"
MATRIX_CSV="$(grep -E '^ *matrix_csv:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
RESULTS_DIR="$(grep -E '^ *results_dir:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
MODEL_PATH="$(grep -E '^ *model_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
CUDA_BIN="$(grep -E '^ *cuda_binary_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
TRT_BIN="$(grep -E '^ *trt_binary_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
CONFIG_PATH="$(grep -E '^ *config_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
GPU_LOG_INTERVAL="$(grep -E '^ *gpu_log_interval_sec:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
GPU_UTIL_THRESHOLD="$(grep -E '^ *gpu_util_threshold:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
GPU_MEM_THRESHOLD="$(grep -E '^ *gpu_mem_threshold:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"

if [ -z "${MATRIX_CSV}" ] || [ -z "${RESULTS_DIR}" ] || [ -z "${MODEL_PATH}" ]; then
  echo "Missing benchmark settings in runtime/config.yaml" >&2
  exit 1
fi

GPU_MEM_TOTAL=""
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia-smi available but no GPU detected. Ensure container started with --nv." >&2
    exit 1
  fi
  GPU_MEM_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || true)"
  if [ -z "${GPU_MEM_TOTAL}" ]; then
    echo "nvidia-smi failed to report total memory. Ensure host NVML libs are mounted." >&2
    exit 1
  fi
else
  echo "warning: nvidia-smi not found; GPU utilization will not be logged" >&2
fi

if [[ "${MATRIX_CSV}" != /* ]]; then
  MATRIX_CSV="${ROOT_DIR}/${MATRIX_CSV}"
fi
if [[ "${RESULTS_DIR}" != /* ]]; then
  RESULTS_DIR="${ROOT_DIR}/${RESULTS_DIR}"
fi
if [[ -n "${MODEL_PATH}" && "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${ROOT_DIR}/${MODEL_PATH}"
fi
if [[ -n "${CONFIG_PATH}" && "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
fi
if [[ -n "${CUDA_BIN}" && "${CUDA_BIN}" != /* ]]; then
  CUDA_BIN="${ROOT_DIR}/${CUDA_BIN}"
fi
if [[ -n "${TRT_BIN}" && "${TRT_BIN}" != /* ]]; then
  TRT_BIN="${ROOT_DIR}/${TRT_BIN}"
fi
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-1}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-90}"
GPU_MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-90}"

if [ ! -f "${MATRIX_CSV}" ]; then
  echo "Missing matrix csv: ${MATRIX_CSV}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUMMARY="${RUN_DIR}/summary.tsv"
echo -e "backend\tthreads\tbatch_size\tvisits\tnum_positions\tboard_size\tvisits_per_sec\tgpu_util_avg\tgpu_util_max\tgpu_mem_avg\tgpu_mem_max\tlog_path" > "${SUMMARY}"

tail -n +2 "${MATRIX_CSV}" | while IFS=',' read -r backend threads batch_size visits num_positions board_size; do
  case "${backend}" in
    cuda)
      BIN_PATH="${CUDA_BIN}"
      ;;
    trt)
      BIN_PATH="${TRT_BIN}"
      ;;
    *)
      echo "Unknown backend '${backend}' in ${MATRIX_CSV}" >&2
      exit 1
      ;;
  esac

  if [ -z "${BIN_PATH}" ] || [ ! -x "${BIN_PATH}" ]; then
    echo "Missing or non-executable binary for backend '${backend}': ${BIN_PATH}" >&2
    exit 1
  fi

  if [ ! -f "${MODEL_PATH}" ]; then
    echo "Missing model file: ${MODEL_PATH}" >&2
    exit 1
  fi

  BIN_DIR="$(dirname "${BIN_PATH}")"
  ACTIVE_CONFIG="${CONFIG_PATH}"
  if [ -z "${ACTIVE_CONFIG}" ]; then
    ACTIVE_CONFIG="${BIN_DIR}/default_gtp.cfg"
  fi
  if [ ! -f "${ACTIVE_CONFIG}" ]; then
    echo "Missing config file: ${ACTIVE_CONFIG}" >&2
    exit 1
  fi

  LOG_PATH="${RUN_DIR}/${backend}_t${threads}_b${batch_size}_v${visits}_n${num_positions}_s${board_size}.log"
  GPU_LOG="${RUN_DIR}/${backend}_t${threads}_b${batch_size}_v${visits}_n${num_positions}_s${board_size}_gpu.csv"
  echo "Running ${backend} threads=${threads} batch=${batch_size} visits=${visits} positions=${num_positions} size=${board_size}"
  {
    echo "command: ${BIN_PATH} benchmark -config ${ACTIVE_CONFIG} -model ${MODEL_PATH} -threads ${threads} -visits ${visits} -numpositions ${num_positions} -boardsize ${board_size} -fixed-batch-size ${batch_size}"
    GPU_MON_PID=""
    : > "${GPU_LOG}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l "${GPU_LOG_INTERVAL}" > "${GPU_LOG}" &
      GPU_MON_PID=$!
    fi
    "${BIN_PATH}" benchmark \
      -config "${ACTIVE_CONFIG}" \
      -model "${MODEL_PATH}" \
      -threads "${threads}" \
      -visits "${visits}" \
      -numpositions "${num_positions}" \
      -boardsize "${board_size}" \
      -fixed-batch-size "${batch_size}"
    if [ -n "${GPU_MON_PID}" ]; then
      kill "${GPU_MON_PID}" || true
      wait "${GPU_MON_PID}" 2>/dev/null || true
    fi
  } > "${LOG_PATH}" 2>&1

  if grep -E "CUDA Error|terminate called|Error:" "${LOG_PATH}" >/dev/null 2>&1; then
    echo "Benchmark failed for ${backend} (see ${LOG_PATH})" >&2
    exit 1
  fi

  visits_per_sec="$(grep -Eo 'visits/(s|sec)[[:space:]]*=[[:space:]]*[0-9]+(\\.[0-9]+)?' "${LOG_PATH}" | head -n 1 | awk -F'=' '{gsub(/ /, "", $2); print $2}')"
  if [ -z "${visits_per_sec}" ]; then
    echo "Could not parse visits/s from ${LOG_PATH}" >&2
    exit 1
  fi

  gpu_util_avg="$(awk -F',' '{sum+=$1; n++} END{if(n>0) printf "%.2f", sum/n; else print "0"}' "${GPU_LOG}")"
  gpu_util_max="$(awk -F',' 'max<$1{max=$1} END{if(NR>0) print max; else print "0"}' "${GPU_LOG}")"
  gpu_mem_avg="$(awk -F',' -v t="${GPU_MEM_TOTAL}" '{sum+=$2; n++} END{if(n>0) printf "%.2f", (sum/n)/t*100; else print "0"}' "${GPU_LOG}")"
  gpu_mem_max="$(awk -F',' -v t="${GPU_MEM_TOTAL}" 'max<$2{max=$2} END{if(NR>0) printf "%.2f", max/t*100; else print "0"}' "${GPU_LOG}")"

  echo -e "${backend}\t${threads}\t${batch_size}\t${visits}\t${num_positions}\t${board_size}\t${visits_per_sec}\t${gpu_util_avg}\t${gpu_util_max}\t${gpu_mem_avg}\t${gpu_mem_max}\t${LOG_PATH}" >> "${SUMMARY}"

  if awk -v u="${gpu_util_max}" -v m="${gpu_mem_max}" -v ut="${GPU_UTIL_THRESHOLD}" -v mt="${GPU_MEM_THRESHOLD}" 'BEGIN{exit !(u<ut || m<mt)}'; then
    echo "Warning: GPU not saturated for ${backend} t=${threads} b=${batch_size} (util_max=${gpu_util_max} mem_max=${gpu_mem_max})" | tee -a "${RUN_DIR}/warnings.log"
  fi
done

echo "Benchmark run complete: ${RUN_DIR}"
