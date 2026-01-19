#!/bin/bash
set -euo pipefail

ROOT_DIR="/scratch/Projects/SPEC-SF-AISG/source_files/KataGo"
CONFIG_YAML="${ROOT_DIR}/runtime/config.yaml"
MATRIX_CSV="$(grep -E '^ *matrix_csv:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
RESULTS_DIR="$(grep -E '^ *results_dir:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
MODEL_PATH="$(grep -E '^ *model_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
CUDA_BIN="$(grep -E '^ *cuda_binary_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"
TRT_BIN="$(grep -E '^ *trt_binary_path:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"

if [ -z "${MATRIX_CSV}" ] || [ -z "${RESULTS_DIR}" ] || [ -z "${MODEL_PATH}" ]; then
  echo "Missing benchmark settings in runtime/config.yaml" >&2
  exit 1
fi

MATRIX_CSV="${ROOT_DIR}/${MATRIX_CSV}"
RESULTS_DIR="${ROOT_DIR}/${RESULTS_DIR}"

if [ ! -f "${MATRIX_CSV}" ]; then
  echo "Missing matrix csv: ${MATRIX_CSV}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUMMARY="${RUN_DIR}/summary.tsv"
echo -e "backend\tthreads\tbatch_size\tvisits\tnum_positions\tboard_size\tlog_path" > "${SUMMARY}"

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

  LOG_PATH="${RUN_DIR}/${backend}_t${threads}_b${batch_size}_v${visits}_n${num_positions}_s${board_size}.log"
  echo "Running ${backend} threads=${threads} batch=${batch_size} visits=${visits} positions=${num_positions} size=${board_size}"
  {
    echo "command: ${BIN_PATH} benchmark -model ${MODEL_PATH} -threads ${threads} -visits ${visits} -numpositions ${num_positions} -boardsize ${board_size} -fixed-batch-size ${batch_size}"
    "${BIN_PATH}" benchmark \
      -model "${MODEL_PATH}" \
      -threads "${threads}" \
      -visits "${visits}" \
      -numpositions "${num_positions}" \
      -boardsize "${board_size}" \
      -fixed-batch-size "${batch_size}"
  } > "${LOG_PATH}" 2>&1

  echo -e "${backend}\t${threads}\t${batch_size}\t${visits}\t${num_positions}\t${board_size}\t${LOG_PATH}" >> "${SUMMARY}"
done

echo "Benchmark run complete: ${RUN_DIR}"
