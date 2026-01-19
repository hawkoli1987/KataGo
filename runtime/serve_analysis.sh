#!/bin/bash
set -euo pipefail

ROOT_DIR="/scratch/Projects/SPEC-SF-AISG/source_files/KataGo"
CONFIG_YAML="${ROOT_DIR}/runtime/config.yaml"
PORT="$(grep -E '^ *port:' "${CONFIG_YAML}" | awk '{print $2}' | tr -d '\"')"

if [ -z "${PORT}" ]; then
  echo "Missing port in runtime/config.yaml" >&2
  exit 1
fi

INPUT_DIR="${ROOT_DIR}/runtime/assets/analysis/inputs"
OUTPUT_DIR="${ROOT_DIR}/runtime/assets/analysis/outputs"
SERVER_LOG="${ROOT_DIR}/runtime/analysis_server.log"

mkdir -p "${OUTPUT_DIR}"

uvicorn runtime.analysis_server:app --host 0.0.0.0 --port "${PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  status="$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/analysis" || true)"
  if [ "${status}" != "000" ]; then
    break
  fi
  sleep 1
done

read -r -p "Input basename (without .jsonl): " BASENAME
INPUT_PATH="${INPUT_DIR}/${BASENAME}.jsonl"
OUTPUT_PATH="${OUTPUT_DIR}/${BASENAME}.jsonl"

if [ ! -f "${INPUT_PATH}" ]; then
  echo "Missing input file: ${INPUT_PATH}" >&2
  exit 1
fi

curl -s -X POST "http://127.0.0.1:${PORT}/analysis" --data-binary "@${INPUT_PATH}" > "${OUTPUT_PATH}"
echo "Saved output to ${OUTPUT_PATH}"
echo "Curl example:"
echo "curl -s -X POST http://127.0.0.1:${PORT}/analysis --data-binary @${INPUT_PATH}"

wait "${SERVER_PID}"
