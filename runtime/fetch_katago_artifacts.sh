#!/bin/bash
set -euo pipefail

BASE=/scratch/Projects/SPEC-SF-AISG/katago
# Downloads CUDA/TRT KataGo binaries plus a specific model into ${BASE}.
mkdir -p "${BASE}/bin" "${BASE}/models"

CUDA_URL=https://github.com/lightvector/KataGo/releases/download/v1.16.4/katago-v1.16.4-cuda12.1-cudnn8.9.7-linux-x64.zip
TRT_URL=https://github.com/lightvector/KataGo/releases/download/v1.16.4/katago-v1.16.4-trt8.6.1-cuda12.1-linux-x64.zip
MODEL_URL=https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s12223529728-d5663671073.bin.gz

curl -L -o /tmp/katago_cuda.zip "${CUDA_URL}"
curl -L -o /tmp/katago_trt.zip "${TRT_URL}"

unzip -q -o /tmp/katago_cuda.zip -d "${BASE}/bin/katago-cuda"
unzip -q -o /tmp/katago_trt.zip -d "${BASE}/bin/katago-trt"

CUDA_BIN="$(find "${BASE}/bin/katago-cuda" -type f -name katago | head -n 1)"
TRT_BIN="$(find "${BASE}/bin/katago-trt" -type f -name katago | head -n 1)"

ln -sf "${CUDA_BIN}" "${BASE}/bin/katago-cuda/katago"
ln -sf "${TRT_BIN}" "${BASE}/bin/katago-trt/katago"

curl -L -o "${BASE}/models/kata1-b28c512nbt-s12223529728-d5663671073.bin.gz" "${MODEL_URL}"
chmod +x "${BASE}/bin/katago-cuda/katago" "${BASE}/bin/katago-trt/katago"
