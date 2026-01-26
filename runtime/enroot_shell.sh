#!/bin/bash
# Start an interactive shell inside the KataGo enroot container
# Run this after getting an interactive PBS session, or directly on the compute node.
#
# Usage:
#   1. Get interactive PBS session: qsub -I -l select=1:mem=200gb:ncpus=32:ngpus=1:host=hopper-34 -l walltime=04:00:00 -q AISG_debug
#   2. Run this script: bash /scratch/Projects/SPEC-SF-AISG/source_files/KataGo/runtime/enroot_shell.sh

set -euo pipefail

# Configuration
export SHARED_FS="/scratch_aisg/SPEC-SF-AISG"
export SHARED_FS2="/scratch/Projects/SPEC-SF-AISG"
export SQSH_DIR="${SHARED_FS}/sqsh"
export SQSH_FILE="${SQSH_DIR}/go.sqsh"
export CONTAINER_NAME="katago_dev"
export ENROOT_DATA_PATH="${SHARED_FS}/.enroot/data"
export HOST_NVIDIA_MOUNT="/opt/host-nvidia"
export HOST_LIBCUDA="/lib64/libcuda.so.1"
export HOST_LIBNVML="/lib64/libnvidia-ml.so.1"
export LD_LIBRARY_PATH="${HOST_NVIDIA_MOUNT}:${LD_LIBRARY_PATH:-}"

mkdir -p "${ENROOT_DATA_PATH}"

if [ ! -f "${SQSH_FILE}" ]; then
  echo "Missing ${SQSH_FILE}. Run runtime/enroot_create.pbs first." >&2
  exit 1
fi

# Create container if it doesn't exist
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "Creating container ${CONTAINER_NAME}..."
    enroot create -n "${CONTAINER_NAME}" "${SQSH_FILE}"
fi

# Ensure mount target exists inside the container rootfs
enroot start --root --rw \
    --mount="${HOME}:${HOME}" \
    --mount="${SHARED_FS}:${SHARED_FS}" \
  "${CONTAINER_NAME}" \
  /bin/bash -lc "mkdir -p ${HOST_NVIDIA_MOUNT}"

echo "=============================================="
echo "Starting interactive shell in KataGo container"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

# Start interactive shell
exec enroot start --root --rw \
    --env LD_LIBRARY_PATH \
    --mount="${HOME}:${HOME}" \
    --mount="${SHARED_FS}:${SHARED_FS}" \
    --mount="${SHARED_FS2}:${SHARED_FS2}" \
    --mount="${HOST_LIBCUDA}:${HOST_NVIDIA_MOUNT}/libcuda.so.1" \
    --mount="${HOST_LIBNVML}:${HOST_NVIDIA_MOUNT}/libnvidia-ml.so.1" \
    --mount="/usr/bin/nvidia-smi:/usr/bin/nvidia-smi" \
  "${CONTAINER_NAME}" \
  /bin/bash -l
