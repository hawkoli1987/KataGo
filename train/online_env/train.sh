#!/bin/bash
# KataGo Winrate Prediction Training Script
# Uses DAPO algorithm with Qwen3-8B
#
# This script runs inside the Singularity container on each node.
# Only LOG_DIR, JOB_ID, JOB_NAME, SHARED_FS, SHARED_FS2 are passed from train.pbs.

set -euox pipefail

# ============================================================================
# Parse arguments (no fallbacks)
# ============================================================================
num_gpus_pernode=$1
num_gpus=$2
num_node=$3
master_addr=$4
master_port=$5
node_rank=$6

echo "=== KataGo Winrate Training ==="
echo "=== Distributed Config ==="
echo "  num_gpus_pernode: ${num_gpus_pernode}"
echo "  num_node: ${num_node}"
echo "  node_rank: ${node_rank}"
echo "  master_addr: ${master_addr}"
echo "  master_port: ${master_port}"

# ============================================================================
# Directory paths
# ============================================================================
export CACHE_ROOT="${SHARED_FS}/cache"
export DATA_ROOT="${SHARED_FS}/data_yuli"
export verl_dir="${SHARED_FS}/yuli/ARF-Training/repos/verl"
export CKPT_DIR="${SHARED_FS}/ckpt/verl"
export PYTHONPATH="${verl_dir}"

# Custom reward function path
REWARD_FN_PATH="./reward.py"

# ============================================================================
# Cache directories
# ============================================================================
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0

# ============================================================================
# Training configuration
# ============================================================================
export WANDB_PROJECT="katago_winrate"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO

# ============================================================================
# Container-specific settings
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG_LEVEL=0
export NVTE_DEBUG=0

# ============================================================================
# Load tokens
# ============================================================================
source "$HOME/.hf_token"
source "$HOME/.wandb"

# ============================================================================
# Runtime directories
# ============================================================================
export RAY_TMPDIR="/tmp/ray_${JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

export HYDRA_RUN_DIR="${LOG_DIR}/hydra"
mkdir -p "${HYDRA_RUN_DIR}"

mkdir -p "${LOG_DIR}/nccl"
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl/$(hostname).log"

# ============================================================================
# Debug info
# ============================================================================
echo "=== Container Environment ==="
echo "Hostname: $(hostname)"
echo "Python: $(which python)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "verl_dir: ${verl_dir}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "REWARD_FN_PATH: ${REWARD_FN_PATH}"
echo "PWD: $(pwd)"

# Save full container env for debugging
printenv | sort > "${LOG_DIR}/env/EnvVar_container_$(hostname).log"

# ============================================================================
# Fix CUDA_VISIBLE_DEVICES if it contains UUIDs instead of indices
# ============================================================================
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    echo "Converting CUDA_VISIBLE_DEVICES from UUIDs to indices..."
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    INDICES=""
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        if [ -z "$INDICES" ]; then
            INDICES="$i"
        else
            INDICES="$INDICES,$i"
        fi
    done
    export CUDA_VISIBLE_DEVICES="$INDICES"
    echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
fi

# ============================================================================
# Validate key dependencies
# ============================================================================
python3 - <<'PY' || echo "WARNING: Some deps missing, continuing anyway..."
import torch, vllm
print(f"torch: {torch.__version__}, vllm: {vllm.__version__}")
try:
    import verl
    print(f"verl: OK (from {verl.__file__})")
except ImportError as e:
    print(f"verl: MISSING - {e}")
    raise

# Check that requests is available for reward function
import requests
print(f"requests: OK")
PY

# ============================================================================
# Verify custom reward function exists
# ============================================================================
if [ ! -f "${REWARD_FN_PATH}" ]; then
    echo "ERROR: Custom reward function not found at ${REWARD_FN_PATH}"
    exit 1
fi
echo "Custom reward function found: ${REWARD_FN_PATH}"

# ============================================================================
# Multi-node: Start Ray cluster
# ============================================================================
if [ "${num_node}" -gt 1 ]; then
    RAY_PORT=6379
    if [ "${node_rank}" -eq 0 ]; then
        echo "Starting Ray head node on port ${RAY_PORT}..."
        ray start --head --node-ip-address="${master_addr}" --port=${RAY_PORT} --num-cpus=8 --num-gpus=${num_gpus_pernode} --temp-dir="${RAY_TMPDIR}"
        sleep 10
        echo "Ray head started, waiting for ${num_node} workers to join..."
        sleep $((num_node * 20))
        echo "Ray cluster status:"
        ray status || true
        export RAY_ADDRESS="${master_addr}:${RAY_PORT}"
    else
        echo "Worker node ${node_rank}: waiting 20s for head node..."
        sleep 20
        echo "Connecting to Ray head at ${master_addr}:${RAY_PORT}..."
        for attempt in 1 2 3 4 5; do
            if ray start --address="${master_addr}:${RAY_PORT}" --node-ip-address="${master_addr}" --num-cpus=8 --num-gpus=${num_gpus_pernode} --temp-dir="${RAY_TMPDIR}"; then
                echo "Worker node ${node_rank} joined Ray cluster"
                break
            else
                echo "Attempt ${attempt} failed, retrying in 10s..."
                sleep 10
            fi
        done
        echo "Worker node ${node_rank} waiting for head to complete..."
        while true; do sleep 60; done
    fi
fi

# ============================================================================
# Configure Ray for veRL (multi-node only)
# ============================================================================
RAY_INIT_ARGS=""
if [ "${num_node}" -gt 1 ] && [ -n "${RAY_ADDRESS:-}" ]; then
    RAY_INIT_ARGS="+ray_kwargs.ray_init.address=auto"
    echo "Will connect to Ray cluster at RAY_ADDRESS=${RAY_ADDRESS}"
fi

# ============================================================================
# DAPO Algorithm Configuration
# ============================================================================
# DAPO: Extends GRPO with asymmetric clipping, token-level loss, dynamic sampling
ALGO_ARGS="
    algorithm.adv_estimator=grpo
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.use_kl_in_reward=False
    ++algorithm.filter_groups.enable=True
    ++algorithm.filter_groups.metric=acc
    ++algorithm.filter_groups.max_num_gen_batches=10
    ++actor_rollout_ref.actor.use_kl_loss=False
    ++actor_rollout_ref.actor.clip_ratio_low=0.2
    ++actor_rollout_ref.actor.clip_ratio_high=0.28
    ++actor_rollout_ref.actor.loss_agg_mode=token-mean
    critic.enable=False
    ++reward_model.reward_manager=dapo
"
echo "Algorithm: DAPO"
echo "Algorithm args: ${ALGO_ARGS}"

# ============================================================================
# Launch veRL Training for KataGo Winrate Prediction
# ============================================================================
python3 -m verl.trainer.main_ppo \
    hydra.run.dir="${HYDRA_RUN_DIR}" \
    ${RAY_INIT_ARGS} \
    \
    data.train_files="${DATA_ROOT}/RL/katago/train.parquet" \
    data.val_files="${DATA_ROOT}/RL/katago/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=64 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    \
    ${ALGO_ARGS} \
    \
    trainer.logger=wandb \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPT_DIR}/${JOB_NAME}" \
    trainer.resume_mode=disable \
    trainer.nnodes=${num_node} \
    trainer.n_gpus_per_node=${num_gpus_pernode} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee "${LOG_DIR}/verl.log"
