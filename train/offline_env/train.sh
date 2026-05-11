#!/bin/bash
# KataGo Offline Training Script
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

echo "=== KataGo Offline Training ==="
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

# Custom reward function path (use absolute path based on LOG_DIR)
REWARD_FN_PATH="${LOG_DIR}/reward.py"

# ============================================================================
# Essential: Set data paths for offline training
# ============================================================================
# Read from config.yaml or use defaults
SCRIPT_DIR="${SHARED_FS}/yuli/ARF-Training/repos/KataGo"
CONFIG_FILE="${SCRIPT_DIR}/configs/config.yaml"
TRAIN_PARQUET="${SCRIPT_DIR}/data/RL/katago_offline/19x19/train.parquet"
TEST_PARQUET="${SCRIPT_DIR}/data/RL/katago_offline/19x19/test.parquet"

# Try to read from config if available
if [ -f "${CONFIG_FILE}" ]; then
    PREPROCESSED_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('${CONFIG_FILE}')); print(config.get('data', {}).get('preprocessed_dir', ''))" 2>/dev/null || echo "")
    if [ -n "${PREPROCESSED_DIR}" ]; then
        TRAIN_PARQUET="${SCRIPT_DIR}/${PREPROCESSED_DIR}/train.parquet"
        TEST_PARQUET="${SCRIPT_DIR}/${PREPROCESSED_DIR}/test.parquet"
    fi
fi

echo "Data files:"
echo "  Train: ${TRAIN_PARQUET}"
echo "  Test: ${TEST_PARQUET}"

# ============================================================================
# Cache directories
# ============================================================================
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0
export TIKTOKEN_RS_CACHE_DIR="${LOG_DIR}/tiktoken_rs_cache"
mkdir -p "${TIKTOKEN_RS_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# ============================================================================
# Training configuration
# ============================================================================
# Essential: Change project name for offline training
export WANDB_PROJECT="verl_train"
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

# Essential: Check yaml for offline reward function (instead of requests)
import yaml
print(f"yaml: OK")
PY

# ============================================================================
# Verify custom reward function exists
# ============================================================================
if [ ! -f "${REWARD_FN_PATH}" ]; then
    echo "ERROR: Custom reward function not found at ${REWARD_FN_PATH}"
    exit 1
fi
echo "Custom reward function found: ${REWARD_FN_PATH}"

# Get CPU count, but cap at reasonable value (Ray may fail if too high)
CPUS_PER_NODE=$(nproc)
# Cap at 64 to avoid Ray allocation issues in containers
if [ "${CPUS_PER_NODE}" -gt 64 ]; then
    CPUS_PER_NODE=64
fi
echo "CPUS_PER_NODE=${CPUS_PER_NODE} (from nproc, capped at 64)"

# ============================================================================
# Multi-node: Start Ray cluster
# ============================================================================
if [ "${num_node}" -gt 1 ]; then
    RAY_PORT=6379
    LOCAL_NODE_IP="$(hostname -I | awk '{print $1}')"
    RAY_HEAD_ADDR_FILE="${LOG_DIR}/ray_head_addr"
    if [ "${node_rank}" -eq 0 ]; then
        echo "Starting Ray head node on port ${RAY_PORT}..."
        ray stop --force || true
        sleep 2
        # Start Ray head in background to avoid blocking
        ray start --head --node-ip-address="${LOCAL_NODE_IP}" --port=${RAY_PORT} --num-cpus=${CPUS_PER_NODE} --num-gpus=${num_gpus_pernode} --temp-dir="${RAY_TMPDIR}" &
        sleep 10
        echo "${LOCAL_NODE_IP}" > "${RAY_HEAD_ADDR_FILE}"
        echo "Ray head started, waiting for ${num_node} workers to join..."
        # Wait for Ray to be ready before proceeding
        for i in {1..10}; do
            if ray status --address="${LOCAL_NODE_IP}:${RAY_PORT}" > /dev/null 2>&1; then
                echo "Ray cluster is ready"
                break
            fi
            echo "Waiting for Ray to be ready (attempt $i/10)..."
            sleep 5
        done
        sleep $((num_node * 20))
        echo "Ray cluster status:"
        ray status || true
        export RAY_ADDRESS="${LOCAL_NODE_IP}:${RAY_PORT}"
    else
        echo "Worker node ${node_rank}: waiting 20s for head node..."
        sleep 20
        while [ ! -s "${RAY_HEAD_ADDR_FILE}" ]; do
            echo "Waiting for Ray head address file at ${RAY_HEAD_ADDR_FILE}..."
            sleep 5
        done
        RAY_HEAD_ADDR="$(cat "${RAY_HEAD_ADDR_FILE}")"
        echo "Connecting to Ray head at ${RAY_HEAD_ADDR}:${RAY_PORT}..."
        ray stop --force || true
        for attempt in 1 2 3 4 5; do
            if ray start --address="${RAY_HEAD_ADDR}:${RAY_PORT}" --node-ip-address="${LOCAL_NODE_IP}" --num-cpus=${CPUS_PER_NODE} --num-gpus=${num_gpus_pernode} --temp-dir="${RAY_TMPDIR}" --disable-usage-stats; then
                echo "Worker node ${node_rank} joined Ray cluster"
                break
            else
                if [ "${attempt}" -eq 5 ]; then
                    echo "ERROR: Failed to join Ray cluster after 5 attempts"
                    exit 1
                fi
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
# Algorithm configuration (GRPO baseline)
# ============================================================================
ALGO="${ALGO:-grpo}"
ALGO_ARGS=""
case "${ALGO}" in
    grpo)
        # GRPO: Critic-free, uses group relative advantages, no KL penalty
        ALGO_ARGS="
            algorithm.adv_estimator=grpo
            algorithm.kl_ctrl.kl_coef=0.0
            algorithm.use_kl_in_reward=False
            critic.enable=False
        "
        ;;
    dapo)
        # Keep DAPO as optional fallback
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
        ;;
    *)
        echo "ERROR: Unknown algorithm '${ALGO}'. Must be grpo or dapo"
        exit 1
        ;;
esac
echo "Algorithm: ${ALGO}"
echo "Algorithm args: ${ALGO_ARGS}"

# ============================================================================
# Essential: Enable thinking mode for offline training
# ============================================================================
THINKING_ARGS="
    +data.apply_chat_template_kwargs.enable_thinking=True
"
echo "Thinking mode: ENABLED"

# ============================================================================
# Experiment configuration (baseline tuned for 2x H200)
# ============================================================================
# Model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

# Data
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# Rollout (vLLM)
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
ROLLOUT_LOG_PROB_BS="${ROLLOUT_LOG_PROB_BS:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-}"
VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-slow}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"

# Reference model
REF_LOG_PROB_BS="${REF_LOG_PROB_BS:-4}"

# Actor
ACTOR_MICRO_BS="${ACTOR_MICRO_BS:-8}"
ACTOR_MINI_BS="${ACTOR_MINI_BS:-512}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_LR_DECAY_STEPS="${ACTOR_LR_DECAY_STEPS:-1}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-False}"

# Actor distributed strategy: megatron or fsdp
ACTOR_STRATEGY="${ACTOR_STRATEGY:-megatron}"
ACTOR_TP_SIZE="${ACTOR_TP_SIZE:-4}"
ACTOR_PP_SIZE="${ACTOR_PP_SIZE:-1}"

# Training
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-0}"
SAVE_FREQ="${SAVE_FREQ:-60}"
TEST_FREQ="${TEST_FREQ:-10}"

echo "=== Experiment Config ==="
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "  MAX_PROMPT_LENGTH: ${MAX_PROMPT_LENGTH}"
echo "  MAX_RESPONSE_LENGTH: ${MAX_RESPONSE_LENGTH}"
echo "  GPU_MEMORY_UTIL: ${GPU_MEMORY_UTIL}"
echo "  ROLLOUT_LOG_PROB_BS: ${ROLLOUT_LOG_PROB_BS}"
echo "  ROLLOUT_N: ${ROLLOUT_N}"
echo "  ROLLOUT_QUANTIZATION: ${ROLLOUT_QUANTIZATION:-none}"
echo "  REF_LOG_PROB_BS: ${REF_LOG_PROB_BS}"
echo "  ACTOR_MICRO_BS: ${ACTOR_MICRO_BS}"
echo "  ACTOR_MINI_BS: ${ACTOR_MINI_BS}"
echo "  ACTIVATION_OFFLOAD: ${ACTIVATION_OFFLOAD}"
echo "  ENFORCE_EAGER: ${ENFORCE_EAGER}"
echo "  ROLLOUT_TP_SIZE: ${ROLLOUT_TP_SIZE}"
echo "  ACTOR_STRATEGY: ${ACTOR_STRATEGY}"
echo "  ACTOR_TP_SIZE: ${ACTOR_TP_SIZE}"
echo "  ACTOR_PP_SIZE: ${ACTOR_PP_SIZE}"
echo "  TOTAL_EPOCHS: ${TOTAL_EPOCHS}"
echo "  TOTAL_STEPS: ${TOTAL_STEPS}"

# ============================================================================
# Build strategy-specific arguments
# ============================================================================
STRATEGY_ARGS=""
BASE_CONFIG=""
if [ "${ACTOR_STRATEGY}" = "megatron" ]; then
    echo "Using Megatron backend with TP=${ACTOR_TP_SIZE}, PP=${ACTOR_PP_SIZE}"
    BASE_CONFIG="--config-name=ppo_megatron_trainer"
    STRATEGY_ARGS="
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP_SIZE}
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP_SIZE}
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP_SIZE}
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP_SIZE}
        critic.enable=False
    "
else
    echo "Using FSDP backend"
    STRATEGY_ARGS=""
fi

# Build quantization args (only if ROLLOUT_QUANTIZATION is set)
QUANTIZATION_ARGS=""
if [ -n "${ROLLOUT_QUANTIZATION}" ]; then
    echo "Enabling vLLM quantization: ${ROLLOUT_QUANTIZATION}"
    QUANTIZATION_ARGS="actor_rollout_ref.rollout.quantization=${ROLLOUT_QUANTIZATION}"
fi

# ============================================================================
# Launch veRL Training for KataGo Offline Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    ${BASE_CONFIG} \
    hydra.run.dir="${HYDRA_RUN_DIR}" \
    ${RAY_INIT_ARGS} \
    \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${TEST_PARQUET}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_activation_offload=${ACTIVATION_OFFLOAD} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${ACTOR_LR_DECAY_STEPS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_BS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=${VLLM_TOKENIZER_MODE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_BS} \
    ${STRATEGY_ARGS} \
    ${QUANTIZATION_ARGS} \
    ${THINKING_ARGS} \
    \
    reward_model.use_reward_loop=False \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    \
    ${ALGO_ARGS} \
    \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPT_DIR}/${JOB_NAME}" \
    trainer.resume_mode=disable \
    trainer.nnodes=${num_node} \
    trainer.n_gpus_per_node=${num_gpus_pernode} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} 2>&1 | tee "${LOG_DIR}/verl.log"
