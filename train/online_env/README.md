# KataGo Winrate Estimation Training

Train an LLM to estimate Go game winrates using veRL (RLHF framework).

| Component | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen3-8B` |
| Algorithm | DAPO (GRPO variant) |
| Reward | `-MSE` vs KataGo engine ground truth |
| Dataset | 18,966 positions (17,069 train / 1,897 val) |

---

## 1. Data Pipeline

### 1.1 Source Data

The training data originates from **KataGo self-play games** available at:
- URL: `http://katagoarchive.org/kata1/traininggames/`
- Downloaded: `2025-01-20sgfs.tar.bz2` (~15,000 SGF games)
- Download script: `runtime/download_game_data.sh`

### 1.2 Data Extraction Process

The script `runtime/prepare_katago_data.py` converts SGF files to veRL parquet format:

1. **Parse SGF files**: Extract moves, metadata (komi, rules, board size), and embedded winrate evaluations from move comments
2. **Filter games**: Only 19x19 standard games
3. **Sample positions**: Select 3 positions per game from moves 20-200 (mid-game focus)
4. **Adjust perspective**: Convert Black-centric winrates to current-player perspective
5. **Build prompts**: Create system + user message with position JSON
6. **Store KataGo query**: Save original query in `extra_info.katago_query` for real-time engine calls
7. **Split data**: 90% train / 10% validation, shuffled randomly
8. **Output**: Parquet files at `/scratch_aisg/SPEC-SF-AISG/data_yuli/RL/katago/`

---

## 2. Training Logic

### 2.1 veRL Data Processing

veRL loads parquet files and processes each row:
- `prompt`: List of messages `[{"role": "system", ...}, {"role": "user", ...}]`
- Applied via HuggingFace chat template to create input tokens
- `extra_info`: Passed to reward function for ground truth lookup

### 2.2 Ground Truth via KataGo Engine

The reward function (`train/reward.py`) makes real-time HTTP requests:
1. Extracts `katago_query` from `extra_info`
2. POSTs to `http://hopper-34:9000/analysis`
3. Parses `rootInfo.winrate` from response
4. Returns winrate for current player to move

Requests are **synchronous** - each sample blocks until response received.

### 2.3 Thinking Mode Status

**Current configuration**: Thinking mode is **DISABLED**

Evidence from vLLM initialization log:
- `reasoning_parser=''` (empty)
- `enable_in_reasoning=False`

To enable thinking mode, add to rollout config:
```
++actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template_kwargs.enable_thinking=True
```

Note: Enabling thinking requires updating `reward.py` to parse `<think>...</think>` blocks.

### 2.4 Response Extraction

The `extract_winrate_from_response()` function in `reward.py`:
1. Searches for JSON pattern `{"winrate": X.XX}`
2. Falls back to standalone float 0.XX
3. Falls back to percentage format (e.g., "56%")
4. Returns `None` if no valid winrate found

### 2.5 Reward Computation

```
reward = -(predicted_winrate - ground_truth_winrate)²
```

| Scenario | Reward |
|----------|--------|
| Perfect prediction | 0.0 |
| Off by 0.1 | -0.01 |
| Off by 0.5 | -0.25 |
| Invalid format | -1.0 |
| Engine error | -0.25 |

---

## 3. WandB Metrics

### 3.1 Key Metrics to Monitor

Based on actual logged data from run `ukpdf2ux`:

| Metric | Description | What to look for |
|--------|-------------|------------------|
| `reward/mean` | Average reward per batch | Should increase toward 0 |
| `reward/std` | Reward variance | Should decrease (consistent predictions) |
| `actor/loss` | Policy gradient loss | Should decrease |
| `actor/grad_norm` | Gradient magnitude | Should stabilize |
| `timing/step` | Seconds per step | Currently ~85s |

### 3.2 Signs of Effective Training

- `reward/mean` trending upward (less negative)
- `reward/std` decreasing (more consistent)
- `actor/loss` decreasing over time
- No NaN values in any metric

### 3.3 Response Sampling

**Current status**: Sample logging is disabled (`log_val_generations=0`)

To enable response sampling during validation:
```
trainer.log_val_generations=10
```

This logs 10 sample responses per validation run to WandB.

---

## 4. Future Work

### 4.1 Async HTTP Requests for Throughput

**Problem**: Current reward computation is sequential (~30ms per request × 16 batch = ~0.5s overhead).

**Solution**: Use `aiohttp` with `asyncio.gather()` for parallel requests.

**Estimated gain**: 
- Current: 16 sequential requests × 30ms = 480ms
- Async: 16 parallel requests = 30-50ms (10x faster)
- Net step time reduction: ~5-10%

**Implementation path**:
1. Modify `reward.py` to use async `get_ground_truth_winrate_async()`
2. Use `asyncio.run()` wrapper in `compute_score()`
3. Batch requests with `asyncio.gather(*tasks)`

### 4.2 Pre-compute Ground Truth

For maximum throughput, pre-compute all winrates offline:
1. Run all positions through KataGo engine once
2. Store winrate in parquet `ground_truth` column
3. Modify reward function to use stored value instead of HTTP call

**Estimated gain**: Eliminates network latency entirely (~10-15% step time).

### 4.3 Enable Thinking Mode

To train with chain-of-thought reasoning:
1. Add `enable_thinking=True` to rollout config
2. Update `max_response_length` to 256+ for reasoning tokens
3. Modify `reward.py` to extract answer after `</think>` tag

### 4.4 Increase Batch Size

Current memory-constrained settings:
- `train_batch_size=16`
- `ppo_mini_batch_size=4`
- `tensor_parallel_size=2`

Options to increase throughput:
- Request node with 8 GPUs instead of 4
- Use `tensor_parallel_size=4` with more GPUs
- Reduce `max_prompt_length` if positions allow

### 4.5 Qualitative Evaluation Script

Create an offline evaluation script to:
1. Load checkpoint from `/scratch_aisg/SPEC-SF-AISG/ckpt/verl/katago_winrate/global_step_N/`
2. Generate responses for sample positions
3. Compare predicted vs ground truth winrate
4. Save to CSV with columns: position_id, predicted, ground_truth, error

---

## 5. File Structure

```
repos/KataGo/
├── runtime/
│   ├── download_game_data.sh      # Download KataGo self-play SGFs
│   ├── prepare_katago_data.py     # Convert SGF → veRL parquet
│   └── assets/analysis/inputs/    # Downloaded SGF data
└── train/
    ├── train.pbs                  # PBS job submission script
    ├── train.sh                   # Training launch script (DAPO)
    ├── reward.py                  # Custom reward function
    └── README.md                  # This file

data_yuli/RL/katago/
├── train.parquet                  # Training data (17,069 positions)
└── test.parquet                   # Validation data (1,897 positions)

ckpt/verl/katago_winrate/
└── global_step_N/                 # Checkpoints saved every 10 steps
```

---

## 6. Quick Commands

```bash
# Check job status
qstat -u huangyl | grep katago

# View training progress
tail -30 /scratch_aisg/SPEC-SF-AISG/log/verl/katago_winrate/165461/verl.log

# View WandB dashboard
# https://wandb.ai/aisg-arf/katago_winrate/runs/ukpdf2ux

# Test reward function
cd /scratch_aisg/SPEC-SF-AISG/yuli/ARF-Training/repos/KataGo/train
python reward.py
```
