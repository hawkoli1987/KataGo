# KataGo Offline Training

Train an LLM to predict root winrate and top move using offline KataGo analysis data.

## Overview

This training setup uses pre-computed KataGo analysis results stored in JSONL files, which are preprocessed into parquet format for veRL training.

| Component | Value |
|-----------|-------|
| Base Model | Configurable (Qwen3-8B, DeepSeekv3.2, Gemma3, etc.) |
| Algorithm | DAPO (GRPO variant) |
| Reward | Weighted sum of three components: r_wr, r_move, r_legal |
| Data Source | Preprocessed parquet files from `train/preprocess.py` |
| Thinking Mode | Always enabled |

## Data Pipeline

1. **Extract positions**: Run `make extract` to convert SGF files to position JSONL
2. **Analyze positions**: Run `make analyze_offline` to get KataGo analysis results
3. **Preprocess**: Run `make preprocess` to combine position and move data into parquet
4. **Train**: Run `make train` to start training

## Reward Function

The reward function (`train/offline_env/reward.py`) computes a weighted sum of three components:

### r_wr (Root Winrate Reward)
- **Range**: [-1, 0]
- **Computation**: Negative MSE between predicted and ground truth root winrate
- **Penalty**: -1 if move is illegal or format error

### r_move (Top Move Matching Reward)
- **Range**: {0, 1}
- **Computation**: 1 if LLM move matches KataGo top move (normalized to lowercase), 0 otherwise
- **Penalty**: 0 if move is illegal

### r_legal (Legal Move Reward)
- **Range**: {0, -1}
- **Computation**: 0 if move is legal, -1 if illegal (based on KataGo error response)

### Weighted Sum
```
reward_total = r_wr_weight * r_wr + r_move_weight * r_move + r_legal_weight * r_legal
```

Weights are configured in `configs/config.yaml` under `training.rewards`.

## WandB Logging

**Current Status**: veRL automatically logs the total reward as `reward/mean` and `reward/std`.

**Individual Reward Components**: The reward function stores individual components in `extra_info["_reward_components"]` for each sample, but veRL does not automatically log these to WandB. To log individual components, you would need to:

1. Modify veRL's reward manager to extract and log individual components
2. Or create a custom logging hook that reads from `extra_info`

**Note**: This limitation is documented in the refactor plan. The total reward is logged, which should be sufficient for monitoring training progress.

## Rollout Logging

**Current Status**: veRL does not have built-in support for saving sample rollouts at checkpoint intervals.

**Workaround**: You can manually extract samples from veRL's checkpoint directories or modify veRL to add a checkpoint callback.

**Desired Format**: `{LOG_DIR}/rollout.llm_log.jsonl` with fields:
- `timestamp`, `game_id`, `ply`, `prompt`, `raw_response`, `reasoning`, `parsed_move`, `win_rate`, `error`
- Plus reward fields: `r_wr`, `r_move`, `r_legal`, `reward_total`

## Thinking Mode

Thinking mode is always enabled for all supported model families:
- Qwen3
- DeepSeekv3.2
- Gemma3

The reward function extracts predictions from after the `</think>` tag.

## Usage

```bash
# 1. Preprocess data (required before training)
make preprocess

# 2. Submit training job
make train

# 3. Monitor training
# Check WandB dashboard or LOG_DIR/verl.log
```

## Configuration

Training configuration is in `configs/config.yaml`:
- `training.rewards`: Reward weights
- `data.preprocessed_dir`: Output directory for parquet files
- Model can be overridden via `MODEL_NAME` environment variable

## Testing

Test the reward function:
```bash
python train/offline_env/reward.py
```

Test data mapping:
```bash
pytest tests/test_data_mapping.py -v
```
