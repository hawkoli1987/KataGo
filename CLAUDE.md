# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a fork of the open-source [KataGo](https://github.com/lightvector/KataGo) Go engine extended with a custom **offline RL training pipeline** that teaches LLMs to play Go. An LLM (e.g. Qwen3-4B) is trained via GRPO/DAPO to predict KataGo's winrate and best move for positions taken from real SGF games. Evaluation is done via a **ladder system** that pits the LLM-based player against KataGo reference models of increasing strength.

The original KataGo C++ engine and Python selfplay training code live under `cpp/` and `python/`. Everything added for this project lives under `configs/`, `train/`, `eval/`, `runtime/`, and `tests/`.

## Single source of truth

**`configs/config.yaml`** is the central config for all paths, ports, reward weights, data limits, PBS settings, and KataGo binary/model locations. Most scripts read from it; do not hardcode values that belong there.

`configs/prompts.json` contains prompt templates used by the preprocessor.

## Common commands

All Python commands assume the `.venv` at the repo root is active (`source .venv/bin/activate`) unless noted otherwise. The Makefile handles activation automatically. For background jobs, always use the full path `.venv/bin/python` instead of `python` — shell activation does not carry into `nohup`/`&` subprocesses.

```bash
# Install Python dependencies
make install

# Start / stop KataGo analysis servers (candidate on :9200, reference on :9201)
bash runtime/manage_servers.sh start candidate   # submits PBS job, waits for HEALTHY
bash runtime/manage_servers.sh status            # check status + endpoint
bash runtime/manage_servers.sh stop candidate

# Full data + training pipeline (run in order):
make extract          # SGF → JSONL positions  (reads sgf_dir from config.yaml)
make analyze_offline  # positions → KataGo move analysis JSONL (server must be running)
make preprocess       # positions + moves → parquet for veRL (uses data.limit from config.yaml)
make train            # submit PBS job: qsub train/offline_env/train.pbs

# Evaluation tests (requires servers running)
make test-eval

# Run individual test files
pytest tests/test_offline_reward.py -v
pytest tests/test_eval.py -v

# Run reward function self-test
python train/offline_env/reward.py
```

## Architecture: data → training pipeline

### 1. Data collection (`runtime/`)
- `runtime/extract_positions.py` — parses SGF files into JSONL position records (board state, move history).
- `runtime/analyze_offline.py` — sends positions to the KataGo analysis HTTP API and writes move analysis JSONL (with `root_winrate` and `top_move` per position).
- `runtime/analysis_server.py` — FastAPI wrapper around the KataGo analysis CLI, run via PBS (`runtime/katago_server.pbs`) or interactively.
- `runtime/manage_servers.sh` — start/stop/wait/status helper for candidate (:9200) and reference (:9201) KataGo servers.

### 2. Preprocessing (`train/preprocess.py`)
Joins `data/positions/` and `data/moves/` JSONL files by `id`, formats prompts using `configs/prompts.json`, and writes `data/RL/katago_offline/19x19/train.parquet` + `test.parquet` for veRL. The `extra_info` column carries `root_winrate`, `top_move`, and `move_error` consumed by the reward function.

### 3. Offline RL training (`train/offline_env/`)
- `train.pbs` — PBS job script (2 nodes × 8 GPU H200). Reads overridable env vars (`MODEL_PATH`, `TRAIN_BATCH_SIZE`, etc.). Copies `train.sh`, `reward.py`, and `configs/` to `LOG_DIR` before launching.
- `train.sh` — runs inside the Singularity container. Sets up Ray for multi-node, configures GRPO or DAPO via `ALGO` env var, and launches `verl.trainer.main_ppo`.
- `reward.py` — veRL `custom_reward_function`. Computes a **three-component weighted reward**:
  - `r_wr` — negative MSE between predicted and ground-truth winrate (range `[-1, 0]`)
  - `r_move` — 1.0 if predicted move matches KataGo's top move, else 0.0
  - `r_legal` — 0.0 if move is legal, −1.0 if illegal
  - Weights are read from `configs/config.yaml → training.rewards`.

### 4. Evaluation (`eval/`)
- `eval/ladder.py` — runs a ladder evaluation: the candidate plays KataGo reference models at increasing ELO levels; promotion threshold is configurable in `config.yaml`.
- `eval/players/` — `OpenAIPlayer` (vLLM/OpenAI-compatible) and `KataGoPlayer` (HTTP).
- `eval/game.py` — game loop handling GTP-over-HTTP.
- Reference models are listed in `assets/models/manifest.json`.

## HPC / container workflow

The project runs on an HPC cluster (PBS scheduler, enroot containers).

- **Container image**: `go.sqsh` at `/scratch/Projects/SPEC-SF-AISG/sqsh/`. All Python deps (fastapi, uvicorn, pydantic, aiohttp, httpx, sgfmill, tqdm, pandas, pyarrow) are baked in. Rebuild with `qsub runtime/enroot_create.pbs` (runs on `AISG_large`, takes ~30 min).
- **Data directory**: `data/` is a symlink → `/scratch_aisg/SPEC-SF-AISG/data/RL/katago`. Must exist before running any data pipeline step.
- **Analysis server**: runs as a PBS batch job on whichever GPU node is allocated (not the login node). The serving hostname changes per job. Always use `bash runtime/manage_servers.sh status` to find the current endpoint — do not hardcode hostnames.
- **Enroot paths**: all enroot data must be on scratch, never `$HOME`. `ENROOT_DATA_PATH=/scratch_aisg/SPEC-SF-AISG/yuli/workdir/shared_fs/cache/.enroot`.
- **Interactive dev session**: `qsub -I -q AISG_debug -l select=1:mem=200gb:ncpus=32:ngpus=1`, then `bash runtime/enroot_start.pbs`.
- **Training job**: `make train` (or `qsub train/offline_env/train.pbs`). Logs go to `${SHARED_FS}/log/go/<JOB_NAME>/<JOB_ID>/`.
- Tokens required at runtime: `~/.hf_token` (HuggingFace) and `~/.wandb` (W&B), sourced inside `train.sh`.

## C++ engine

`cpp/` is the original KataGo C++ engine (MCTS + neural net inference). Build with CMake:

```bash
cd cpp
cmake . -DUSE_BACKEND=TENSORRT   # or CUDA, OPENCL, EIGEN
make -j4
```

Precompiled TRT binary is at `assets/bin/katago-trt/katago`. Do not modify `cpp/` unless working on the engine itself.

## Linting

```bash
# Python (ruff)
ruff check .
ruff format .
```
