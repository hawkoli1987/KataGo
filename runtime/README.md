# Container lifecycle
- Create the container image (host, no allocation needed):
  `bash -n runtime/enroot_create.pbs`
  Prereqs: `go.sqsh` exists at `/scratch/Projects/SPEC-SF-AISG/sqsh/go.sqsh`.

- Start an interactive container session:
  1) Request an interactive allocation:
     `qsub -I -q AISG_debug -l select=1:mem=200gb:ncpus=32:ngpus=1:host=hopper-34`
  2) From tmux `go`, enter the container:
     `bash runtime/enroot_start.pbs`
  Prereqs: run from tmux `go` on `hopper-34`. This script mounts host `libcuda`
  and `libnvidia-ml` so `nvidia-smi` works inside the container.

# Artifact fetch helpers
- Download KataGo CUDA/TRT binaries + model:
  `bash runtime/fetch_katago_artifacts.sh`
  Prereqs: network access, `curl`, `unzip`. Downloads a fixed model + binaries.
  Runs on host or inside container.

- List latest KataGo release assets (CUDA/TRT):
  `python3 runtime/get_katago_release_assets.py`
  Prereqs: network access. Optional args: `--repo`, `--include`, `--kinds`.

- Scrape model URLs from katagotraining.org:
  `python3 runtime/get_katago_model_urls.py`
  Prereqs: network access. Use to discover alternate model URLs.
  Optional args: `--base-url`, `--limit`.

# Benchmarks
- Run benchmark matrix:
  `bash runtime/benchmark/run_benchmarks.sh`
  Prereqs: run inside the container via `runtime/enroot_start.pbs`.
  `runtime/config.yaml` must point to valid model/binaries and config.

- Extract results (optional):
  `bash runtime/benchmark/extract_results.sh runtime/benchmark/results/<run_id>`
  Prereqs: `summary.tsv` exists in the run directory.

- Select best config:
  `bash runtime/benchmark/select_best.sh runtime/benchmark/results/<run_id>/summary.tsv`
  Prereqs: `summary.tsv` from a completed run.

# Analysis service
- Start an interactive container session:
  `bash runtime/enroot_start.pbs`
  Prereqs: run from tmux `go` on `hopper-34`.

- Start the FastAPI analysis server in a PBS job:
  `bash runtime/enroot_start.pbs --serve 9000`
  Prereqs: run from tmux `go` on the login node.
  This invokes `uvicorn runtime.analysis_server:app` inside the container.

- Send a test request to the running server:
  `bash runtime/test_analysis.sh [input.jsonl] [output.jsonl] [host:port]`
  Defaults: input/output to `assets/analysis/inputs/lategame_19.jsonl` and
  `assets/analysis/outputs/lategame_19.jsonl`, host to
  `localhost:<config port>`.

- Sample inputs/outputs:
  Inputs: `assets/analysis/inputs/*.jsonl`
  Outputs: `assets/analysis/outputs/*.jsonl`


# Eval

## How KataGo Match/Gatekeeper Works

### Commands Available
1. **`katago match`** - General-purpose match between multiple bots (different models or search settings)
2. **`katago gatekeeper`** - Automated evaluation for training pipeline: tests candidate models against current best

### Gatekeeper System (cpp/command/gatekeeper.cpp)
**Note:** This is NOT a multi-level system. It's a binary test: candidate vs current best.

| Aspect | Description |
|--------|-------------|
| **Model Loading** | `NNEvaluator` loads neural net from file. Candidate from `test-models-dir`, baseline from `accepted-models-dir` |
| **Pairing** | `MatchPairer` creates alternating Black/White matchups. Each bot plays both sides evenly |
| **Game Execution** | `GameRunner.runGame()` plays a single game using `Search` (MCTS) to select moves |
| **Games per Test** | Configured via `numGamesPerGating`. Early termination if outcome already decided |
| **Win Threshold** | `requiredCandidateWinProp` (default 0.5). Ties favor candidate |
| **Output** | SGFs to `sgf-output-dir/<modelName>/`. Logs win/loss counts |
| **Result** | Winner moved to `accepted-models-dir`, loser to `rejected-models-dir` |

### Key C++ Classes
- `NNEvaluator` - Wraps neural network, provides policy/value predictions
- `Search` - MCTS search using NNEvaluator for node expansion
- `MatchPairer::BotSpec` - Contains {botIdx, botName, nnEval, searchParams}
- `GameRunner` - Orchestrates game setup and execution

## LLM Evaluation Plan

### Goal
Evaluate an LLM's Go-playing strength by having it play against KataGo reference models in a ladder-style competition with Elo rating.

### Architecture
Python wrapper approach: Create scripts that mediate games between an LLM player and KataGo (in GTP mode), saving SGFs and computing Elo ratings.

---

## Scripts to Implement

### 1. `eval/download_reference_models.py`
Download historical KataGo models representing different strength levels.

**Functionality:**
- Fetch model metadata from katagotraining.org (or use curated list)
- Download selected models to `eval/assets/models/`
- Models should span a range of strengths (e.g., early training checkpoints to latest)
- Name files with strength tier (e.g., `level_01_b6c96.bin.gz`, `level_10_kata1-b40.bin.gz`)

**Model Selection Criteria:**
- Pick ~10-15 models spanning the full training history
- Include known Elo ratings if available from KataGo training logs
- Ensure models are compatible with current KataGo binary

**Output:**
```
assets/models/
├── level_01_<name>.bin.gz   # ~1000 Elo (early training)
├── level_02_<name>.bin.gz   # ~1500 Elo
├── ...
├── level_10_<name>.bin.gz   # ~3500 Elo (strongest)
└── manifest.json            # {level: {path, approx_elo, sha256, source_url}}
```

---

### 2. `eval/llm_player.py`
LLM player abstraction supporting multiple backends (text-only LLM).

**Backends:**
```python
class LLMPlayer(ABC):
    @abstractmethod
    def get_move(self, move_history: list, rules: str, komi: float, color: str) -> str:
        """Return move in KataGo GTP format (e.g., 'D4', 'Q16', 'pass')"""
        pass

class OpenAICompatiblePlayer(LLMPlayer):
    """OpenAI-compatible API (vLLM, OpenAI, etc.)"""
    def __init__(self, api_base: str, model: str, api_key: str = None): ...

class HuggingFacePlayer(LLMPlayer):
    """Direct HuggingFace model loading"""
    def __init__(self, model_name_or_path: str, device: str = "cuda"): ...
```

**Prompt Design (TBD):**
```
You are playing Go as {color}. The board is 19x19.

Rules: {rules}
Komi: {komi}

Move history: {move_history}

Your turn. Output ONLY your move in GTP coordinate format (e.g., "D4", "Q16", "pass").
Move:
```

**Input Format (KataGo format):**
- NO board state visualization - LLM must track board state from move history
- Move history as list of [color, coord] pairs: `[["B", "D4"], ["W", "Q16"], ["B", "D16"], ...]`
- Rules as explicit full rule string (see Game Variations table below)
- Komi as float

**Output Format:**
- Move in GTP coordinate format: letter (A-T, skipping I) + number (1-19)
- Examples: `D4`, `Q16`, `C10`, `pass`

**Move Validation:**
- Parse LLM response, extract coordinate
- If invalid or illegal move → **forfeit the game immediately** (count as loss)

---

### 3. `eval/eval.py`
Main evaluation script: ladder-style competition pipeline.

**Usage:**
```bash
python eval/eval.py \
  --candidate-type openai \
  --candidate-endpoint https://your-vllm-endpoint.example.com/v1 \
  --candidate-model "deepseek-ai/DeepSeek-V3.2" \
  --model-name "my-go-llm-v1" \
  --games-per-level 48 \
  --promotion-threshold 0.55
```

**Or with HuggingFace:**
```bash
python eval/eval.py \
  --candidate-type huggingface \
  --candidate-model "path/to/checkpoint" \
  --model-name "my-go-llm-v1" \
  --games-per-level 48 \
  --promotion-threshold 0.55
```

**CLI Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--model-name` | **Yes** | Name for this evaluation run (used as output folder name) |
| `--candidate-type` | Yes | `openai` or `huggingface` |
| `--candidate-model` | Yes | Model name/path |
| `--candidate-endpoint` | If openai | API base URL |
| `--games-per-level` | No | Default: 48 (should be multiple of 48) |
| `--promotion-threshold` | No | Default: 0.55 |

**Ladder Competition Flow:**
```
1. Load manifest.json to get reference models sorted by strength (level 1 = weakest)
2. Start at level 1
3. For current level:
   a. Play games_per_level games against reference model
      - Cover all combinations of: rules, komi, side
      - Fixed: 19x19 board size
   b. Compute win rate and update Elo estimate
   c. If win rate >= promotion_threshold:
      - Log: "Promoted from level X to level X+1"
      - Move to next level, repeat step 3
   d. Else:
      - Log: "Stopped at level X"
      - Terminate ladder
4. Output final Elo rating and summary
```

**Fixed Game Variations per Level:**
Each level plays `games_per_level` games covering ALL combinations:
| Variation | Values |
|-----------|--------|
| Rules | See explicit rule strings below |
| Komi | 5.5, 6.5, 7.5 |
| Side | Candidate as Black, Candidate as White |
| Board | **Fixed 19x19** |

**Explicit Rule Strings (KataGo format):**
| Name | Rule String |
|------|-------------|
| Japanese | `koSIMPLEscoreTERRITORYtaxSEKIsui0` |
| Chinese | `koSIMPLEscoreAREAtaxNONEsui0whbN` |
| Korean | `koPOSITIONALscoreAREAtaxNONEsui0whbN` |
| AGA | `koSITUATIONALscoreAREAtaxNONEsui0whbN-1` |
| New Zealand | `koSITUATIONALscoreAREAtaxNONEsui1` |
| Tromp-Taylor | `koPOSITIONALscoreAREAtaxNONEsui1` |
| Stone Scoring | `koSIMPLEscoreAREAtaxALLsui0` |
| Ancient Territory | `koSIMPLEscoreTERRITORYtaxALLsui0` |

Total combinations = 8 rules × 3 komis × 2 sides = **48 games minimum per level**.
Set `games_per_level` to a multiple of 48 for balanced coverage (e.g., 48, 96).

**Elo Calculation:**
- Use standard Elo formula with K-factor
- Reference models have known approximate Elo from KataGo training history
- Update candidate Elo after each game
- Final Elo = estimate after all games played

**Output:**
```
data/eval/<model_name>/
├── config.json              # Run configuration
├── games/
│   ├── level_01/
│   │   ├── game_001.sgf
│   │   ├── game_002.sgf
│   │   └── ...
│   ├── level_02/
│   └── ...
├── results.json             # Per-level win/loss, Elo progression
└── summary.json             # Final Elo, highest level reached, total games
```

**results.json schema:**
```json
{
  "candidate": {"type": "openai", "model": "my-go-llm"},
  "levels": [
    {
      "level": 1,
      "reference_model": "level_01_b6c96.bin.gz",
      "reference_elo": 1000,
      "games_played": 20,
      "wins": 18,
      "losses": 2,
      "win_rate": 0.90,
      "promoted": true,
      "candidate_elo_after": 1150
    },
    ...
  ],
  "final_elo": 2450,
  "highest_level": 7,
  "total_games": 140,
  "stopped_reason": "win_rate_below_threshold"
}
```

---

## Implementation Order

1. **`eval/download_reference_models.py`** - Curate and download reference model ladder
2. **`eval/llm_player.py`** - LLM abstraction with OpenAI-compatible API and HuggingFace backends
3. **`eval/eval.py`** - Main ladder evaluation pipeline

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Move format | KataGo GTP format (`D4`, `Q16`) | Native KataGo format; unambiguous coordinates |
| Rule format | Explicit rule strings (`koSIMPLEscoreAREA...`) | Precise; no ambiguity in rule interpretation |
| LLM input | Move history + rules + komi only | No board visualization; LLM must track state from history |
| Invalid move handling | Forfeit game | Strict evaluation; LLM must produce valid moves |
| Board size | Fixed 19x19 | Standard professional size; simplifies comparison |
| Game variations | All 48 combinations per level | Ensures robust evaluation across all rule/komi/side combos |
| Promotion threshold | Configurable (default 55%) | Allows tuning strictness |
| Elo calculation | Per-game update | Provides progression visibility |
| Reference models | ~10-15 levels | Sufficient granularity for Elo estimation |
| Output folder | `--model-name` (mandatory) | Clear identification of evaluation runs |

## refactor; eval
1. the current test is full of mock data, please remove all mock data, use actual data for test@tests/test_eval.py:184-187 

2. in test, GAMES_PER_LEVEL_TEST should be 5, MAX_LEVELS_TEST should be 3, MIN_VALID_MOVES should be 10
@tests/test_eval.py:42-44

3. in all locations in this repo, where the katago binary and model is used, migrate the katago binaries and katago model weights into this repo, 
- binaries: /scratch/Projects/SPEC-SF-AISG/katago/bin -> assets/bin; 
- modes: /scratch/Projects/SPEC-SF-AISG/katago/models -> assets/models
e.g. @katago_player.py (22-24) @runtime/config.yaml:9-11 

4. always use tensorRT instead of cuda version of katago binary

5. @eval/katago_server.py seems to be duplicating the functions of @runtime/enroot_start.pbs. If so, let's remove it, and start the reference model using existing scripts with minor modifications

6. identify all scripts that creates the following, make sure they were all stored inside `log` instead, 
- gtp_logs
- go_engine.o<job_id>

## refactor: using venv for dev
- create pyproject.toml based on runtime/enroot_create.pbs, which captures all dependencies required to run applications in this repo
- build .venv and use it for dev and testing purpose
- modify the Makefile, to use it for orchestration, it should have the following capabilities:
  - 'make install': install the .venv based on pyproject.toml
  - 'make activate': activate the .venv environment
  - 'make run': start the katago analysis engine using runtime/enroot_start.pbs, and print its fastapi endpoint, 
  - 'make test', run all pytests inside 'tests'
  - 'make extract', extract sgf data from a directory into jsonl, following 'runtime/extract_positions.py' sample script
  - 'make train', train a model in veRL, by `qsub train/train.pbs`
- [DONE] persist the eval test results into `data/eval/_test_runs`

Remaining code issues to fix:
- eval/katago_server.py - duplicate/unused file that may need cleanup or removal
- train/reward.py - needs implementation per specifications below (r_wr, r_move, r_legal rewards)
- train/offline_env/ - needs creation with training scripts per specifications below
- pytest for verifying position_dir and move_dir have exact file-to-file and row-to-row mapping
- wandb integration for logging rewards during training

# Train
once you have obtained the 'positions' (input) and 'moves' (ground-truth provided by Katago offline analysis) data, please use the 19x19 data for veRL training. 
use the scripts in @train/online_env as a reference, which was used for online go analysis engine serving. 
create a new set of script in @train/offline_env, use it for training, and set the `train/offline_env.train.pbs` as the default training entrypoint in Makefile

modify the reward.py, such that it will load the prompt template from `configs/prompts.json`, which instructs the llm to predict the `root_winrate` (the win_rate before LLM's current turn of move) and `top_move` (the best LLM move for the current turn)
the reward.py should also load and expand the model prompt, with the variable values including following
  - rule_string
  - komi
  - current_player
  - move_history (loaded from `position_dir`)
the reward.py should have options to include different rewards, and use their weighted sum as the overall reward for each rollout. Each reward is computed based on LLM prediction and katago analysis result (loaded from `move_dir`). The weight of each loss should be read from the config.yaml
  - r_wr, ∈ [-1,0], negative MSE on the root winrate, before the current player move
  - r_move, ∈ {0, 1}, if the LLM move as the current player corresponds with katago's best move
  - r_legal, ∈ {0, -1}, if the LLM move is legal
Let's use a pytest to assert that the jsonl files in the position_dir and move_dir have exact file-to-file and row-to-row mapping. 
If that's confirmed, we can always retrieve the correct ground-truth for each input prompt, and compute the weighted-sum reward based on LLM prediction and ground-truth
In WanDB, let's log:
- the individual reward value during the training, for each train step
- the weighted sum reward during the training, for each train step
- the individual reward value during the test/validation (using the 'test' split of data), for each test step
- the weighted sum reward value during the test/validation (using the 'test' split of data), for each test step

need to enable thinking model of the model during rollout
need to selectively (at each checkpointing interval) save out one sample model rollout, into "<LOG_DIR>/rollout.llm_log.jsonl". The content should be using the same format as that in the eval pipeline, but with the addition fields containing the applicable rewards
wandb needs to log the 

---

## Refactor Plan and Clarifying Questions

### Clarifying Questions and Answers

1. **Prompt Template Format**: ✅ ANSWERED
   - Modify existing `"go_move"` template to predict both `root_winrate` AND `top_move` in a single response
   - Output format: JSON `{"root_winrate": <float>, "top_move": "<GTP_coordinate>"}`

2. **Reward Weights Configuration**: ✅ ANSWERED
   - Add to `configs/config.yaml` under `training.rewards` section as specified

3. **Move Validation (r_legal)**: ✅ ANSWERED
   - Use KataGo analysis endpoint error response to determine legality
   - If KataGo returns `{"error": "Illegal move ...", "field": "moves", "id": "..."}`, the move is illegal
   - Invalid format (e.g., "XX", "Z99") counts as illegal move
   - If move is illegal: `r_wr = -1`, `r_move = 0`, `r_legal = -1`

4. **Data Format and Mapping**: ✅ ANSWERED
   - Position format: `{"id": "...", "moves": [[color, coord], ...], "rules": "...", "komi": <float>, "boardXSize": <int>, "boardYSize": <int>, "analyzeTurns": [...]}`
   - Move format: KataGo analysis response with `rootInfo.winrate` and `moveInfos[0].move`, or error response `{"error": "...", "field": "...", "id": "..."}`
   - Pytest should verify all 4: directory structure, file names, row counts, and `id` field matching row-by-row

5. **Root Winrate Timing**: ✅ ANSWERED
   - Use `rootInfo.winrate` from KataGo response (already for current player)
   - Compare LLM's predicted `root_winrate` vs KataGo's `rootInfo.winrate` from move_dir data

6. **Top Move Matching (r_move)**: ✅ ANSWERED
   - Compare LLM's `top_move` vs KataGo's `moveInfos[0].move` (top move by order/visits)
   - Always normalize to lowercase for comparison (e.g., "D4" → "d4")

7. **WandB Logging**: ✅ ANSWERED
   - Log individual rewards: `train/reward/r_wr`, `train/reward/r_move`, `train/reward/r_legal`, `train/reward/total` (or `reward/...` if no existing `reward` namespace in veRL)
   - Validation: `val/reward/r_wr`, `val/reward/r_move`, `val/reward/r_legal`, `val/reward/total`
   - Log per global batch (train) and val global batch (validation)

8. **Rollout Logging Format**: ✅ ANSWERED
   - Include all eval pipeline fields PLUS reward fields: `r_wr`, `r_move`, `r_legal`, `reward_total`
   - Log one sample per checkpoint interval

9. **Thinking Mode**: ✅ ANSWERED
   - Always enable thinking mode
   - Handle different model families: Qwen3, DeepSeekv3.2, Gemma3 (config may vary per model)
   - Extract predictions from after `</think>` tag

10. **Data Split**: ✅ ANSWERED
    - Use file naming convention: `train.jsonl` and `test.jsonl` in both `position_dir` and `move_dir`
    - Set filepaths in config.yaml

11. **Makefile Integration**: ✅ ANSWERED
    - `make train` should point to `train/offline_env/train.pbs`
    - No separate target for online training (offline_env only)

12. **Reward Function Signature**: ✅ ANSWERED
    - Use preprocessor script `train/preprocess.py` to combine `position_dir` and `move_dir` JSONL files into parquet format
    - Preprocessor should:
      - Read position JSONL files (e.g., `19x19/train.jsonl`, `19x19/test.jsonl`)
      - Read corresponding move JSONL files with matching `id` fields
      - Combine data into parquet format with all needed fields in `extra_info`:
        - `position_id`: the `id` from position file
        - `root_winrate`: from `rootInfo.winrate` in move response (or None if error)
        - `top_move`: from `moveInfos[0].move` in move response (or None if error)
        - `move_error`: error message if KataGo returned error, None otherwise
        - `rule_string`, `komi`, `current_player`, `move_history`: from position data
      - Output parquet files ready for veRL training

### Refactor Plan

#### Phase 1: Data Structure and Validation
1. Create pytest test `tests/test_data_mapping.py` to verify position_dir and move_dir have exact file-to-file and row-to-row mapping
2. Test should verify:
   - Directory structure matches (same subdirectories like `19x19/`, `16x16/`, etc.)
   - File names match (e.g., `train.jsonl`, `test.jsonl`)
   - Row counts match
   - `id` fields match row-by-row
   - Position data structure is valid

#### Phase 2: Prompt Template Updates
1. Modify existing `"go_move"` template in `configs/prompts.json` to predict both `root_winrate` and `top_move`
2. Template should include variables: `rule_string`, `komi`, `current_player`, `move_history` (formatted as string from position data)
3. Output format should be JSON: `{"root_winrate": <float>, "top_move": "<GTP_coordinate>"}`
4. Update template to instruct LLM to predict winrate before making the move

#### Phase 3: Configuration Updates
1. Add reward weights section to `configs/config.yaml`:
   ```yaml
   training:
     rewards:
       r_wr_weight: 1.0      # Weight for root winrate reward
       r_move_weight: 1.0     # Weight for top move matching reward
       r_legal_weight: 1.0    # Weight for legal move reward
   ```
2. Add training configuration section for offline training:
   ```yaml
   training:
     offline_env:
       position_dir: "data/positions"
       move_dir: "data/moves"
       board_size: "19x19"  # Use 19x19 data for training
       train_file: "19x19/train.jsonl"  # Training split file
       test_file: "19x19/test.jsonl"    # Validation split file
   ```

#### Phase 4A: Preprocessor Script
1. Create `train/preprocess.py` script that:
   - Reads config.yaml to get `position_dir`, `move_dir`, `train_file`, `test_file` paths
   - For each split (train/test):
     - Load position JSONL file (e.g., `data/positions/19x19/train.jsonl`)
     - Load corresponding move JSONL file (e.g., `data/moves/19x19/train.jsonl`)
     - Verify row-by-row `id` matching (raise error if mismatch)
     - For each row pair:
       - Extract `root_winrate` from `rootInfo.winrate` (or None if error response)
       - Extract `top_move` from `moveInfos[0].move` (or None if error response)
       - Extract `move_error` from error message if present (or None)
       - Extract `current_player` from `rootInfo.currentPlayer` (or from position data)
       - Build prompt using template from `configs/prompts.json` with variables: `rule_string`, `komi`, `current_player`, `move_history`
       - Create parquet row with:
         - `prompt`: List of messages `[{"role": "system", ...}, {"role": "user", ...}]`
         - `extra_info`: Dict containing `position_id`, `root_winrate`, `top_move`, `move_error`, `rule_string`, `komi`, `current_player`, `move_history`
         - `ground_truth`: String representation of root_winrate (for veRL compatibility)
   - Save parquet files to output directory (e.g., `data/RL/katago_offline/train.parquet`, `test.parquet`)
   - Log statistics: total rows processed, error count, etc.

#### Phase 4B: Reward Function Refactoring
1. Create `train/offline_env/reward.py` based on `train/online_env/reward.py`
2. Modify `compute_score` function to:
   - Extract `root_winrate` and `top_move` from LLM response (after `</think>` tag if thinking mode enabled)
   - Load ground truth from `extra_info` (pre-populated by preprocessor):
     - `root_winrate`: ground truth winrate from KataGo
     - `top_move`: ground truth top move from KataGo
     - `move_error`: error message if move was illegal in KataGo analysis
   - Compute three rewards:
     - `r_wr`: negative MSE on root winrate (range [-1, 0]). If move is illegal (`move_error` present), return -1
     - `r_move`: 1 if LLM move matches KataGo top move (normalized to lowercase), 0 otherwise. If move is illegal, return 0
     - `r_legal`: 0 if move is legal (`move_error` is None), -1 if illegal (`move_error` present)
   - Load weights from config.yaml
   - Return weighted sum: `r_wr_weight * r_wr + r_move_weight * r_move + r_legal_weight * r_legal`
3. Add helper functions:
   - `extract_root_winrate_and_move(response_str, model_family)` -> (winrate, move) - handles Qwen3, DeepSeekv3.2, Gemma3 thinking formats
   - `normalize_move(move)` -> str - normalize GTP coordinate to lowercase
   - `load_reward_weights(config_path)` -> dict

#### Phase 5: Training Scripts Creation
1. Create `train/offline_env/` directory
2. Copy and modify `train/online_env/train.pbs` to `train/offline_env/train.pbs`:
   - Update job name (e.g., `go_offline`)
   - Update paths to use offline_env scripts
3. Copy and modify `train/online_env/train.sh` to `train/offline_env/train.sh`:
   - Update data paths to use parquet files created by `train/preprocess.py` (e.g., `data/RL/katago_offline/train.parquet`, `test.parquet`)
   - Enable thinking mode in rollout config: `++actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template_kwargs.enable_thinking=True`
   - Detect model family and adjust thinking config if needed (Qwen3, DeepSeekv3.2, Gemma3)
   - Update reward function path to `train/offline_env/reward.py`
   - Configure WandB logging (veRL should handle this automatically if reward function returns structured data)
4. Add preprocessing step documentation: Run `train/preprocess.py` before training to generate parquet files

#### Phase 6: WandB Logging Integration
1. Modify veRL training to log (check if `reward` namespace exists, use `train/reward` if not):
   - `train/reward/r_wr` (individual reward for winrate, per global batch)
   - `train/reward/r_move` (individual reward for move matching, per global batch)
   - `train/reward/r_legal` (individual reward for legality, per global batch)
   - `train/reward/total` (weighted sum, per global batch)
   - `val/reward/r_wr`, `val/reward/r_move`, `val/reward/r_legal`, `val/reward/total` (per val global batch)
2. This may require custom logging hooks or modifying veRL's reward manager
3. Return individual rewards from `compute_score` via `extra_info` or custom return structure if veRL supports it

#### Phase 7: Rollout Logging
1. Add checkpoint callback to save sample rollouts to `{LOG_DIR}/rollout.llm_log.jsonl`
2. Format should match eval pipeline format (`timestamp`, `game_id`, `ply`, `prompt`, `raw_response`, `reasoning`, `parsed_move`, `win_rate`, `error`) PLUS reward fields:
   - `r_wr`, `r_move`, `r_legal`, `reward_total`
3. Save one sample per checkpoint interval

#### Phase 8: Makefile Updates
1. Update `make train` target to use `train/offline_env/train.pbs`
2. Add `make preprocess` target to run `train/preprocess.py`:
   ```makefile
   preprocess: $(VENV_DIR)/bin/activate
       $(ACTIVATE) && python train/preprocess.py
   ```
3. Document that `make preprocess` should be run before `make train`

#### Phase 9: Testing
1. Run pytest to verify data mapping
2. Test reward function with sample inputs
3. Verify WandB logging works correctly
4. Verify rollout logging saves correctly at checkpoints

### Implementation Order
1. Phase 1 (Data validation test) - Foundation
2. Phase 2 (Prompt template) - Required for reward function and preprocessor
3. Phase 3 (Config updates) - Required for preprocessor and reward function
4. Phase 4A (Preprocessor script) - Combine JSONL files into parquet format
5. Phase 4B (Reward function) - Core logic using preprocessed data
6. Phase 5 (Training scripts) - Execution
7. Phase 6 (WandB logging) - Monitoring
8. Phase 7 (Rollout logging) - Debugging/analysis
9. Phase 8 (Makefile) - Convenience (add `make preprocess` target)
10. Phase 9 (Testing) - Validation

### Key Decisions Summary

- **Prompt**: Modify existing `go_move` template to output `{"root_winrate": <float>, "top_move": "<coord>"}`
- **Rewards**: Three-component reward with weights from config.yaml
  - `r_wr`: negative MSE on root winrate ([-1, 0]), -1 if illegal move
  - `r_move`: 1 if matches KataGo top move (normalized lowercase), 0 otherwise
  - `r_legal`: 0 if legal, -1 if illegal (based on KataGo error response)
- **Move Validation**: Use KataGo analysis endpoint error response; invalid format = illegal
- **Data Mapping**: Verify directory structure, file names, row counts, and id matching
- **Data Preprocessing**: Use `train/preprocess.py` to combine position_dir and move_dir JSONL files into parquet format with all fields in `extra_info`
- **WandB**: Log `train/reward/*` and `val/reward/*` metrics per global batch
- **Thinking Mode**: Always enabled, handle Qwen3/DeepSeekv3.2/Gemma3, extract after `</think>` tag
- **Data Split**: Use `train.jsonl` and `test.jsonl` naming convention
- **Makefile**: `make preprocess` → run preprocessor, `make train` → `train/offline_env/train.pbs`

### All Questions Answered ✅

**Q12 - Data Loading Strategy**: ✅ ANSWERED
- Use preprocessor script `train/preprocess.py` to combine JSONL files into parquet format
- Preprocessor reads `position_dir` and `move_dir` JSONL files, matches by `id`, and creates parquet files with all needed fields in `extra_info`
- Reward function reads from `extra_info` (no file I/O during training)

