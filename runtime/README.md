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