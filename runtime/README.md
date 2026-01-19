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
  Prereqs: network access, `curl`, `unzip`. Runs on host or inside container.

- List latest KataGo release assets (CUDA/TRT):
  `python3 runtime/get_katago_release_assets.py`
  Prereqs: network access. Optional args: `--repo`, `--include`, `--kinds`.

- Scrape model URLs from katagotraining.org:
  `python3 runtime/get_katago_model_urls.py`
  Prereqs: network access. Optional args: `--base-url`, `--limit`.

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
- Start the FastAPI analysis server:
  `uvicorn runtime.analysis_server:app --host 0.0.0.0 --port 9100`
  Prereqs: run inside the container via `runtime/enroot_start.pbs`.
  Install deps in-container: `pip install fastapi uvicorn pyyaml`.

- Run the helper script (starts server, prompts for input, writes output):
  `bash runtime/serve_analysis.sh`
  Prereqs: same as above. Inputs live in `runtime/assets/analysis/inputs`.

- Sample inputs/outputs:
  Inputs: `runtime/assets/analysis/inputs/*.jsonl`
  Outputs: `runtime/assets/analysis/outputs/*.jsonl`