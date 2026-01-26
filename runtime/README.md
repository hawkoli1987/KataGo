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
  Defaults: input/output to `runtime/assets/analysis/inputs/lategame_19.jsonl` and
  `runtime/assets/analysis/outputs/lategame_19.jsonl`, host to
  `localhost:<config port>`.

- Sample inputs/outputs:
  Inputs: `runtime/assets/analysis/inputs/*.jsonl`
  Outputs: `runtime/assets/analysis/outputs/*.jsonl`


1. how was this created?@repos/KataGo/runtime/go_engine.o165134, can we move it into the log_dir automatically during job launch?
2.  could you help me create a enroot_start_interactive.shbased on @repos/KataGo/runtime/enroot_start.pbs, so it will only launch a job with the same resource inside the same enroot container, but allow us to run commands interactively inside the container. No need to start the katago analysis engine server
3. you should test the step 2 implementation inside the tmux session 'go'