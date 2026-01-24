# Scope
host the Katago server as an FastAPI endpoint on my HPC using 1 single H200 GPU
exposing only the analysis endpoint
before that, we need to run 'benchmark' at multiple configurations, in order to identify the optimal configuration for such serving, and then stick to it afterwards

# Env:
- first, we need to build an enroot container based on Katago dependencies. Let's create a PBS script 'runtime/enroot_create.pbs', which will be used to create such image. The existing content inside this file maybe used as a reference.
- then, let's prepare for the deployment with either cuda or TensorRT backend in linux system (HPC configure), compile the executable if needed. 
- download and use the latest strongest model checkpoint
- download at least 5 sample input and their corresponding output for the Katago's analysis end-point, under `assets/analysis/inputs` and `assets/analysis/outputs` respectively
- create a pbs script 'runtime/enroot_start.pbs', which allows use to start the interactive session inside container, wchich runs on a single GPU on compute node 'hopper-34', and use it for dev and testing
- For dev and testing, let's run all commands for testing and debugging inside the tmux session 'go', unless absolutely necessary to do it outside.
- let's then create the fastapi scripts under 'runtime', which runs the katago analysis engine as a FastAPI service, exposing the default port 9100, which should be set as in the runtime/config.yaml as a single source of truth
- let's create a third script 'runtime/serve_analysis.sh', which starts the FastAPI server listening to the port. For user to test run this script, it will use an interative prompt to request the input (in terms of filepath, from `assets/analysis/inputs/<basename>`), and it will save the corresponding outputs to `assets/analysis/outputs/<basename>` by default. But the API endpoint itself will directly take in the input content and return the output content, in similar way as the CLI endpoint

# Alignment (decisions)
- Optimize for **max requests/sec** at **high concurrency** (saturate compute + HBM) on **1x H200**.
- Benchmark **both CUDA and TensorRT** backends; prefer doing all build/runtime via the **enroot container** to avoid HPC permission issues.
- Game config (rules/komi/etc) is **request-specific** (client sets it per request).
- Model: use the **strongest publicly available** KataGo model checkpoint (pin exact filename/URL in runtime docs once chosen).
- API: accept **exact KataGo CLI analysis JSON** input; return **raw KataGo JSON output verbatim**.
- Samples: at least **5** inputs: **1 mid-game**, **1 late-game**, **3 tsumego**, spanning **2 different board sizes**; outputs are **rough-equivalent** (shape/keys stable, not bitwise identical).
- Node/port: dev on compute node **hopper-34** is guaranteed; start with **same-node curl** to `localhost:9100`.

# Concrete implementation steps (2–3)
## Step 1 — Container + dev workflow (enroot + PBS)
- Finalize `runtime/enroot_create.pbs` to build an enroot image containing KataGo build/runtime deps (CUDA + TensorRT capable).
- Finalize `runtime/enroot_start.pbs` to start an interactive job on `hopper-34` with **1 GPU**, drop into the container, and attach to tmux session `go` for all dev/debug.
- Add `runtime/config.yaml` with **port: 9100** as the single source of truth (used by all scripts/services).

## Step 2 — Benchmark CUDA vs TensorRT and lock serving config
- Implement a benchmarking harness (PBS or inside the interactive job) to run `katago benchmark` across:
  - CUDA backend configs (threads/batch/numSearchThreads/etc as applicable)
  - TensorRT backend configs (incl. engine build + runtime settings)
  - High-concurrency load to saturate H200 compute + memory
- Record results (req/s, plus basic latency stats) and pick the best config per backend, then pick the overall winner.
- Freeze the chosen config in `runtime/config.yaml` (and/or a dedicated `runtime/katago_serving.cfg`) and stop varying it for serving.

## Step 3 — FastAPI analysis service + assets + helper script
- Implement FastAPI service under `runtime/` exposing only **analysis** on port from `runtime/config.yaml`.
  - Input: raw CLI analysis JSON payload
  - Output: raw KataGo JSON response (verbatim)
- Add sample inputs under `assets/analysis/inputs/` and their rough-equivalent outputs under `assets/analysis/outputs/`.
- Implement `runtime/serve_analysis.sh`:
  - Starts the FastAPI server
  - Prompts for an input basename from `assets/analysis/inputs/<basename>`
  - Writes response to `assets/analysis/outputs/<basename>` by default
  - Also prints a same-node curl example for quick verification