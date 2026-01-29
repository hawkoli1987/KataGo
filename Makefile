SHELL := /bin/bash
PYTHON := python3
VENV_DIR := .venv
ACTIVATE := source $(VENV_DIR)/bin/activate
CONFIG_FILE := configs/config.yaml
MANAGE_SCRIPT := runtime/manage_servers.sh

# Read configuration from config.yaml
SGF_DIR ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['data']['sgf_dir'])" 2>/dev/null || echo "data/raw-sgfs")
POSITION_DIR ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['data']['position_dir'])" 2>/dev/null || echo "data/positions")
MOVE_DIR ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['data']['move_dir'])" 2>/dev/null || echo "data/moves")
ANALYSIS_LIMIT ?= $(shell $(PYTHON) -c "import yaml; config=yaml.safe_load(open('$(CONFIG_FILE)')); limit=config.get('data', {}).get('limit'); print(limit) if limit else print('')" 2>/dev/null || echo "")

# Server configuration
CANDIDATE_PORT ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['servers']['candidate']['port'])" 2>/dev/null || echo "9200")
REFERENCE_PORT ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['servers']['reference']['port'])" 2>/dev/null || echo "9201")
CANDIDATE_MODEL ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['servers']['candidate']['model_path'])" 2>/dev/null || echo "")
REFERENCE_MODEL ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['servers']['reference']['model_path'])" 2>/dev/null || echo "")
OPENAI_API_BASE ?= $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(CONFIG_FILE)'))['servers']['openai']['primary'])" 2>/dev/null || echo "http://localhost:8002/v1")

.PHONY: install activate test-eval run run_all run_katago_engine stop status extract analyze_offline analyze_quick preprocess train clean help

# Test results directory
TEST_RESULTS_DIR := data/eval/_test_runs

# =============================================================================
# Installation
# =============================================================================

install: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: pyproject.toml
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	touch $(VENV_DIR)/bin/activate

activate:
	@echo "Run this command to activate the virtual environment:"
	@echo "  source $(VENV_DIR)/bin/activate"

# =============================================================================
# Server Management
# =============================================================================

# Start both candidate and reference KataGo servers
run_all: $(VENV_DIR)/bin/activate
	@echo "Starting candidate and reference KataGo servers..."
	@chmod +x $(MANAGE_SCRIPT)
	$(MANAGE_SCRIPT) start all
	@echo ""
	@echo "Waiting for servers to be ready..."
	$(MANAGE_SCRIPT) wait all 180
	@echo ""
	@echo "Server status:"
	$(MANAGE_SCRIPT) status

# Start single KataGo engine (candidate only) - legacy target
run_katago_engine: $(VENV_DIR)/bin/activate
	@echo "Starting candidate KataGo server..."
	@chmod +x $(MANAGE_SCRIPT)
	$(MANAGE_SCRIPT) start candidate
	@echo ""
	@echo "Waiting for server to be ready..."
	$(MANAGE_SCRIPT) wait candidate 120
	@echo ""
	$(MANAGE_SCRIPT) status

# Stop all servers
stop:
	@echo "Stopping all KataGo servers..."
	@chmod +x $(MANAGE_SCRIPT)
	$(MANAGE_SCRIPT) stop all

# Show server status
status:
	@chmod +x $(MANAGE_SCRIPT)
	$(MANAGE_SCRIPT) status

# =============================================================================
# Testing
# =============================================================================

# Run all evaluation tests (promotion tests for KataGo and OpenAI)
# Requires: KataGo servers running (via `make run_all`)
test-eval: $(VENV_DIR)/bin/activate
	@echo "Running evaluation tests (promotion tests)..."
	@echo "Note: Ensure KataGo servers are running with 'make run_all'"
	@echo ""
	@mkdir -p $(TEST_RESULTS_DIR)
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	RESULT_FILE="$(TEST_RESULTS_DIR)/test_eval_$$TIMESTAMP.xml"; \
	LOG_FILE="$(TEST_RESULTS_DIR)/test_eval_$$TIMESTAMP.log"; \
	echo "Results will be saved to: $$RESULT_FILE"; \
	$(ACTIVATE) && \
		OPENAI_API_BASE=$(OPENAI_API_BASE) \
		pytest tests/test_eval.py -v --junitxml="$$RESULT_FILE" 2>&1 | tee "$$LOG_FILE"; \
	echo ""; \
	echo "Test results saved to: $$RESULT_FILE"; \
	echo "Test log saved to: $$LOG_FILE"; \
	echo "Evaluation results saved to: data/eval/_test_runs/*/results.json"

# =============================================================================
# Data Processing
# =============================================================================

# Extract SGF data to JSONL positions
extract: $(VENV_DIR)/bin/activate
	@if [ ! -d "$(SGF_DIR)" ]; then \
		echo "Error: SGF directory does not exist: $(SGF_DIR)"; \
		echo "Update sgf_dir in $(CONFIG_FILE) or set SGF_DIR=<path>"; \
		exit 1; \
	fi
	$(ACTIVATE) && python runtime/extract_positions.py \
		--input-dir "$(SGF_DIR)" \
		--output-dir "$(POSITION_DIR)"

# Analyze positions and save model responses
# Requires: KataGo server running (via `make run_katago_engine` or `make run_all`)
analyze_offline: $(VENV_DIR)/bin/activate
	@if [ ! -d "$(POSITION_DIR)" ]; then \
		echo "Error: Position directory does not exist: $(POSITION_DIR)"; \
		echo "Run 'make extract' first to generate position files"; \
		exit 1; \
	fi
	@chmod +x $(MANAGE_SCRIPT)
	@CANDIDATE_ENDPOINT=$$($(MANAGE_SCRIPT) endpoints 2>/dev/null | grep "CANDIDATE_ENDPOINT=" | cut -d= -f2); \
	if [ -z "$$CANDIDATE_ENDPOINT" ]; then \
		echo "Error: Cannot find candidate server endpoint."; \
		echo "Ensure KataGo server is running with 'make run_all' or 'make run_katago_engine'"; \
		exit 1; \
	fi; \
	ANALYSIS_ENDPOINT="$${CANDIDATE_ENDPOINT}/analysis"; \
	echo "Analyzing positions from $(POSITION_DIR)"; \
	echo "Output will be saved to $(MOVE_DIR)"; \
	echo "Using endpoint: $${ANALYSIS_ENDPOINT}"; \
	if [ -n "$(ANALYSIS_LIMIT)" ]; then \
		echo "Limit: $(ANALYSIS_LIMIT) positions per file"; \
	fi; \
	echo ""; \
	mkdir -p "$(MOVE_DIR)"; \
	find "$(POSITION_DIR)" -name "*.jsonl" | while read input_file; do \
		rel_path=$${input_file#$(POSITION_DIR)/}; \
		output_file="$(MOVE_DIR)/$${rel_path}"; \
		output_dir=$$(dirname "$$output_file"); \
		mkdir -p "$$output_dir"; \
		echo "Processing: $$rel_path"; \
		if [ -n "$(ANALYSIS_LIMIT)" ]; then \
			$(ACTIVATE) && python runtime/analyze_offline.py \
				--input "$$input_file" \
				--output "$$output_file" \
				--endpoint "$${ANALYSIS_ENDPOINT}" \
				--limit $(ANALYSIS_LIMIT); \
		else \
			$(ACTIVATE) && python runtime/analyze_offline.py \
				--input "$$input_file" \
				--output "$$output_file" \
				--endpoint "$${ANALYSIS_ENDPOINT}"; \
		fi; \
	done; \
	echo ""; \
	echo "Analysis complete. Results in $(MOVE_DIR)"


# =============================================================================
# Training
# =============================================================================

# Preprocess position and move JSONL files into parquet format for veRL training
# Uses limit from config.yaml data.limit (default: 1000 for testing)
preprocess: $(VENV_DIR)/bin/activate
	@echo "Preprocessing training data..."
	@PREPROCESS_LIMIT=$$($(PYTHON) -c "import yaml; config=yaml.safe_load(open('$(CONFIG_FILE)')); limit=config.get('data', {}).get('limit'); print(limit) if limit else print('')" 2>/dev/null || echo ""); \
	if [ -z "$$PREPROCESS_LIMIT" ]; then \
		echo "Warning: No limit found in config.yaml, processing all data"; \
		$(ACTIVATE) && python train/preprocess.py; \
	else \
		echo "Using limit from config.yaml: $$PREPROCESS_LIMIT rows per split"; \
		$(ACTIVATE) && python train/preprocess.py --limit $$PREPROCESS_LIMIT; \
	fi
	@echo ""
	@echo "Preprocessing complete. Parquet files ready for training."

train:
	@echo "Submitting training job to PBS queue..."
	qsub train/online_env/train.pbs
	@echo "Job submitted. Check queue status with 'qstat -u $$USER'"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
