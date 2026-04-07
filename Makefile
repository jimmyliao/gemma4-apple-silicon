# gemma4-apple-silicon Makefile
# Requires: uv, ollama, gemini CLI
# Usage: make <target>

.PHONY: help install monitor download-data prepare-qna benchmark-nothink benchmark-think benchmark-all benchmark-vision clean

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_NOTHINK  := gemma4-e2b-nothink
OLLAMA_THINK    := gemma4:e2b
MLX_MODEL       := mlx-community/gemma-4-E2B-it-4bit
MONITOR_SCRIPT  := scripts/monitor.sh
RESULTS_DIR     := results

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  gemma4-apple-silicon benchmark"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  install           Install Python deps via uv"
	@echo "  monitor           Live RAM & CPU monitor (run in separate pane)"
	@echo "  monitor-mlx       Monitor mlx_lm process instead"
	@echo ""
	@echo "  download-data     Download hhhuang/TaiwanVQA dataset"
	@echo "  prepare-qna       Use gemini CLI to filter 50 text + 50 vision QnA"
	@echo ""
	@echo "  benchmark-nothink Run Ollama nothink benchmark (50 text + 50 vision)"
	@echo "  benchmark-think   Run Ollama think benchmark (10-probe → 50 full)"
	@echo "  benchmark-mlx     Run MLX server benchmark"
	@echo "  benchmark-all     Run all benchmarks sequentially"
	@echo ""
	@echo "  mem-check         Quick memory snapshot"
	@echo "  ollama-status     Check ollama loaded models"
	@echo "  clean             Remove generated results"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	uv sync

# ── Monitoring (run in a SEPARATE tmux pane) ──────────────────────────────────
monitor:
	@echo "▶ Run this in a separate tmux pane to watch live metrics"
	@bash $(MONITOR_SCRIPT) ollama

monitor-mlx:
	@bash $(MONITOR_SCRIPT) mlx

mem-check:
	@vm_stat | awk '/Pages free/{f=$$3+0}/Pages inactive/{i=$$3+0}/Pages active/{a=$$3+0}/Pages wired/{w=$$4+0}END{\
	ps=16384;gb=1073741824;\
	printf "Free:      %.2f GB\nInactive:  %.2f GB (reclaimable)\nActive:    %.2f GB\nWired:     %.2f GB\nAvailable: %.2f GB\n",\
	f*ps/gb,i*ps/gb,a*ps/gb,w*ps/gb,(f+i)*ps/gb}'
	@echo "Swap:    $$(sysctl -n vm.swapusage | awk '{print $$6}')"

ollama-status:
	@ollama ps

# ── Data Preparation ──────────────────────────────────────────────────────────
download-data:
	@mkdir -p data results
	uv run python data/00_download_dataset.py

prepare-qna:
	@if [ ! -f data/qna_raw_text_500.json ]; then \
		echo "▶ Raw pools missing — running download-data first..."; \
		$(MAKE) download-data; \
	fi
	@echo "▶ Stratified sampling: 50 text + 50 vision (by topic+difficulty)"
	uv run python data/02_stratified_sample.py
	@echo "▶ Fetching images for vision questions..."
	uv run python data/01_fetch_vision_images.py
	@echo "✅ data/qna_text_50.json + data/qna_vision_50.json"
	@echo ""
	@echo "ℹ  Next article TODO: replace stratified sampling with gemini CLI"
	@echo "   to compare quality-based vs random selection."

# ── Benchmarks ────────────────────────────────────────────────────────────────
benchmark-nothink: data/qna_text_50.json
	@mkdir -p $(RESULTS_DIR)
	@echo "▶ Benchmark: Ollama nothink — text-only 50 questions"
	uv run python benchmark/benchmark.py \
		--backend ollama \
		--model $(OLLAMA_NOTHINK) \
		--dataset data/qna_text_50.json \
		--mode text \
		--repeats 3 \
		--output $(RESULTS_DIR)/nothink_text.json
	@echo "▶ Benchmark: Ollama nothink — vision 50 questions"
	uv run python benchmark/benchmark.py \
		--backend ollama \
		--model $(OLLAMA_NOTHINK) \
		--dataset data/qna_vision_50.json \
		--mode vision \
		--repeats 3 \
		--output $(RESULTS_DIR)/nothink_vision.json

benchmark-think: data/qna_text_50.json
	@mkdir -p $(RESULTS_DIR)
	@echo "▶ Memory probe: 10 questions with think model (safety check)"
	uv run python benchmark/benchmark.py \
		--backend ollama \
		--model $(OLLAMA_THINK) \
		--dataset data/qna_text_50.json \
		--mode text \
		--repeats 1 \
		--limit 10 \
		--output $(RESULTS_DIR)/think_probe.json
	@echo "▶ Benchmark: Ollama think — text-only 50 questions"
	uv run python benchmark/benchmark.py \
		--backend ollama \
		--model $(OLLAMA_THINK) \
		--dataset data/qna_text_50.json \
		--mode text \
		--repeats 3 \
		--output $(RESULTS_DIR)/think_text.json
	@echo "▶ Benchmark: Ollama think — vision 50 questions"
	uv run python benchmark/benchmark.py \
		--backend ollama \
		--model $(OLLAMA_THINK) \
		--dataset data/qna_vision_50.json \
		--mode vision \
		--repeats 3 \
		--output $(RESULTS_DIR)/think_vision.json

benchmark-mlx: data/qna_text_50.json
	@mkdir -p $(RESULTS_DIR)
	@echo "▶ Benchmark: MLX server — text-only 50 questions"
	uv run python benchmark/benchmark.py \
		--backend mlx \
		--model $(MLX_MODEL) \
		--dataset data/qna_text_50.json \
		--mode text \
		--repeats 3 \
		--output $(RESULTS_DIR)/mlx_text.json
	@echo "▶ Benchmark: MLX server — vision 50 questions"
	uv run python benchmark/benchmark.py \
		--backend mlx \
		--model $(MLX_MODEL) \
		--dataset data/qna_vision_50.json \
		--mode vision \
		--repeats 3 \
		--output $(RESULTS_DIR)/mlx_vision.json

benchmark-all: benchmark-nothink benchmark-think benchmark-mlx
	@echo "▶ Generating summary report..."
	uv run python benchmark/report.py --results-dir $(RESULTS_DIR)
	@echo "✅ results/summary.json + results/summary.md"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf results/*.json results/*.md
	@echo "Cleaned results/"
