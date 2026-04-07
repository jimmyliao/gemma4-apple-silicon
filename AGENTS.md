# gemma4-apple-silicon — AGENTS.md

**Purpose**: Benchmark Gemma4 E2B (5.1B params) on Apple Silicon and Edge devices.
**Audience**: AI assistants (Claude Code / Gemini CLI / Codex) working on this repo.
**Repo language**: Code EN, docs bilingual (繁中 + EN).

---

## Project context

This repo accompanies a GDE blog article comparing local LLM inference engines:
- **Phase 1 (current)**: M1 Air 16GB — Ollama (llama.cpp/Metal) vs MLX server (omlx/vmlx)
- **Phase 2 (next)**: Jetson Orin Nano 8GB (`cf-jetson`) — Ollama vs vLLM (CUDA)

Dataset: `hhhuang/TaiwanVQA` (CC-BY-SA-4.0) — 繁中 visual QA, 8000 test rows
Use: 50 text-only (speed test) + 50 vision (speed + accuracy)

Model: `gemma4:e2b` / `gemma4-e2b-nothink` — actually 5.1B params, Q4_K_M, ~7.3GB unified memory

---

## Directory layout

```
gemma4-apple-silicon/
├── Makefile              # All targets — always use `make` not raw scripts
├── pyproject.toml        # uv-managed, python >=3.11
├── data/
│   ├── 00_download_dataset.py      # HF streaming (memory-safe, <500MB RAM)
│   ├── 01_fetch_vision_images.py   # Fetch base64 for selected 50 vision qs
│   ├── 02_stratified_sample.py     # Stratify by topic+difficulty, pick 50+50
│   ├── qna_raw_text_500.json       # Pool (no image bytes)
│   ├── qna_raw_vision_meta_500.json# Pool (metadata only)
│   ├── qna_text_50.json            # Final text benchmark set
│   └── qna_vision_50.json          # Final vision benchmark set (base64 images)
├── benchmark/
│   └── benchmark.py      # CLI: --backend {ollama,mlx,vllm} --mode {text,vision}
├── scripts/
│   └── monitor.sh        # watch | snap | log (RAM/CPU/swap)
└── results/
    ├── {backend}_{mode}.json       # Per-run structured results
    └── snapshots.jsonl             # Manual snapshots (baseline/during/after)
```

---

## Critical rules for AI assistants

1. **ALWAYS use `uv run python`** — never bare `python3`, never activate venv manually
2. **ALWAYS use `make` targets** — see `make help` for list
3. **NEVER materialize the full dataset into memory** — use `streaming=True`
   - Past incident: naive `list(ds["test"])` consumed 38 GB virtual memory, killed M1
4. **Memory budget on M1 16GB Air**: gemma4 E2B uses ~7.3 GB wired (Metal); leaves ~8 GB for system + Python
5. **Ollama quirks**:
   - `gemma4:e2b` defaults to thinking mode — must pass `"think": false` in API payload for speed test
   - Model name `gemma4-e2b-nothink` is misleading: same weights, same thinking; only the `think: false` API flag turns it off
   - TTFT is measured from first non-empty `response` chunk in stream (ignore `thinking` chunks)
6. **TaiwanVQA insight**: `is_ocr=False` does NOT mean "text-only answerable". All questions are paired with images; text mode is purely a speed test — expect low accuracy (~10-15%)

---

## Common workflows

### Local M1 benchmark (Phase 1)

```bash
# 1. Environment + data
make install              # uv sync
make download-data        # HF streaming → raw 500+500 pools
make prepare-qna          # stratified sample → 50+50, fetch vision images

# 2. Live monitoring (run in a separate tmux pane)
make monitor              # watch ollama RAM/CPU/swap

# 3. Snapshots around benchmark (manual)
bash scripts/monitor.sh snap baseline
bash scripts/monitor.sh snap before-nothink
# ... run benchmark ...
bash scripts/monitor.sh snap after-nothink

# 4. Benchmarks
make benchmark-nothink    # Ollama, think=false, text + vision
make benchmark-think      # Ollama, think=true, with 10-question safety probe first
make benchmark-mlx        # omlx or vmlx (requires MLX server running on :8000)
```

### CLI reference (benchmark.py)

```bash
uv run python benchmark/benchmark.py \
    --backend ollama \              # ollama | mlx | vllm
    --model gemma4:e2b \
    --dataset data/qna_text_50.json \
    --mode text \                   # text | vision
    --repeats 3 \
    --think \                       # optional: enable thinking mode
    --max-tokens 16 \               # short answer, just A/B/C/D
    --limit 5 \                     # optional: smoke test
    --output results/run_X.json
```

---

## Phase 2: Jetson Orin Nano 8GB (cf-jetson)

### Hardware constraints
- **Unified memory**: 8 GB shared CPU+GPU (vs M1's 16 GB)
- **Compute**: Ampere GPU, 1024 CUDA cores, 67 TOPS (int8)
- **JetPack**: 6.x (aarch64, CUDA 12.x, Ubuntu 22.04)
- **Access**: Cloudflare Tunnel (SSH via `ssh cf-jetson`)

### Memory budget (target: <7 GB used, leave 1 GB headroom)
```
Component                       Size
OS + services                   ~1.8 GB
Ollama/vLLM server process      ~0.3 GB
gemma4:e2b Q4_K_M (MUST quantize)~2.7 GB   ← NOT the F16 7.3 GB M1 version
bge-m3 (if concurrent)          ~1.2 GB
Inference KV cache              ~0.5 GB
─────────────────────────────────────
Total                           ~6.5 GB   (OK with margin)
```

### Backend strategy

| Backend | Status on Jetson Orin | Priority |
|---------|----------------------|----------|
| Ollama  | ✅ Official ARM64 build, CUDA auto-detect | **primary** |
| vLLM    | ⚠️ No JetPack wheel; needs community build | secondary |

**vLLM on Jetson requires**:
- `jetson-containers` project (Dusty NV) — prebuilt containers for JetPack
- Or build from source with `VLLM_TARGET_DEVICE=cuda` + JetPack CUDA toolkit
- Estimated 2-4 hour first build; subsequent runs use the container

### Phase 2 workflow

```bash
# On M1 → prepare Jetson repo (same code, same dataset)
ssh cf-jetson "mkdir -p ~/workspace/lab"
rsync -av --exclude '.venv' --exclude 'results' \
    ~/workspace/lab/gemma4-apple-silicon/ \
    cf-jetson:~/workspace/lab/gemma4-apple-silicon/

# On cf-jetson → setup
ssh cf-jetson
cd ~/workspace/lab/gemma4-apple-silicon
# (install uv, ollama)
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh

# Pull Q4 model (NOT F16 — watch the size)
ollama pull gemma4:e2b    # confirm Q4_K_M from `ollama show`
# If F16, force quantized variant from HF GGUF

# Run benchmarks
make install
make prepare-qna          # dataset is small, will re-download on Jetson
make monitor              # Linux version of monitor (to implement)
make benchmark-nothink    # Ollama
# make benchmark-vllm     # if vLLM container ready
```

### Monitor on Linux (TODO)

`scripts/monitor.sh` currently uses macOS-specific `vm_stat` / `sysctl vm.swapusage`.
For Jetson/Linux, add `scripts/monitor-linux.sh` using:
- `/proc/meminfo` for RAM
- `tegrastats` for GPU/unified memory
- `free -h` for swap
- `ps aux` for process RSS (same as mac)

Makefile should auto-select based on `uname`:
```make
MONITOR_SCRIPT := scripts/monitor-$(shell uname -s | tr '[:upper:]' '[:lower:]').sh
```

### Benchmark comparison matrix (final deliverable)

| Device | Backend | Model | Mode | TTFT | tok/s | Mem | Accuracy |
|--------|---------|-------|------|------|-------|-----|----------|
| M1 16G | Ollama  | gemma4 E2B Q4 | text   | TBD | TBD | TBD | TBD |
| M1 16G | Ollama  | gemma4 E2B Q4 | vision | TBD | TBD | TBD | TBD |
| M1 16G | omlx    | gemma4 E2B 4bit | text   | TBD | TBD | TBD | TBD |
| M1 16G | vmlx    | gemma4 E2B 4bit | text   | TBD | TBD | TBD | TBD |
| Jetson 8G | Ollama | gemma4 E2B Q4 | text   | TBD | TBD | TBD | TBD |
| Jetson 8G | Ollama | gemma4 E2B Q4 | vision | TBD | TBD | TBD | TBD |
| Jetson 8G | vLLM  | gemma4 E2B 4bit | text   | TBD | TBD | TBD | TBD |

---

## Article output (bilingual)

Target publishing venues:
- **EN**: Medium (GDE Cloud AI tag)
- **繁中**: Substack / personal blog

Key narrative points:
1. Why run LLMs locally on Apple Silicon / Edge
2. Gemma4 E2B = sweet spot (5.1B params, <8 GB quantized)
3. Engine comparison (llama.cpp Metal vs MLX native)
4. Memory pressure monitoring as a practice (show `monitor.sh snap`)
5. TaiwanVQA as a bilingual benchmark — insight: model correctly refuses without image
6. Jetson Orin: can 8 GB really host a 5B model + embeddings? (Phase 2)
7. Next article teaser: LLM-based quality filtering (replace stratified sampling with gemini CLI)

---

## Known gotchas

1. **HuggingFace datasets + image columns** — NEVER `list(ds)` on VQA datasets, use `streaming=True`
2. **Ollama thinking mode** — `gemma4:e2b` defaults to think; use `"think": false` API flag
3. **TaiwanVQA text-only filter is misleading** — all questions assume images present
4. **Memory on M1 16GB** — need to stop OrbStack to have >4GB free for inference
5. **Vision mode is ~3× slower than text** — tok/s drops from ~25 to ~4 (image encode overhead)
6. **Accuracy extraction**: use regex `\b([ABCD])\b` on uppercased output (model may answer with "**A**" or "答案是 A")

---

## Repo hygiene

- `results/*.json` — commit these (they are the article's data)
- `data/qna_*.json` — commit raw pools + final 50+50; they are small and reproducible
- `data/qna_vision_50.json` — **59 MB with base64 images**, consider Git LFS before going public
- `.venv/` — never commit
- `__pycache__/` — never commit
- Add `.gitignore` before first public push

---

*Last updated: 2026-04-07*
*Maintainer: @jimmyliao (Agent-Eva / Claude Code)*
