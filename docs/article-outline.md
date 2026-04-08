# Article Outline — Gemma 4 on Apple Silicon: A Debugging & Benchmarking Journey

**Target**: GDE Medium (EN) + Substack (繁中)
**Author**: Jimmy Liao (GDE Cloud AI)
**Date**: 2026-04-08
**Length**: ~2500-3500 words (EN) / ~3000-4500 字 (繁中)

---

## Hook (opening, ~200 words)

> "I tried to run Gemma 4 E2B on my M1 Air 16GB. The model loaded, the forward pass ran, the GPU spun. And the output was `台灣：台灣：台灣：台灣：`."

Three stories in one article:
1. A silent failure bug that wastes your afternoon
2. Memory management as a first-class M1 engineering concern
3. The Flash Attention myth on Apple Silicon — when it helps, when it hurts

**Reader takeaway**: You can run small LLMs locally on 16GB Apple Silicon, but "running" ≠ "working". Always have a cross-validation path.

---

## Part 1: The Silent Failure

### 1.1 The setup (what I wanted)
- M1 Air 16GB, no other hardware
- Gemma 4 E2B — "Effective 2B" = 2.7B active + 2.4B PLE embedding table
- Target: benchmark MLX vs Ollama for HISP RAG (real work context)
- 50 text + 50 vision QnAs from TaiwanVQA (real Chinese content, not English synthetic)

### 1.2 The symptom
- Converted to MLX 4-bit on Colab (L4) → ran fine there, produced `'A\n'`
- Same file on M1 → `'台灣：台灣：台灣：...'`
- **Cross-backend divergence = first red flag**

### 1.3 The false trails
The hypothesis tree that got burned (write each as "what I thought → what killed it"):
1. **File corruption** → md5 matched, refuted
2. **PLE quantization bug** → wrote `m1_dequantize_ple.py`, PLE → bf16, still broken
3. **Full dequantization** → all layers bf16, still broken
4. **Random init** → gives normal wide logits, not echo → something in the trained weights + mlx-lm forward path interacts badly

Diagnostic tooling built along the way:
- Layer-by-layer hidden state probe
- `previous_kvs` / shared KV cache inspection
- Weight key ↔ model parameter matching (100% match → loading not the bug)
- `partial_rotary_factor` verification

### 1.4 The "aha" moment — it's not us
- Control test: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` on same mlx-lm → produces `"Paris"` correctly
- Gemma 4 specifically broken
- Search GitHub: **Issue #1123** filed 18 hours earlier by @BrendanL79 — exact same symptom, same commit (`dcbf6e3`)
- Second confirmer @kernelpool on X/Twitter

> "When your bug is the top result on GitHub for the very feature you're testing, the story isn't 'I hit a bug'. The story is 'a whole class of users is about to hit this bug quietly'."

### 1.5 Why this is a "silent failure landmine"
- No error message
- Model loads fine (weight keys match 100%)
- Forward pass runs (hidden states evolve normally)
- GPU sits at 30% utilization
- Just produces garbage that *looks like real tokens*
- **The only way to catch it is a smoke test with a known answer**

### 1.6 Lessons
- Smoke test every conversion (before downstream benchmarks)
- Have a control model from a different family
- `cross_backend_smoke_test.sh` — run the same prompt on Ollama + MLX, diff output
- Your conversion pipeline's #1 unit test should be: "can the model answer 'what's the capital of France?'"

---

## Part 2: Memory as a First-Class Concern on 16GB M1

### 2.1 The 16GB unified memory trap
- Wired memory + app memory + cached files + swap all compete for the same 16GB
- Gemma 4 E2B Q4_K_M = 7.16GB wired
- macOS base = ~4GB
- Leaves ~5GB for Python, Chrome, your editor, Claude Code, tmux...
- Once you cross the threshold, **swap thrashes and your tok/s drops 3-5×**

### 2.2 The "you can't just run benchmarks" reality
- First Run 4 attempt: free memory 0.06G, swap 4856MB, pressure HIGH → contaminated results
- After killing next-server + one Claude session: free 7.4G, pressure GREEN → clean results
- **Memory monitoring must be part of your benchmark protocol**

### 2.3 Programmatic memory optimizations
The three levers (with numbers from real runs):
| Lever | Effect | Code |
|---|---|---|
| `num_ctx=2048` | KV cache reservation ↓ 75% | Ollama API `options` |
| `keep_alive=30s` | Model unload between runs | Ollama API field |
| `OLLAMA_KV_CACHE_TYPE=q8_0` + `FLASH_ATTENTION=1` | KV cache bytes ↓ 50% | env var |

**Measured impact**: Run 8 (with all three) peak RSS = **2.57GB** vs Run 7 (only num_ctx + keep_alive) = **3.95GB** → ~35% memory saved.

### 2.4 Memory monitoring script
Include the `scripts/monitor.sh snap | watch | log` pattern — works on both macOS (vm_stat) and Linux (/proc/meminfo). Show the JSONL output.

---

## Part 3: The Flash Attention Myth on M1

### 3.1 Why people enable it
- Ollama docs imply "faster" and "less memory"
- thor3323 on Threads: M4 mini 16GB with FA gets **42 tok/s** on Gemma 4 E2B
- My M1 Air: **25 tok/s** baseline — am I leaving 67% on the table?

### 3.2 The benchmark matrix (2×2×2 = 8 runs, 150 calls each)
Show the full table from `run_matrix_2026-04-08.md`. Key numbers:

**eval tok/s FA delta**:
- nothink text: **-2%** (FA slightly slower on short prompts)
- nothink vision: **-5%**
- think text: **+13%** ← the sweet spot
- think vision: **-12%** (confounded by q8_0 KV cache + memory pressure)

**Wall-clock FA delta**:
- nothink text: **+5%** slower
- nothink vision: **-8%** faster
- think text: **-10%** faster
- think vision: **+10%** slower

### 3.3 Why the inconsistency
- FA optimizes attention memory bandwidth
- **Benefit ∝ KV cache size ∝ sequence length**
- Short sequences: FA overhead > savings → negative return
- Long reasoning: FA savings > overhead → positive return
- M1's Metal FA kernel is younger than M4's → more overhead at small sizes

### 3.4 The M4 vs M1 gap
- thor3323's M4 mini: E2B 42 tok/s, E4B 23 tok/s
- My M1 Air: E2B ~25 tok/s
- **Root cause**: memory bandwidth
  - M1 Air: 68 GB/s
  - M4 mini: 120 GB/s (1.76×)
  - M1 Max: 400 GB/s (5.88×)
- LLM inference is **memory-bound** on Apple Silicon, not compute-bound
- No software trick closes this gap

### 3.5 The practical recommendation
- **Default OFF** on M1 Air — FA gives you -2 to +13% swing depending on workload
- **If you're always doing long thinking/reasoning**: turn it ON, get +13%
- **If you're doing short-answer RAG**: leave it OFF, skip the variability

---

## Part 4: What Actually Works on M1 Today (for Gemma 4)

### 4.1 The matrix of reality (2026-04-08)
| Backend | Gemma 4 E2B | Status |
|---|---|---|
| Ollama (llama.cpp Metal) | ✅ | Works, 25 tok/s baseline |
| mlx-lm 0.31.2 | ❌ | Issue #1123 silent failure |
| vmlx (MLX-based server) | ❌ | Same underlying mlx-lm |
| omlx | ❌ | Same underlying mlx-lm |
| vLLM | ❌ | No Metal backend |
| llama.cpp direct | ✅ | Same as Ollama |

### 4.2 The waiting game
- mlx-lm Issue #1123 is open, no fix in 2 days
- When fixed, **vmlx + speculative decoding** (draft: Qwen2.5-0.5B) is the most promising path to beat Ollama on M1
- Until then, Ollama with `FLASH_ATTENTION=1` for long-output workloads is your best bet

### 4.3 TaiwanVQA specifically
- **Vision accuracy ~18%** on text-only attempts (random guess = 25%)
- Nothing is text-only in practice — every question assumes the image is present
- **Vision accuracy goes up when you actually pass the image**
- Thinking mode doesn't help accuracy on short-answer questions; the 256-token budget runs out during reasoning and the final answer never gets emitted
- **Regex extractor bug**: `\b([ABCD])\b` picks the first letter in reasoning, not the final answer. Post-processing fix: take last-occurring letter.

---

## Conclusion (~300 words)

Three things I wish someone had told me before this experiment:

1. **"It runs" ≠ "It works"**. Cross-validate every model conversion against a known answer on a different backend. The silent failure is always worse than the crash.

2. **Memory is a first-class concern on 16GB M1**. Your benchmark is only as clean as the memory pressure it ran under. Monitor it, free it, report it in every number you publish.

3. **Flash Attention is not universally better on M1**. Measure, don't assume. It's a 15-percent swing in either direction depending on your workload shape.

The side benefit of all this: I have a 637-line investigation doc (`docs/ple-bug-investigation.md`) and a full 8-run benchmark matrix that took 3 hours of real wall-clock time to produce. Both will outlast the current mlx-lm bug.

**Next article teaser**: When Issue #1123 is fixed, I'll re-run the same benchmark with `vmlx` + speculative decoding and see if we can finally beat Ollama's 25 tok/s on M1 Air.

---

## Supporting artifacts

- GitHub repo: `jimmyliao/gemma4-apple-silicon` (public)
- Investigation doc: `docs/ple-bug-investigation.md` (637 lines, step-by-step hypothesis tree)
- Benchmark matrix: `results/run_matrix_2026-04-08.md`
- Raw JSON results: `results/m_*.json` (8 files, 150 calls each)
- Upstream issue: [ml-explore/mlx-lm#1123](https://github.com/ml-explore/mlx-lm/issues/1123)

---

## Visuals needed

1. **Hero screenshot**: Activity Monitor during Run 4 vs Run 8 side by side (memory pressure red → green)
2. **Diagram**: Gemma 4 PLE architecture (PLE + sliding attention + shared KV layers 15-34) — hand-drawn
3. **Chart**: eval_tps bar chart, FA on/off × 4 scenarios (from run_matrix table)
4. **Chart**: wall-clock line showing wall delta across scenarios
5. **Chart**: memory footprint across all 8 runs (peak RSS + swap delta)
6. **Code snippet**: the `monitor.sh snap` one-liner and output
7. **Screenshot**: GitHub Issue #1123 (proof of the shared community pain)

---

## Publishing checklist

- [ ] Draft EN version on GDE Medium
- [ ] 繁中 version on Substack (memo.jimmyliao.net)
- [ ] Cross-link both versions
- [ ] Tag: #Gemma4 #AppleSilicon #LocalLLM #Ollama #MLX #FlashAttention
- [ ] Include full `run_matrix_2026-04-08.md` as a gist or appendix
- [ ] Tweet/Threads teaser on publication day
- [ ] Thanks: @thor3323 (Threads), @BrendanL79 (GitHub), @kernelpool (X)
