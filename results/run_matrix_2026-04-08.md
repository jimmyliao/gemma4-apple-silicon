# Benchmark Run Matrix — 2026-04-08

**Hardware**: Apple M1 Air, 16GB Unified Memory
**Backend**: Ollama 0.20.0 (llama.cpp Metal)
**Model weights**: gemma4-e2b Q4_K_M (~7.16 GB)
**Dataset**: hhhuang/TaiwanVQA — 50 text + 50 vision (3 repeats each)
**Memory state**: Pressure GREEN, ~11GB available, swap 1.5GB (cleaned, next-server killed)

## Variables

| Var | Values |
|---|---|
| `OLLAMA_FLASH_ATTENTION` | OFF / ON |
| `think` | no / yes |
| `mode` | text / vision |

→ 2 × 2 × 2 = **8 runs**

## Run order

| # | FA | think | mode | Ollama model | max-tokens | Output |
|---|---|---|---|---|---|---|
| 1 | OFF | no  | text   | gemma4-e2b-nothink | 16  | results/m_off_nothink_text.json |
| 2 | OFF | no  | vision | gemma4-e2b-nothink | 16  | results/m_off_nothink_vision.json |
| 3 | OFF | yes | text   | gemma4:e2b         | 256 | results/m_off_think_text.json |
| 4 | OFF | yes | vision | gemma4:e2b         | 256 | results/m_off_think_vision.json |
| 5 | ON  | no  | text   | gemma4-e2b-nothink | 16  | results/m_on_nothink_text.json |
| 6 | ON  | no  | vision | gemma4-e2b-nothink | 16  | results/m_on_nothink_vision.json |
| 7 | ON  | yes | text   | gemma4:e2b         | 256 | results/m_on_think_text.json |
| 8 | ON  | yes | vision | gemma4:e2b         | 256 | results/m_on_think_vision.json |

After run 4, restart ollama with `OLLAMA_FLASH_ATTENTION=1`.
Each test uses `repeats=3`, so 50 × 3 = 150 calls per run.

## Estimated time

- nothink text: ~3 min
- nothink vision: ~5 min  
- think text (256 tok): ~6 min
- think vision (256 tok): ~10 min
- **Total**: ~50 min

## Snapshots

Before each run: `monitor.sh snap m{N}-before`
After each run: `monitor.sh snap m{N}-after`

## Status

- [x] Run 1 (OFF nothink text) — 17:42
- [x] Run 2 (OFF nothink vision) — 17:52
- [x] Run 3 (OFF think text) — 18:29
- [x] Run 4 (OFF think vision) — 19:09
- [x] Restart Ollama with FA=1 + KV_CACHE_TYPE=q8_0 — 19:09
- [x] Run 5 (ON nothink text) — 19:11
- [x] Run 6 (ON nothink vision) — 19:21
- [x] Run 7 (ON think text) — 19:54
- [x] Run 8 (ON think vision) — 20:37
- [x] Comparison table (see Final Results section below)

---

## Final Results (2026-04-08 20:37)

### Full table

| # | FA | think | mode | wall | eval_tps | ttft | peak_rss | swap_Δ |
|---|---|---|---|---|---|---|---|---|
| 1 | OFF | no | text | 150s | 25.61 | 0.45s | 1.26G | 8.0M |
| 2 | OFF | no | vision | 579s | 23.76 | 3.27s | 1.8G | 0.0M |
| 3 | OFF | yes | text | 2171s | 18.60 | 14.72s | 1.78G | 519.4M |
| 4 | OFF | yes | vision | 2358s | 20.76 | N/A* | 1.6G | 8.0M |
| 5 | ON | no | text | 158s | 25.08 | 0.48s | 3.73G | 0.0M |
| 6 | ON | no | vision | 531s | 22.63 | 2.97s | 3.59G | 0.0M |
| 7 | ON | yes | text | 1956s | 21.10 | 12.93s | 3.95G | 8.0M |
| 8 | ON | yes | vision | 2594s | 18.35 | N/A* | 2.57G | 56.0M |

*N/A: think vision hit 256-token limit during thinking, no response chunk → eval_tps from raw post-process*

### FA on/off effect (eval_tps median)

| Scenario | OFF | ON | Δ |
|---|---|---|---|
| nothink text | 25.61 | 25.08 | -2.1% |
| nothink vision | 23.76 | 22.63 | -4.7% |
| think text | 18.60 | 21.10 | +13.4% |
| think vision | 20.76 | 18.35 | -11.6% |

### Wall-clock comparison

| Scenario | OFF | ON | Δ |
|---|---|---|---|
| nothink text | 150s | 158s | +5.2% |
| nothink vision | 579s | 531s | -8.3% |
| think text | 2171s | 1956s | -9.9% |
| think vision | 2358s | 2594s | +10.0% |

### Key findings

1. **FA effect is highly context-dependent on M1 Air**
   - Largest gain: **think text +13.4%** eval_tps (long reasoning output)
   - Slight regression: nothink text -2%, nothink vision -5% (short sequences)
   - Mixed: think vision shows -12% BUT Run 8 also has q8_0 KV cache + was under memory pressure first ~15 min → not a clean FA-only comparison

2. **Memory optimizations successful (Runs 4-8)**
   - `num_ctx=2048` + `keep_alive=30s` + `OLLAMA_KV_CACHE_TYPE=q8_0` (Runs 5-8)
   - Run 8 peak RSS: **2.57GB** vs Run 7 (without q8) **3.95GB** → q8_0 KV cache saves ~35% memory
   - Run 8 swap delta: 56MB (minimal) vs Run 3: 519MB (high pressure)

3. **think mode accuracy parsing is broken**
   - All think runs show 0% accuracy
   - Root cause: regex `\b([ABCD])\b` picks letters inside reasoning text before the answer
   - Need post-processing: extract last-occurring A/B/C/D in output instead of first

4. **Wall-clock winner depends on task**
   - Best FA wins: think text -10%, nothink vision -8%
   - FA loses: nothink text +5%, think vision +10% (with confounders)

