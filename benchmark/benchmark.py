#!/usr/bin/env python3
"""
benchmark.py — Ollama / MLX server benchmark for Gemma4 on Apple Silicon

Measures: TTFT, throughput (tok/s), total latency, accuracy (A/B/C/D match),
          system memory + ollama RSS during inference.

Usage:
  uv run python benchmark/benchmark.py \\
      --backend ollama \\
      --model gemma4-e2b-nothink \\
      --dataset data/qna_text_50.json \\
      --mode text \\
      --repeats 3 \\
      --output results/nothink_text.json
"""
import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Backend clients ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MLX_URL = "http://localhost:8000/v1/chat/completions"


def call_ollama_stream(model: str, prompt: str, image_b64: str | None = None,
                       max_tokens: int = 16, think: bool = True,
                       num_ctx: int = 2048, keep_alive: str = "5m") -> dict:
    """Call Ollama streaming API. Returns dict with timing + output.

    think=False forces non-thinking mode (Ollama 0.5+ /api/generate `think` field).
    num_ctx: context window size — smaller = less KV cache memory pressure.
    keep_alive: model unload timeout. "5m" default, "0" unload immediately.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": think,
        "keep_alive": keep_alive,
        "options": {"num_predict": max_tokens, "temperature": 0.0, "num_ctx": num_ctx},
    }
    if image_b64:
        payload["images"] = [image_b64]

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    ttft = None  # first response token
    ttf_thinking = None  # first thinking token (if any)
    output_text = ""
    thinking_text = ""
    eval_count = 0
    prompt_eval_count = 0
    eval_duration_ns = 0
    prompt_eval_duration_ns = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode())
                # Capture thinking tokens (Ollama 0.5+ thinking mode)
                if chunk.get("thinking"):
                    if ttf_thinking is None:
                        ttf_thinking = time.perf_counter() - t_start
                    thinking_text += chunk["thinking"]
                # Capture response tokens
                if chunk.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    output_text += chunk["response"]
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                    break
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e), "ttft_sec": None, "total_sec": None}

    total_sec = time.perf_counter() - t_start
    return {
        "output": output_text.strip(),
        "thinking": thinking_text.strip() if thinking_text else None,
        "ttft_sec": ttft,
        "ttf_thinking_sec": ttf_thinking,
        "total_sec": total_sec,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "eval_tps": (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else None,
        "wall_tps": (eval_count / total_sec) if total_sec else None,
    }


def call_mlx_stream(model: str, prompt: str, image_b64: str | None = None,
                    max_tokens: int = 16) -> dict:
    """Call MLX server (omlx/vmlx) — OpenAI compatible /v1/chat/completions."""
    if image_b64:
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        MLX_URL, data=data, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    ttft = None
    output_text = ""
    eval_count = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except Exception:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    output_text += delta
                    eval_count += 1
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e), "ttft": None, "total_sec": None}

    total_sec = time.perf_counter() - t_start
    return {
        "output": output_text.strip(),
        "ttft_sec": ttft,
        "total_sec": total_sec,
        "eval_count": eval_count,
        "prompt_eval_count": None,
        "eval_tps": None,
        "wall_tps": (eval_count / total_sec) if total_sec else None,
    }


# ── System monitor (background thread) ────────────────────────────────────────

PAGE_SIZE = 16384
GB = 1024 ** 3


def get_mem_snapshot() -> dict:
    out = subprocess.check_output(["vm_stat"]).decode()
    free = inact = active = wired = 0
    for line in out.splitlines():
        if "Pages free" in line:
            free = int(line.split(":")[1].strip().rstrip("."))
        elif "Pages inactive" in line:
            inact = int(line.split(":")[1].strip().rstrip("."))
        elif "Pages active" in line:
            active = int(line.split(":")[1].strip().rstrip("."))
        elif "Pages wired" in line:
            wired = int(line.split(":")[1].strip().rstrip("."))
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
    swap_mb = float(swap.split()[5].rstrip("M"))
    # ollama RSS
    ollama_rss_gb = 0.0
    ollama_cpu = 0.0
    try:
        ps = subprocess.check_output(
            ["ps", "-axo", "pid,pcpu,rss,comm"], text=True
        )
        for line in ps.splitlines():
            if "ollama" in line.lower() and "grep" not in line:
                parts = line.split()
                ollama_cpu += float(parts[1])
                ollama_rss_gb += float(parts[2]) / 1024 / 1024
    except Exception:
        pass
    return {
        "free_gb": round(free * PAGE_SIZE / GB, 2),
        "inactive_gb": round(inact * PAGE_SIZE / GB, 2),
        "active_gb": round(active * PAGE_SIZE / GB, 2),
        "wired_gb": round(wired * PAGE_SIZE / GB, 2),
        "swap_mb": round(swap_mb, 1),
        "ollama_cpu": round(ollama_cpu, 1),
        "ollama_rss_gb": round(ollama_rss_gb, 2),
    }


class SystemMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                snap = get_mem_snapshot()
                snap["ts"] = time.time()
                self.samples.append(snap)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        free = [s["free_gb"] for s in self.samples]
        rss = [s["ollama_rss_gb"] for s in self.samples]
        cpu = [s["ollama_cpu"] for s in self.samples]
        swap = [s["swap_mb"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "free_gb_min": min(free),
            "free_gb_max": max(free),
            "ollama_rss_gb_max": max(rss),
            "ollama_cpu_max": max(cpu),
            "ollama_cpu_avg": round(sum(cpu) / len(cpu), 1),
            "swap_mb_min": min(swap),
            "swap_mb_max": max(swap),
            "swap_mb_delta": round(max(swap) - min(swap), 1),
        }


# ── Accuracy ──────────────────────────────────────────────────────────────────

ANS_RE = re.compile(r"\b([ABCD])\b")


def extract_answer(output: str) -> str | None:
    """Extract first A/B/C/D from model output."""
    if not output:
        return None
    m = ANS_RE.search(output.upper())
    return m.group(1) if m else None


# ── Main benchmark loop ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "mlx"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--mode", choices=["text", "vision"], default="text")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of questions (for probing)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--think", action="store_true",
                    help="Enable thinking mode (Ollama 0.5+). Default: off (faster).")
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="Ollama context window. Smaller = less KV cache memory. Default: 2048.")
    ap.add_argument("--keep-alive", default="5m",
                    help="Ollama model unload timeout. Default: 5m.")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    questions = json.loads(args.dataset.read_text())
    if args.limit:
        questions = questions[: args.limit]
    print(f"▶ Backend: {args.backend} | Model: {args.model} | Mode: {args.mode}")
    print(f"▶ Dataset: {args.dataset.name} ({len(questions)} questions × {args.repeats} repeats)")

    if args.backend == "ollama":
        def call_fn(model, prompt, img, max_tokens):
            return call_ollama_stream(model, prompt, img, max_tokens, think=args.think,
                                      num_ctx=args.num_ctx, keep_alive=args.keep_alive)
    else:
        def call_fn(model, prompt, img, max_tokens):
            return call_mlx_stream(model, prompt, img, max_tokens)

    # Pre-snapshot before warm-up
    pre = get_mem_snapshot()
    print(f"▶ Pre-warmup mem: free={pre['free_gb']}G ollama={pre['ollama_rss_gb']}G")

    # Warm-up: load model into memory once
    print("▶ Warm-up call (load model)...")
    warm = call_fn(args.model, "Hello", None, max_tokens=4)
    print(f"  warm-up TTFT={warm.get('ttft_sec', 'n/a')} total={warm.get('total_sec', 'n/a')}")

    post_warm = get_mem_snapshot()
    print(f"▶ Post-warmup mem: free={post_warm['free_gb']}G ollama={post_warm['ollama_rss_gb']}G")

    # Start background monitor
    monitor = SystemMonitor(interval=1.0)
    monitor.start()

    # Run benchmark
    results = []
    correct = 0
    n_total = 0
    t_bench_start = time.perf_counter()

    for repeat in range(args.repeats):
        print(f"\n=== Repeat {repeat + 1}/{args.repeats} ===")
        for i, q in enumerate(questions):
            img = q.get("image_base64") if args.mode == "vision" else None
            r = call_fn(args.model, q["prompt"], img, max_tokens=args.max_tokens)
            r["question_id"] = q.get("question_id", "")
            r["expected"] = q.get("answer", "")
            r["predicted"] = extract_answer(r.get("output", ""))
            r["correct"] = (r["predicted"] == r["expected"])
            r["repeat"] = repeat
            results.append(r)
            n_total += 1
            if r["correct"]:
                correct += 1
            if (i + 1) % 10 == 0:
                acc = correct / n_total * 100
                tps = r.get("wall_tps") or 0
                print(f"  [{i+1}/{len(questions)}] tok/s≈{tps:.1f} acc={acc:.1f}%")

    t_bench_total = time.perf_counter() - t_bench_start
    monitor.stop()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    valid = [r for r in results if r.get("ttft_sec") is not None]
    ttfts = [r["ttft_sec"] for r in valid]
    totals = [r["total_sec"] for r in valid]
    wall_tps = [r["wall_tps"] for r in valid if r.get("wall_tps")]
    eval_tps = [r["eval_tps"] for r in valid if r.get("eval_tps")]

    def stats(arr):
        if not arr:
            return None
        return {
            "n": len(arr),
            "mean": round(statistics.mean(arr), 4),
            "median": round(statistics.median(arr), 4),
            "p95": round(sorted(arr)[int(len(arr) * 0.95)], 4) if len(arr) >= 20 else None,
            "min": round(min(arr), 4),
            "max": round(max(arr), 4),
        }

    summary = {
        "backend": args.backend,
        "model": args.model,
        "mode": args.mode,
        "dataset": args.dataset.name,
        "n_questions": len(questions),
        "repeats": args.repeats,
        "n_calls": len(results),
        "n_valid": len(valid),
        "wall_clock_sec": round(t_bench_total, 2),
        "accuracy_pct": round(correct / n_total * 100, 2) if n_total else 0,
        "ttft_sec": stats(ttfts),
        "total_sec": stats(totals),
        "wall_tps": stats(wall_tps),
        "eval_tps": stats(eval_tps),
        "system": monitor.summary(),
        "pre_mem": pre,
        "post_warmup_mem": post_warm,
        "post_bench_mem": get_mem_snapshot(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "summary": summary,
        "results": results,
    }, ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"✅ Saved → {args.output}")
    print(f"   Accuracy:  {summary['accuracy_pct']}%")
    print(f"   TTFT med:  {summary['ttft_sec']['median'] if summary['ttft_sec'] else 'n/a'}s")
    print(f"   Wall tok/s med: {summary['wall_tps']['median'] if summary['wall_tps'] else 'n/a'}")
    print(f"   Eval tok/s med: {summary['eval_tps']['median'] if summary['eval_tps'] else 'n/a'}")
    print(f"   Free mem min:   {summary['system']['free_gb_min']}G")
    print(f"   Ollama RSS max: {summary['system']['ollama_rss_gb_max']}G")
    print(f"   Swap delta:     {summary['system']['swap_mb_delta']}MB")


if __name__ == "__main__":
    main()
