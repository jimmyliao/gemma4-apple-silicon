#!/usr/bin/env python3
"""
Stratified random sampling: pick 50 text + 50 vision QnA from 500 candidates.

Strategy: stratify by (topic, difficulty) to ensure diversity.
Falls back to pure random if strata are too small.

Outputs:
  data/qna_text_50.json       — 50 text-only questions (no image)
  data/qna_vision_50_ids.json — 50 vision question_ids (image fetched in step 03)

Run: uv run python data/02_stratified_sample.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
random.seed(42)

TARGET_PER_GROUP = 50


def stratified_sample(items, target, key_fn):
    """Sample `target` items, stratified by `key_fn`. Even distribution per stratum."""
    strata = defaultdict(list)
    for it in items:
        strata[key_fn(it)].append(it)

    n_strata = len(strata)
    base_per_stratum = max(1, target // n_strata)
    selected = []

    # Round 1: take base_per_stratum from each
    for k, lst in strata.items():
        random.shuffle(lst)
        take = min(base_per_stratum, len(lst))
        selected.extend(lst[:take])

    # Round 2: fill remaining quota by random sampling from leftovers
    if len(selected) < target:
        leftover = []
        used_ids = {id(x) for x in selected}
        for k, lst in strata.items():
            for it in lst:
                if id(it) not in used_ids:
                    leftover.append(it)
        random.shuffle(leftover)
        selected.extend(leftover[: target - len(selected)])

    # If still over (small target/big strata), trim
    return selected[:target]


def summarize(label, items):
    print(f"\n=== {label} ===")
    print(f"Count: {len(items)}")
    topics = defaultdict(int)
    diffs = defaultdict(int)
    for it in items:
        topics[it.get("topic") or "unknown"] += 1
        diffs[str(it.get("difficulty", "0") or "0")] += 1
    print("Topics:")
    for t, c in sorted(topics.items(), key=lambda x: -x[1])[:8]:
        print(f"  {t}: {c}")
    print(f"Difficulty: {dict(sorted(diffs.items()))}")


def main():
    # ── Text 50 ──────────────────────────────────────────────────────────────
    src1 = DATA_DIR / "qna_raw_text_500.json"
    items1 = json.loads(src1.read_text())
    print(f"Loaded {len(items1)} text candidates from {src1.name}")

    text_50 = stratified_sample(
        items1, TARGET_PER_GROUP,
        key_fn=lambda r: (r.get("topic", "unknown"), r.get("difficulty", 0)),
    )
    out1 = DATA_DIR / "qna_text_50.json"
    out1.write_text(json.dumps(text_50, ensure_ascii=False, indent=2))
    summarize("TEXT 50", text_50)
    print(f"→ {out1.name}")

    # ── Vision 50 (metadata only, image fetched in next step) ────────────────
    src2 = DATA_DIR / "qna_raw_vision_meta_500.json"
    items2 = json.loads(src2.read_text())
    print(f"\nLoaded {len(items2)} vision candidates from {src2.name}")

    vision_50 = stratified_sample(
        items2, TARGET_PER_GROUP,
        key_fn=lambda r: (r.get("topic", "unknown"), r.get("difficulty", 0)),
    )
    out2 = DATA_DIR / "qna_vision_50_ids.json"
    out2.write_text(json.dumps(vision_50, ensure_ascii=False, indent=2))
    summarize("VISION 50", vision_50)
    print(f"→ {out2.name}")

    print("\nNext: uv run python data/01_fetch_vision_images.py  (fetch base64 images)")


if __name__ == "__main__":
    main()
