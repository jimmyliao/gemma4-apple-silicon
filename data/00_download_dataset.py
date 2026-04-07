#!/usr/bin/env python3
"""
Download hhhuang/TaiwanVQA and prepare two candidate pools:
  data/qna_raw_text_500.json        — text-only candidates (NO image bytes)
  data/qna_raw_vision_meta_500.json — vision candidates  (NO image bytes, only metadata)

Memory-safe: uses streaming mode, drops image bytes per row.
Image bytes are only fetched later (data/01_fetch_vision_images.py) for 50 selected.

Run: uv run python data/00_download_dataset.py
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
random.seed(42)


def fmt(row, mode):
    """Extract only metadata fields. NEVER include image bytes."""
    return {
        "question_id": row.get("question_id", ""),
        "topic":       row.get("topic", ""),
        "sub_topic":   row.get("sub_topic", ""),
        "difficulty":  row.get("difficulty_level", 0),
        "is_ocr":      row.get("is_ocr", False),
        "question_type": row.get("question_type", ""),
        "question":    row.get("question", ""),
        "A": row.get("A", ""),
        "B": row.get("B", ""),
        "C": row.get("C", ""),
        "D": row.get("D", ""),
        "answer":      row.get("answer", ""),
        "prompt": (
            f"問題：{row.get('question','')}\n"
            f"A. {row.get('A','')}\nB. {row.get('B','')}\nC. {row.get('C','')}\nD. {row.get('D','')}\n\n"
            f"請選出正確答案（只需回答 A、B、C 或 D）。"
        ),
        "expected_tokens": 3,
        "mode": mode,
    }


def main():
    print("Loading hhhuang/TaiwanVQA (config=data, split=test, streaming) ...")
    from datasets import load_dataset

    # streaming=True → IterableDataset, never loads all rows at once
    ds = load_dataset("hhhuang/TaiwanVQA", name="data", split="test", streaming=True)

    text_pool = []
    vision_pool = []
    n = 0

    for row in ds:
        n += 1
        # Extract metadata IMMEDIATELY, then let the row's image bytes be GC'd
        is_ocr = row.get("is_ocr", False)
        has_image = row.get("image") is not None

        if not is_ocr:
            text_pool.append(fmt(row, "text"))
        if has_image:
            vision_pool.append(fmt(row, "vision"))

        # Progress indicator every 500 rows
        if n % 500 == 0:
            print(f"  ...processed {n} rows | text={len(text_pool)} vision={len(vision_pool)}")

    print(f"\nTotal rows processed: {n}")
    print(f"  Text candidates (is_ocr=False): {len(text_pool)}")
    print(f"  Vision candidates (has image):  {len(vision_pool)}")

    # Sample 500 from each pool
    text_sample = random.sample(text_pool, min(500, len(text_pool)))
    vision_sample = random.sample(vision_pool, min(500, len(vision_pool)))

    out1 = DATA_DIR / "qna_raw_text_500.json"
    out1.write_text(json.dumps(text_sample, ensure_ascii=False, indent=2))
    print(f"\n✅ {out1.name}: {len(text_sample)} entries")

    out2 = DATA_DIR / "qna_raw_vision_meta_500.json"
    out2.write_text(json.dumps(vision_sample, ensure_ascii=False, indent=2))
    print(f"✅ {out2.name}: {len(vision_sample)} entries")

    print("\nNext: make prepare-qna  (uses gemini CLI to filter 50 + 50)")


if __name__ == "__main__":
    main()
