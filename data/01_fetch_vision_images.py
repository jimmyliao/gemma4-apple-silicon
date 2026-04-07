#!/usr/bin/env python3
"""
Fetch images for the 50 vision QnA selected by stratified sampling.

Reads:  data/qna_vision_50_ids.json  (list with question_id)
Writes: data/qna_vision_50.json      (same + base64 image)

Memory-safe: streams the dataset, only keeps image bytes for selected IDs.
"""
import base64
import io
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent


def main():
    ids_file = DATA_DIR / "qna_vision_50_ids.json"
    if not ids_file.exists():
        raise SystemExit(f"ERROR: {ids_file} not found. Run 02_stratified_sample.py first.")

    selected = {r["question_id"]: r for r in json.loads(ids_file.read_text())}
    print(f"Looking for {len(selected)} question IDs in TaiwanVQA test split...")

    from datasets import load_dataset
    ds = load_dataset("hhhuang/TaiwanVQA", name="data", split="test", streaming=True)

    found = {}
    n = 0
    for row in ds:
        n += 1
        qid = row.get("question_id", "")
        if qid in selected and qid not in found:
            entry = selected[qid].copy()
            img = row.get("image")
            if img is None:
                continue
            # img can be a PIL Image, dict with 'bytes', or raw bytes
            if isinstance(img, dict) and "bytes" in img:
                img_bytes = img["bytes"]
            elif hasattr(img, "save"):  # PIL Image
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
            elif isinstance(img, (bytes, bytearray)):
                img_bytes = bytes(img)
            else:
                continue
            entry["image_base64"] = base64.b64encode(img_bytes).decode()
            entry["image_format"] = "jpeg"
            entry["image_size_kb"] = round(len(img_bytes) / 1024, 1)
            found[qid] = entry

            # Early exit when all found
            if len(found) == len(selected):
                print(f"  All {len(selected)} images found at row {n}")
                break

        if n % 1000 == 0:
            print(f"  ...scanned {n} rows, found {len(found)}/{len(selected)}")

    result = list(found.values())
    out = DATA_DIR / "qna_vision_50.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    total_kb = sum(r["image_size_kb"] for r in result)
    print(f"\n✅ Saved {len(result)} vision QnA → {out.name} ({total_kb:.0f} KB images)")

    missing = set(selected) - set(found)
    if missing:
        print(f"⚠️  {len(missing)} question_ids not found: {list(missing)[:5]}")


if __name__ == "__main__":
    main()
