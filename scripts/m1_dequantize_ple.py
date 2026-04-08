#!/usr/bin/env python3
"""
m1_dequantize_ple.py — Route A fix for Gemma 4 PLE quantization bug.

Problem: mlx_lm.convert(quantize=True) on Gemma 4 quantizes Per-Layer
Embedding (PLE) layers, which have hardcoded scalar multipliers
(per_layer_input_scale, per_layer_projection_scale,
embed_tokens_per_layer_scale). Quantization noise gets amplified by
these scalars and propagates through 35 decoder layers, producing
garbage output (e.g., '台灣：台灣：台灣：') on the Metal backend.

Fix: Read existing model.safetensors, find the 72 quantized PLE tensors
(.weight + .scales + .biases triples), dequantize back to bf16, write
new model. mlx-lm's loader will automatically treat layers without
matching `.scales` entries as non-quantized (see mlx_lm/utils.py
class_predicate at line ~348).

Usage:
    uv run python scripts/m1_dequantize_ple.py
    uv run python scripts/m1_dequantize_ple.py \\
        --src models/gemma-4-e2b-it-mlx-4bit \\
        --dst models/gemma-4-e2b-it-mlx-4bit-ple-fixed
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

# Layers that have hardcoded scalar multipliers in gemma4_text.py and must
# NOT be quantized:
#   - embed_tokens          → embed_scale = sqrt(hidden_size) ≈ 39.2
#                              ALSO tied to lm_head (output projection)
#   - embed_tokens_per_layer → embed_tokens_per_layer_scale = sqrt(256) = 16
#   - per_layer_input_gate    → per_layer_input_scale = 2^-0.5 ≈ 0.707
#   - per_layer_projection    → per_layer_projection_scale = 1536^-0.5 ≈ 0.0255
#   - per_layer_model_projection (top-level wrapper for per_layer)
#
# All these scalars amplify quantization noise catastrophically.
SCALE_SENSITIVE_PATTERNS = (
    "embed_tokens",          # matches embed_tokens AND embed_tokens_per_layer
    "per_layer",             # matches per_layer_input_gate, per_layer_projection, per_layer_model_projection
    "lm_head",               # in case it's not tied
)


def is_scale_sensitive(tensor_name: str) -> bool:
    """Identify tensors with hardcoded scalar multipliers (must stay in bf16)."""
    return any(p in tensor_name for p in SCALE_SENSITIVE_PATTERNS)


# Backwards-compat alias (used in older code paths)
is_ple = is_scale_sensitive


def dequantize_ple(src_dir: Path, dst_dir: Path, group_size: int = 64, bits: int = 4):
    src_safetensors = src_dir / "model.safetensors"
    if not src_safetensors.exists():
        raise SystemExit(f"❌ Source model not found: {src_safetensors}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy non-weight metadata files (config, tokenizer, README, etc.)
    print(f"→ Copying non-weight files to {dst_dir}")
    copied = 0
    for f in src_dir.iterdir():
        if f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            continue
        shutil.copy2(f, dst_dir / f.name)
        copied += 1
    print(f"  copied {copied} files")

    # 2. Load all weights
    print(f"\n→ Loading weights from {src_safetensors}")
    weights = mx.load(str(src_safetensors))
    print(f"  loaded {len(weights)} tensors")

    # 3. Identify PLE quantized triples and dequantize
    print(f"\n→ Scanning for PLE quantized layers...")
    ple_bases = set()
    for name in weights:
        if is_ple(name) and name.endswith(".scales"):
            base = name[: -len(".scales")]
            if f"{base}.weight" in weights and f"{base}.biases" in weights:
                ple_bases.add(base)

    print(f"  found {len(ple_bases)} PLE quantized layers to dequantize")
    if len(ple_bases) == 0:
        raise SystemExit("❌ No PLE quantized layers found. Either already fixed or not a Gemma 4 model.")

    # Show examples
    for base in sorted(ple_bases)[:3]:
        print(f"    • {base}")
    if len(ple_bases) > 3:
        print(f"    ... and {len(ple_bases) - 3} more")

    # 4. Build new weight dict
    print(f"\n→ Dequantizing PLE weights to bfloat16...")
    new_weights = {}
    deq_count = 0
    skipped_count = 0
    kept_count = 0

    for name, tensor in weights.items():
        if is_ple(name):
            # Drop .scales and .biases for PLE layers (we'll inline the bf16 weight)
            if name.endswith(".scales") or name.endswith(".biases"):
                base = name.rsplit(".", 1)[0]
                if base in ple_bases:
                    skipped_count += 1
                    continue
                # Not part of a complete triple — keep it as-is
                new_weights[name] = tensor
                continue
            # PLE .weight: dequantize if it has matching scales/biases
            if name.endswith(".weight"):
                base = name[: -len(".weight")]
                if base in ple_bases:
                    w = weights[f"{base}.weight"]
                    s = weights[f"{base}.scales"]
                    b = weights[f"{base}.biases"]
                    deq = mx.dequantize(w, scales=s, biases=b, group_size=group_size, bits=bits)
                    new_weights[name] = deq.astype(mx.bfloat16)
                    deq_count += 1
                    continue
            # PLE non-weight (e.g., RMSNorm.weight which isn't in a triple)
            new_weights[name] = tensor
            kept_count += 1
        else:
            # Non-PLE: keep as-is (still quantized 4-bit)
            new_weights[name] = tensor

    print(f"  dequantized:    {deq_count} tensors")
    print(f"  skipped:        {skipped_count} (.scales/.biases)")
    print(f"  PLE non-quant:  {kept_count} (e.g., RMSNorm)")
    print(f"  total output:   {len(new_weights)} tensors (was {len(weights)})")

    # 5. Save new safetensors
    dst_safetensors = dst_dir / "model.safetensors"
    print(f"\n→ Saving to {dst_safetensors}")
    mx.save_safetensors(str(dst_safetensors), new_weights)
    src_size = src_safetensors.stat().st_size / 1024**3
    dst_size = dst_safetensors.stat().st_size / 1024**3
    print(f"  source size: {src_size:.2f} GB")
    print(f"  output size: {dst_size:.2f} GB ({(dst_size-src_size)*1024:+.0f} MB delta)")

    # 6. Regenerate model.safetensors.index.json
    idx_path = dst_dir / "model.safetensors.index.json"
    idx = {
        "metadata": {"total_size": dst_safetensors.stat().st_size},
        "weight_map": {k: "model.safetensors" for k in new_weights},
    }
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"  wrote index: {idx_path.name}")

    print(f"\n✅ Done. Test with:")
    print(f"   uv run python -c \"")
    print(f"   from mlx_lm import load, generate")
    print(f"   m, t = load('{dst_dir}')")
    print(f"   print(generate(m, t, prompt='問題：台灣首都是？\\nA. 台北\\nB. 高雄\\nC. 台中\\nD. 台南\\n請只回答字母：', max_tokens=10))")
    print(f"   \"")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("models/gemma-4-e2b-it-mlx-4bit"))
    ap.add_argument("--dst", type=Path, default=Path("models/gemma-4-e2b-it-mlx-4bit-ple-fixed"))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    dequantize_ple(args.src, args.dst, args.group_size, args.bits)


if __name__ == "__main__":
    main()
