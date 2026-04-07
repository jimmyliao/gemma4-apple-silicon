# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: L4
#     provenance: []
#     toc_visible: true
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Convert Gemma 4 E2B → MLX 4-bit (Colab)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmyliao/gemma4-apple-silicon/blob/main/notebooks/convert_gemma4_to_mlx.ipynb)
#
# **Goal**: Convert `google/gemma-4-e2b-it` to MLX 4-bit format on Colab GPU,
# then save to Google Drive (or push to HuggingFace) so M1 can pull only the ~2.7 GB
# quantized version instead of downloading the full 10 GB original.
#
# **Why Colab**:
# - Fast download from HF (Google CDN ~100 MB/s)
# - GPU acceleration via `mlx[cuda]` (Apr 2026+ support)
# - Frees M1 from doing the heavy F16 download + conversion
#
# **Hardware**: T4 (free tier) or A100/H100 (Pro). Auto-detected below.
#
# **Repo**: https://github.com/jimmyliao/gemma4-apple-silicon

# %% [markdown]
# ## 1. Runtime detection

# %%
import subprocess
import sys

def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True
        ).strip()
        return out
    except Exception as e:
        return f"No GPU: {e}"

print("Python:", sys.version.split()[0])
print("Platform:", sys.platform)
print("GPU:", gpu_info())

# %% [markdown]
# ## 2. Install MLX (CUDA backend) + dependencies
#
# - `mlx[cuda]` — Apple's MLX framework with CUDA backend (Linux x86_64)
# - `mlx-lm` — language model utilities (convert, generate, quantize)
# - `huggingface_hub` — auth + push to your repo

# %%
# !pip install -q "mlx[cuda]" mlx-lm "huggingface_hub[cli]"

# %%
import mlx.core as mx
print("MLX version:", mx.__version__ if hasattr(mx, "__version__") else "unknown")
print("MLX default device:", mx.default_device())
# Should print "Device(gpu, 0)" on CUDA-enabled runtime

# %% [markdown]
# ## 3. HuggingFace authentication
#
# `google/gemma-4-e2b-it` is a **gated model** — you must:
# 1. Go to https://huggingface.co/google/gemma-4-e2b-it
# 2. Click "Acknowledge license" and accept Google's Gemma Terms
# 3. Create a token at https://huggingface.co/settings/tokens (read access is enough for download; write is needed if pushing)
# 4. Run the cell below

# %%
from huggingface_hub import login, whoami
import getpass

# In Colab: store HF_TOKEN as a Colab Secret (key icon on left sidebar)
# Or paste interactively below
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    print("Loaded HF_TOKEN from Colab secret")
except Exception:
    hf_token = getpass.getpass("HF_TOKEN: ")

login(token=hf_token)
print("Logged in as:", whoami()["name"])

# %% [markdown]
# ## 4. Configure conversion

# %%
SOURCE_MODEL = "google/gemma-4-e2b-it"   # Original 5.1B params, F16
QUANTIZE_BITS = 4
GROUP_SIZE = 64                          # MLX default
LOCAL_OUT_DIR = "/content/gemma-4-e2b-it-mlx-4bit"

# Replace with your HF username
HF_USERNAME = whoami()["name"]
HF_REPO_ID = f"{HF_USERNAME}/gemma-4-e2b-it-mlx-4bit"

print(f"Source : {SOURCE_MODEL}")
print(f"Output : {LOCAL_OUT_DIR}")
print(f"HF repo: {HF_REPO_ID}")
print(f"Quant  : {QUANTIZE_BITS}-bit, group_size={GROUP_SIZE}")

# %% [markdown]
# ## 5. Run conversion
#
# Steps:
# 1. Download `google/gemma-4-e2b-it` (~10 GB) — Colab CDN, ~3-5 min
# 2. Quantize weights to 4-bit (~5-10 min on T4, faster on A100/H100)
# 3. Save MLX format to `LOCAL_OUT_DIR`
#
# **Memory needed**: ~12 GB peak (F16 model + quantization buffers).
# T4 (16 GB VRAM) is enough; A100 (40 GB) is comfortable.

# %%
import time
from mlx_lm import convert

t0 = time.time()
convert(
    hf_path=SOURCE_MODEL,
    mlx_path=LOCAL_OUT_DIR,
    quantize=True,
    q_bits=QUANTIZE_BITS,
    q_group_size=GROUP_SIZE,
)
elapsed = time.time() - t0
print(f"\nConversion done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# %% [markdown]
# ## 6. Verify output

# %%
import os
from pathlib import Path

out = Path(LOCAL_OUT_DIR)
total = 0
print(f"Contents of {out}:")
for p in sorted(out.iterdir()):
    size_mb = p.stat().st_size / (1024 * 1024)
    total += size_mb
    print(f"  {p.name:40s}  {size_mb:>10.1f} MB")
print(f"  {'TOTAL':40s}  {total:>10.1f} MB ({total/1024:.2f} GB)")

# %% [markdown]
# Expected output:
# - `model.safetensors` (or sharded `model-*.safetensors`) — quantized weights, **~2.7 GB total**
# - `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
# - `model.safetensors.index.json` (if sharded)
#
# ⚠️ **Known issue**: gemma-4 PLE (Per-Layer Embedding) layers may produce garbage when
# naively quantized. If smoke test below shows nonsense output, see workaround in
# section 8.

# %% [markdown]
# ## 7. Smoke test — does the converted model produce sane output?

# %%
from mlx_lm import load, generate

print("Loading converted model...")
model, tokenizer = load(LOCAL_OUT_DIR)

prompt = "問題：台灣首都是？\nA. 台北\nB. 高雄\nC. 台中\nD. 台南\n請只回答字母："
print(f"\nPrompt:\n{prompt}\n")

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=20,
    verbose=True,
)
print(f"\nResponse: {response!r}")

# %% [markdown]
# **Expected**: `"A"` or similar single-letter answer.
#
# **If garbage** (e.g., empty / random chars): hit the PLE quantization bug. Workarounds:
# 1. Try `q_bits=8` (less aggressive, larger output ~5 GB)
# 2. Skip PLE layers explicitly via custom convert script (advanced)
# 3. **Fallback to Qwen2.5-3B** — uncomment cell below

# %%
# === FALLBACK CELL — only run if Gemma 4 smoke test failed ===
# Use mlx-community pre-converted Qwen2.5-3B-Instruct-4bit (no PLE bug)
#
# from huggingface_hub import snapshot_download
# QWEN_REPO = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# QWEN_LOCAL = "/content/qwen2.5-3b-instruct-4bit"
# snapshot_download(repo_id=QWEN_REPO, local_dir=QWEN_LOCAL)
#
# from mlx_lm import load, generate
# model_q, tok_q = load(QWEN_LOCAL)
# print(generate(model_q, tok_q, prompt="台灣首都是？回答：", max_tokens=10, verbose=True))
#
# # If sane → use this for the MLX benchmark instead
# # LOCAL_OUT_DIR = QWEN_LOCAL  # uncomment to switch the rest of the notebook

# %% [markdown]
# ## 8a. Save to Google Drive (recommended when on slow external network)
#
# **Why Drive instead of HF Hub**:
# - No HF token / repo setup needed
# - Persists in your account regardless of Colab runtime lifecycle
# - Google Drive desktop app can background-sync to M1 (no manual download)
# - You can pull on M1 only when on home WiFi
#
# **Flow**:
# 1. Mount your Drive (will prompt for OAuth)
# 2. Copy `LOCAL_OUT_DIR` (~2.7 GB) to a folder in Drive
# 3. Verify file count + total size
# 4. M1 retrieval options listed below

# %%
from google.colab import drive
import shutil
from pathlib import Path

drive.mount("/content/drive")

# Drive path: My Drive/AI-models/gemma-4-e2b-it-mlx-4bit/
DRIVE_BASE = Path("/content/drive/MyDrive/AI-models")
DRIVE_OUT = DRIVE_BASE / "gemma-4-e2b-it-mlx-4bit"
DRIVE_BASE.mkdir(parents=True, exist_ok=True)

# If a previous run exists, remove it (clean overwrite)
if DRIVE_OUT.exists():
    print(f"Removing existing: {DRIVE_OUT}")
    shutil.rmtree(DRIVE_OUT)

print(f"Copying {LOCAL_OUT_DIR} → {DRIVE_OUT} ...")
t0 = time.time()
shutil.copytree(LOCAL_OUT_DIR, DRIVE_OUT)
elapsed = time.time() - t0
print(f"Copy done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Verify
total = 0
n_files = 0
for p in sorted(DRIVE_OUT.iterdir()):
    size_mb = p.stat().st_size / (1024 * 1024)
    total += size_mb
    n_files += 1
    print(f"  {p.name:40s}  {size_mb:>10.1f} MB")
print(f"\n✅ Saved to Drive: {n_files} files, {total/1024:.2f} GB total")
print(f"   Path: {DRIVE_OUT}")
print(f"   Web:  https://drive.google.com/drive/my-drive  →  AI-models/gemma-4-e2b-it-mlx-4bit/")

# %% [markdown]
# ### Retrieving from Drive on M1 (later, on home WiFi)
#
# **Option A — Google Drive Desktop app** (easiest, background sync):
# 1. Install: https://www.google.com/drive/download/
# 2. Sign in with the same account
# 3. Choose "Stream files" or "Mirror files" mode
# 4. Files appear at `~/Library/CloudStorage/GoogleDrive-<email>/My Drive/AI-models/gemma-4-e2b-it-mlx-4bit/`
# 5. Symlink or copy to your benchmark dir:
#    ```bash
#    ln -s "~/Library/CloudStorage/GoogleDrive-<your_email>/My Drive/AI-models/gemma-4-e2b-it-mlx-4bit" \
#          ~/workspace/lab/gemma4-apple-silicon/models/gemma-4-e2b-it-mlx-4bit
#    ```
#
# **Option B — `rclone` (CLI, no GUI app needed)**:
# ```bash
# brew install rclone
# rclone config           # one-time: setup gdrive remote (OAuth)
# rclone copy gdrive:AI-models/gemma-4-e2b-it-mlx-4bit ~/workspace/lab/gemma4-apple-silicon/models/gemma-4-e2b-it-mlx-4bit -P
# ```
#
# **Option C — Web download** (manual, for small files only): right-click folder → Download (Drive zips it).
#
# Then point vmlx at the local path:
# ```bash
# vmlx serve ~/workspace/lab/gemma4-apple-silicon/models/gemma-4-e2b-it-mlx-4bit --port 8000
# ```

# %% [markdown]
# ## 8b. (Optional) Also push to HuggingFace Hub
#
# If you want a public/private HF repo as backup. Skip this if Drive is enough.

# %%
PUSH_TO_HF = False   # set True to also upload to HF Hub

if PUSH_TO_HF:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"Created/exists: https://huggingface.co/{HF_REPO_ID}")

    print(f"Uploading {LOCAL_OUT_DIR} → {HF_REPO_ID} ...")
    t0 = time.time()
    api.upload_folder(
        folder_path=LOCAL_OUT_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Upload MLX 4-bit conversion of gemma-4-e2b-it",
    )
    print(f"Upload done in {time.time()-t0:.1f}s")
    print(f"\n✅ Available at: https://huggingface.co/{HF_REPO_ID}")
else:
    print("HF upload skipped. Set PUSH_TO_HF=True to enable.")

# %% [markdown]
# ## 9. Next step — pull on M1
#
# On your M1 Air (back home with WiFi), run:
#
# ```bash
# # Install vmlx (or omlx)
# uv tool install vmlx
#
# # Pull and serve the converted model
# vmlx serve <YOUR_HF_USERNAME>/gemma-4-e2b-it-mlx-4bit --port 8000
# ```
#
# Then run the existing benchmark in the gemma4-apple-silicon repo:
#
# ```bash
# cd ~/workspace/lab/gemma4-apple-silicon
# uv run python benchmark/benchmark.py \
#   --backend mlx \
#   --model <YOUR_HF_USERNAME>/gemma-4-e2b-it-mlx-4bit \
#   --dataset data/qna_text_50.json \
#   --mode text \
#   --repeats 3 \
#   --output results/mlx_text.json
# ```
#
# Compare with `results/nothink_text.json` (Ollama baseline) to see which engine wins.

# %% [markdown]
# ## Bonus: Reference benchmark on Colab GPU (transformers F16)
#
# While we're here on the GPU, run the SAME benchmark with HuggingFace transformers
# in F16 to get a Linux GPU reference number for the article.
#
# This produces a **third data point**: M1 Ollama vs M1 MLX vs Colab GPU transformers.

# %%
# !pip install -q transformers accelerate

# %%
# Optional cell — uncomment to run reference benchmark on Colab GPU
# import torch, time, json
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
# model_hf = AutoModelForCausalLM.from_pretrained(
#     SOURCE_MODEL, torch_dtype=torch.float16, device_map="cuda"
# )
# model_hf.eval()
#
# # Warm-up
# inputs = tok(prompt, return_tensors="pt").to("cuda")
# _ = model_hf.generate(**inputs, max_new_tokens=4)
#
# # Benchmark a few prompts and save results
# # ... (full benchmark loop adapted from benchmark/benchmark.py)
