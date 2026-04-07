#!/usr/bin/env bash
# m1_pull_mlx_model_gws.sh
# Download the MLX-converted Gemma 4 E2B model from Google Drive using
# Google Workspace CLI (`gws`).
#
# Drive folder layout (created by Colab notebook):
#   MyDrive/AI-models/gemma-4-e2b-it-mlx-4bit/
#     ├── model.safetensors        (~2.5 GB)
#     ├── tokenizer.json
#     └── ... (8 files total, ~2.5 GB)
#
# Prerequisites (one-time):
#   brew install googleworkspace-cli jq
#   gws auth setup       # interactive: gcloud + OAuth client
#   gws auth login       # interactive: browser OAuth
#
# Usage:
#   bash scripts/m1_pull_mlx_model_gws.sh
#   bash scripts/m1_pull_mlx_model_gws.sh --dry-run        # list files only
#   bash scripts/m1_pull_mlx_model_gws.sh --folder OTHER   # custom folder name

set -euo pipefail

FOLDER_NAME="gemma-4-e2b-it-mlx-4bit"
DRY_RUN=0

# gws prints "Using keyring backend: keyring" to stdout before JSON output.
# This helper strips it so jq can parse cleanly.
gws_json() {
    gws "$@" 2>/dev/null | sed -n '/^{/,$p'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --folder) FOLDER_NAME="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/models/$FOLDER_NAME"

echo "═══════════════════════════════════════════════════════════════"
echo "  Pull MLX model from Google Drive (via gws CLI)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Folder name: $FOLDER_NAME"
echo "  Local dir  : $LOCAL_DIR"
echo "  Mode       : $([ $DRY_RUN -eq 1 ] && echo 'dry-run' || echo 'real download')"
echo

# ── 1. Preflight ──────────────────────────────────────────────────────────────
for cmd in gws jq; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "❌ '$cmd' not installed. Run:"
        echo "   brew install googleworkspace-cli jq"
        exit 1
    fi
done
echo "✅ gws  $(gws --version 2>/dev/null || echo installed)"
echo "✅ jq   $(jq --version)"

# ── 2. Auth check ─────────────────────────────────────────────────────────────
auth_method=$(gws_json auth status | jq -r '.auth_method // "none"')
if [[ "$auth_method" == "none" ]]; then
    echo
    echo "❌ gws not authenticated. Run interactively:"
    echo "   gws auth setup    # one-time GCP project + OAuth client"
    echo "   gws auth login    # browser OAuth"
    echo
    exit 1
fi
echo "✅ gws auth: $auth_method"

# ── 3. Resolve folder ID ──────────────────────────────────────────────────────
echo
echo "→ Resolving folder ID for '$FOLDER_NAME'..."
folder_query="name='$FOLDER_NAME' and mimeType='application/vnd.google-apps.folder' and trashed=false"
FOLDER_LIST=$(gws_json drive files list \
    --params "$(jq -nc --arg q "$folder_query" '{q: $q, fields: "files(id,name,parents)"}')")

FOLDER_ID=$(echo "$FOLDER_LIST" | jq -r '.files[0].id // empty' 2>/dev/null || true)
if [[ -z "$FOLDER_ID" ]]; then
    echo "❌ Folder '$FOLDER_NAME' not found in your Drive."
    echo "   Did Colab notebook finish the Drive copy step?"
    echo "   Raw API response:"
    echo "$FOLDER_LIST" | head -20
    exit 1
fi
echo "✅ Folder ID: $FOLDER_ID"

# ── 4. List files in folder ───────────────────────────────────────────────────
echo
echo "→ Listing files in folder..."
list_query="parents='$FOLDER_ID' and trashed=false"
FILES_JSON=$(gws_json drive files list \
    --params "$(jq -nc --arg q "$list_query" '{q: $q, pageSize: 1000, fields: "files(id,name,mimeType,size,md5Checksum)"}')")

N_FILES=$(echo "$FILES_JSON" | jq '.files | length')
TOTAL_BYTES=$(echo "$FILES_JSON" | jq '[.files[].size | tonumber? // 0] | add')
TOTAL_GB=$(echo "scale=2; $TOTAL_BYTES / 1024 / 1024 / 1024" | bc)

echo "✅ Found $N_FILES files, total ${TOTAL_GB} GB"
echo
echo "$FILES_JSON" | jq -r '.files[] | "  \(.name)  \((.size | tonumber? // 0) / 1024 / 1024 | floor) MB"'

if [[ $DRY_RUN -eq 1 ]]; then
    echo
    echo "(dry-run) No files downloaded. Re-run without --dry-run to actually pull."
    exit 0
fi

# ── 5. Download each file ─────────────────────────────────────────────────────
mkdir -p "$LOCAL_DIR"
echo
echo "→ Downloading to $LOCAL_DIR ..."

i=0
echo "$FILES_JSON" | jq -c '.files[]' | while IFS= read -r entry; do
    i=$((i + 1))
    fid=$(echo "$entry" | jq -r '.id')
    fname=$(echo "$entry" | jq -r '.name')
    fsize=$(echo "$entry" | jq -r '.size // "0"')
    dest="$LOCAL_DIR/$fname"

    # Resume: skip if already downloaded with matching size
    if [[ -f "$dest" ]]; then
        local_size=$(stat -f%z "$dest" 2>/dev/null || echo 0)
        if [[ "$local_size" == "$fsize" ]]; then
            echo "  [$i/$N_FILES] ✓ $fname ($fsize bytes) — skip (already downloaded)"
            continue
        else
            echo "  [$i/$N_FILES] ⚠ $fname size mismatch (local=$local_size remote=$fsize) — re-download"
            rm -f "$dest"
        fi
    fi

    fsize_mb=$(echo "scale=1; $fsize / 1024 / 1024" | bc)
    echo "  [$i/$N_FILES] ↓ $fname (${fsize_mb} MB)"

    # Download via gws drive files get with alt=media.
    # IMPORTANT: gws writes binary content to the file given by -o/--output,
    # NOT to stdout. stdout receives only metadata JSON ({"bytes":N,"saved_file":..}).
    if ! gws drive files get \
        --params "$(jq -nc --arg id "$fid" '{fileId: $id, alt: "media"}')" \
        --output "$dest" \
        > /tmp/gws_download_meta.json 2>/tmp/gws_download_err.log; then
        echo "    ❌ Download failed for $fname"
        echo "    Error log:"
        cat /tmp/gws_download_err.log | head -10
        echo "    Metadata:"
        cat /tmp/gws_download_meta.json | tail -20
        exit 1
    fi

    # Sanity check post-download
    actual=$(stat -f%z "$dest" 2>/dev/null || echo 0)
    if [[ "$actual" != "$fsize" ]]; then
        echo "    ⚠ Size mismatch after download: expected $fsize, got $actual"
    fi
done

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo
echo "→ Verifying local files..."
LOCAL_SIZE=$(du -sh "$LOCAL_DIR" | awk '{print $1}')
N_LOCAL=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')
echo "✅ Downloaded: $N_LOCAL files, $LOCAL_SIZE"
ls -lh "$LOCAL_DIR" | tail -n +2

REQUIRED_MIN_SIZE=("model.safetensors:1000000000" "config.json:100" "tokenizer.json:1000000")
MISSING=0
for spec in "${REQUIRED_MIN_SIZE[@]}"; do
    f="${spec%:*}"
    min="${spec#*:}"
    if [[ ! -f "$LOCAL_DIR/$f" ]]; then
        echo "❌ Missing required file: $f"
        MISSING=1
        continue
    fi
    actual=$(stat -f%z "$LOCAL_DIR/$f" 2>/dev/null || echo 0)
    if [[ $actual -lt $min ]]; then
        echo "❌ $f too small ($actual bytes < $min minimum) — likely corrupted"
        MISSING=1
    fi
done
[[ $MISSING -eq 0 ]] && echo "✅ All required files present and properly sized"
[[ $MISSING -eq 1 ]] && exit 1

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Done! Next steps:"
echo
echo "  Start vmlx server (in another terminal/pane):"
echo "    bash scripts/m1_serve_vmlx.sh"
echo
echo "  Run MLX benchmark:"
echo "    make benchmark-mlx"
echo "═══════════════════════════════════════════════════════════════"
