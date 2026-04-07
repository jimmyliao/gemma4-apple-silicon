#!/usr/bin/env bash
# m1_pull_mlx_model.sh
# Download the MLX-converted Gemma 4 E2B model from Google Drive (rclone)
# to the local repo's models/ directory.
#
# Prerequisites (one-time setup):
#   brew install rclone
#   rclone config              # name=gdrive, type=drive, OAuth same Google account
#                              # that ran the Colab conversion notebook.
#
# Usage:
#   bash scripts/m1_pull_mlx_model.sh
#   bash scripts/m1_pull_mlx_model.sh --dry-run     # show plan without downloading
#   bash scripts/m1_pull_mlx_model.sh --remote=mygdrive  # custom rclone remote name

set -euo pipefail

REMOTE_NAME="gdrive"
DRIVE_PATH="AI-models/gemma-4-e2b-it-mlx-4bit"
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --remote=*) REMOTE_NAME="${arg#--remote=}" ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/models/gemma-4-e2b-it-mlx-4bit"

echo "═══════════════════════════════════════════════════════════════"
echo "  Pull MLX model from Google Drive"
echo "═══════════════════════════════════════════════════════════════"
echo "  Remote   : ${REMOTE_NAME}:${DRIVE_PATH}"
echo "  Local    : $LOCAL_DIR"
echo "  Mode     : ${DRY_RUN:-real download}"
echo

# 1. Verify rclone installed
if ! command -v rclone >/dev/null 2>&1; then
    echo "❌ rclone not installed. Run:"
    echo "   brew install rclone"
    exit 1
fi
echo "✅ rclone $(rclone version | head -1 | awk '{print $2}')"

# 2. Verify remote configured
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
    echo "❌ rclone remote '${REMOTE_NAME}' not configured. Run:"
    echo "   rclone config"
    echo "   (n)ew remote → name=${REMOTE_NAME} → type=drive → OAuth)"
    exit 1
fi
echo "✅ rclone remote '${REMOTE_NAME}' configured"

# 3. Verify Drive folder exists
echo
echo "→ Listing remote contents..."
if ! rclone lsd "${REMOTE_NAME}:${DRIVE_PATH}" 2>/dev/null && \
     ! rclone ls "${REMOTE_NAME}:${DRIVE_PATH}" 2>/dev/null | head -1 >/dev/null; then
    echo "❌ Drive folder not found: ${REMOTE_NAME}:${DRIVE_PATH}"
    echo "   Did the Colab notebook finish the Drive copy step?"
    exit 1
fi

REMOTE_SIZE=$(rclone size "${REMOTE_NAME}:${DRIVE_PATH}" 2>/dev/null | grep -i "Total size" || echo "size: unknown")
echo "✅ Remote folder found"
echo "   $REMOTE_SIZE"

# 4. Create local dir
mkdir -p "$LOCAL_DIR"

# 5. rclone copy
echo
echo "→ Downloading..."
rclone copy "${REMOTE_NAME}:${DRIVE_PATH}" "$LOCAL_DIR" \
    --progress \
    --transfers 4 \
    --checkers 8 \
    --multi-thread-streams 4 \
    $DRY_RUN

if [[ -n "$DRY_RUN" ]]; then
    echo
    echo "(dry-run) No files downloaded. Re-run without --dry-run to actually pull."
    exit 0
fi

# 6. Verify
echo
echo "→ Verifying local files..."
LOCAL_SIZE=$(du -sh "$LOCAL_DIR" | awk '{print $1}')
N_FILES=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')

echo "✅ Downloaded: $N_FILES files, $LOCAL_SIZE"
ls -lh "$LOCAL_DIR" | tail -n +2 | awk '{print "  " $9 "  " $5}'

# 7. Sanity check
REQUIRED=("model.safetensors" "config.json" "tokenizer.json")
for f in "${REQUIRED[@]}"; do
    if [[ ! -f "$LOCAL_DIR/$f" ]]; then
        echo "❌ Missing required file: $f"
        exit 1
    fi
done
echo "✅ All required files present"

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Done! Next steps:"
echo
echo "  Start vmlx server (in another terminal):"
echo "    bash scripts/m1_serve_vmlx.sh"
echo
echo "  Run MLX benchmark:"
echo "    make benchmark-mlx"
echo "═══════════════════════════════════════════════════════════════"
