#!/usr/bin/env bash
# m1_serve_vmlx.sh
# Start vmlx OpenAI-compatible server with the locally downloaded MLX model.
#
# Prerequisites:
#   uv tool install vmlx        (one-time)
#   bash scripts/m1_pull_mlx_model.sh   (download model from Drive first)
#
# Usage:
#   bash scripts/m1_serve_vmlx.sh                # serve on default port 8000
#   bash scripts/m1_serve_vmlx.sh --port 8001    # custom port
#   bash scripts/m1_serve_vmlx.sh --check        # check vmlx + model presence only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/models/gemma-4-e2b-it-mlx-4bit"
PORT=8000
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  vmlx server (Gemma 4 E2B 4-bit, MLX)"
echo "═══════════════════════════════════════════════════════════════"

# 1. Check vmlx
if ! command -v vmlx >/dev/null 2>&1; then
    echo "❌ vmlx not installed. Run:"
    echo "   uv tool install vmlx"
    exit 1
fi
echo "✅ vmlx: $(vmlx --version 2>/dev/null || echo installed)"

# 2. Check model present
if [[ ! -f "$MODEL_DIR/model.safetensors" ]] && \
   [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "❌ Model not found at $MODEL_DIR"
    echo "   Run first: bash scripts/m1_pull_mlx_model.sh"
    exit 1
fi
MODEL_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
echo "✅ Model: $MODEL_DIR ($MODEL_SIZE)"

# 3. Check port available
if lsof -i ":$PORT" >/dev/null 2>&1; then
    echo "⚠️  Port $PORT already in use:"
    lsof -i ":$PORT" | tail -n +2
    echo "   Use --port to choose another, or kill the process."
    exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo
    echo "✅ All checks passed. Run without --check to start the server."
    exit 0
fi

# 4. Memory snapshot before
echo
echo "→ Memory before vmlx start:"
bash "$REPO_ROOT/scripts/monitor.sh" snap "before-vmlx-start" 2>/dev/null || true

# 5. Start server
echo
echo "→ Starting vmlx serve..."
echo "   Model: $MODEL_DIR"
echo "   Port:  $PORT"
echo "   API:   http://localhost:${PORT}/v1/chat/completions"
echo
echo "Press Ctrl+C to stop."
echo "───────────────────────────────────────────────────────────────"
exec vmlx serve "$MODEL_DIR" --port "$PORT"
