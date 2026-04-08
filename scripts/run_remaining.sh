#!/usr/bin/env bash
# Auto chain Run 4-8 after Run 3 finishes.
# Run 3 (PID 11540) must already be running.
set -u
cd /Users/jimmyliao/workspace/lab/gemma4-apple-silicon

LOG=/tmp/bench_chain.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%H:%M:%S')] Chain started. Waiting for Run 3 (PID 11540) to finish..."
while ps -p 11540 > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] Run 3 finished."

# ============================================================
# Run 4: OFF think vision (longest run)
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Pre-Run 4: force unload gemma4:e2b to free KV cache ==="
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:e2b","keep_alive":0}' > /dev/null
sleep 3
echo "[$(date '+%H:%M:%S')] === Run 4/8: OFF think vision (num_ctx=2048, keep_alive=30s) ==="
bash scripts/monitor.sh snap m4-before > /dev/null
uv run python benchmark/benchmark.py --backend ollama --model gemma4:e2b \
    --dataset data/qna_vision_50.json --mode vision --repeats 3 --think --max-tokens 256 \
    --num-ctx 2048 --keep-alive 30s \
    --output results/m_off_think_vision.json
bash scripts/monitor.sh snap m4-after > /dev/null
echo "[$(date '+%H:%M:%S')] Run 4 done"

# ============================================================
# Restart Ollama with FLASH_ATTENTION=1
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Restarting Ollama with FLASH_ATTENTION=1 ==="
pkill -f "ollama serve" 2>/dev/null
sleep 3
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup /Applications/Ollama.app/Contents/Resources/ollama serve > /tmp/ollama_fa_on.log 2>&1 &
sleep 5
grep FLASH_ATTENTION /tmp/ollama_fa_on.log | head -1
curl -s http://localhost:11434/api/version

# ============================================================
# Run 5: ON nothink text
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Run 5/8: ON nothink text ==="
bash scripts/monitor.sh snap m5-before > /dev/null
uv run python benchmark/benchmark.py --backend ollama --model gemma4-e2b-nothink \
    --dataset data/qna_text_50.json --mode text --repeats 3 \
    --num-ctx 2048 --keep-alive 30s \
    --output results/m_on_nothink_text.json
bash scripts/monitor.sh snap m5-after > /dev/null
echo "[$(date '+%H:%M:%S')] Run 5 done"

# ============================================================
# Run 6: ON nothink vision
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Run 6/8: ON nothink vision ==="
bash scripts/monitor.sh snap m6-before > /dev/null
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4-e2b-nothink","keep_alive":0}' > /dev/null
sleep 3
uv run python benchmark/benchmark.py --backend ollama --model gemma4-e2b-nothink \
    --dataset data/qna_vision_50.json --mode vision --repeats 3 \
    --num-ctx 2048 --keep-alive 30s \
    --output results/m_on_nothink_vision.json
bash scripts/monitor.sh snap m6-after > /dev/null
echo "[$(date '+%H:%M:%S')] Run 6 done"

# ============================================================
# Run 7: ON think text
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Run 7/8: ON think text ==="
bash scripts/monitor.sh snap m7-before > /dev/null
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4-e2b-nothink","keep_alive":0}' > /dev/null
sleep 3
uv run python benchmark/benchmark.py --backend ollama --model gemma4:e2b \
    --dataset data/qna_text_50.json --mode text --repeats 3 --think --max-tokens 256 \
    --num-ctx 2048 --keep-alive 30s \
    --output results/m_on_think_text.json
bash scripts/monitor.sh snap m7-after > /dev/null
echo "[$(date '+%H:%M:%S')] Run 7 done"

# ============================================================
# Run 8: ON think vision
# ============================================================
echo ""
echo "[$(date '+%H:%M:%S')] === Run 8/8: ON think vision ==="
bash scripts/monitor.sh snap m8-before > /dev/null
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:e2b","keep_alive":0}' > /dev/null
sleep 3
uv run python benchmark/benchmark.py --backend ollama --model gemma4:e2b \
    --dataset data/qna_vision_50.json --mode vision --repeats 3 --think --max-tokens 256 \
    --num-ctx 2048 --keep-alive 30s \
    --output results/m_on_think_vision.json
bash scripts/monitor.sh snap m8-after > /dev/null
echo "[$(date '+%H:%M:%S')] Run 8 done"

echo ""
echo "[$(date '+%H:%M:%S')] ALL 8 RUNS COMPLETE"
ls -lh results/m_*.json
