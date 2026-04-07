#!/usr/bin/env bash
# monitor.sh — Apple Silicon memory & CPU monitor
#
# Modes:
#   bash scripts/monitor.sh watch [process]   # live display (default)
#   bash scripts/monitor.sh snap  [label]     # single snapshot → results/snapshots.jsonl
#   bash scripts/monitor.sh log   [process]   # background logging → results/monitor.log
#
# Examples:
#   bash scripts/monitor.sh watch ollama      # watch during benchmark
#   bash scripts/monitor.sh snap "before"     # record before benchmark
#   bash scripts/monitor.sh snap "during"     # record mid-benchmark
#   bash scripts/monitor.sh snap "after"      # record after benchmark
#   bash scripts/monitor.sh log ollama &      # background, stop with: kill %1

MODE="${1:-watch}"
ARG="${2:-ollama}"
PAGE_SIZE=16384
GB=$((1024 * 1024 * 1024))
RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

# ── Collectors ────────────────────────────────────────────────────────────────
mem_json() {
    vm_stat | awk -v ps=$PAGE_SIZE -v gb=$GB '
        /Pages free/     { free=$3+0 }
        /Pages inactive/ { inact=$3+0 }
        /Pages active/   { act=$3+0 }
        /Pages wired/    { wired=$4+0 }
        /Pages occupied/ { comp=$5+0 }
        END {
            printf "{\"free_gb\":%.2f,\"inactive_gb\":%.2f,\"active_gb\":%.2f,\"wired_gb\":%.2f,\"compressed_gb\":%.2f,\"available_gb\":%.2f}",
            free*ps/gb, inact*ps/gb, act*ps/gb, wired*ps/gb, comp*ps/gb, (free+inact)*ps/gb
        }'
}

swap_mb() { sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d",$6}'; }

proc_json() {
    ps aux | awk -v p="$1" '
        $11~p && !/grep/ && !/monitor/ { cpu+=$3; rss+=$6; n++ }
        END { printf "{\"cpu_pct\":%.1f,\"rss_gb\":%.2f}", (n?cpu:0), (n?rss/1048576:0) }'
}

pressure() {
    local free swap
    free=$(vm_stat | awk -v ps=$PAGE_SIZE -v gb=$GB '/Pages free/{printf "%.2f",$3*ps/gb}')
    swap=$(swap_mb)
    if   (( $(echo "$free < 0.5" | bc -l) )) || (( swap > 3000 )); then echo "HIGH"
    elif (( $(echo "$free < 1.5" | bc -l) )) || (( swap > 1500 )); then echo "MED"
    else echo "LOW"; fi
}

# ── Mode: snap (single JSON record) ──────────────────────────────────────────
if [[ "$MODE" == "snap" ]]; then
    LABEL="$ARG"
    SNAP=$(printf '{"timestamp":"%s","label":"%s","mem":%s,"swap_mb":%s,"ollama":%s,"pressure":"%s"}' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$LABEL" \
        "$(mem_json)" \
        "$(swap_mb)" \
        "$(proc_json ollama)" \
        "$(pressure)")
    echo "$SNAP" | tee -a "$RESULTS_DIR/snapshots.jsonl"
    echo ""
    # Human-readable summary
    echo "$SNAP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['mem']
o=d['ollama']
print(f\"  [{d['label']}] {d['timestamp']}\")
print(f\"  Free: {m['free_gb']}G | Available: {m['available_gb']}G | Compressed: {m['compressed_gb']}G\")
print(f\"  Swap: {d['swap_mb']}MB | Pressure: {d['pressure']}\")
print(f\"  Ollama CPU: {o['cpu_pct']}% | Ollama RAM: {o['rss_gb']}G\")
"
    exit 0
fi

# ── Mode: log (background JSONL logging) ──────────────────────────────────────
if [[ "$MODE" == "log" ]]; then
    PROC="$ARG"
    LOGFILE="$RESULTS_DIR/monitor_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "Logging to $LOGFILE (Ctrl+C or kill to stop)"
    while true; do
        printf '{"ts":"%s","mem":%s,"swap_mb":%s,"%s":%s,"pressure":"%s"}\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$(mem_json)" \
            "$(swap_mb)" \
            "$PROC" \
            "$(proc_json "$PROC")" \
            "$(pressure)" >> "$LOGFILE"
        sleep 1
    done
    exit 0
fi

# ── Mode: watch (live table) ──────────────────────────────────────────────────
PROC="$ARG"
PROC_UP="$(echo "$PROC" | tr a-z A-Z)"
# Columns fit in 79-char pane: 9+8+8+9+9+8+8+6 = 65
HDR="%-9s %-8s %-8s %-9s %-9s %-8s %-8s %s"
SEP="$(printf -- '-%.0s' {1..68})"
ROW=0

echo "=== Apple Silicon Monitor | $PROC | $(date '+%Y-%m-%d') ==="
echo "    (full metrics: bash scripts/monitor.sh snap <label>)"
echo ""

while true; do
    if (( ROW % 20 == 0 )); then
        printf "$HDR\n" \
            "TIME" "FREE_GB" "AVAIL_GB" "SWAP_MB" \
            "${PROC_UP}_CPU" "${PROC_UP}_RAM" "PRESS"
        echo "$SEP"
    fi

    read -r free inact <<< \
        "$(vm_stat | awk -v ps=$PAGE_SIZE -v gb=$GB '
            /Pages free/{f=$3+0}/Pages inactive/{i=$3+0}
            END{printf "%.2f %.2f",f*ps/gb,i*ps/gb}')"
    avail=$(echo "$free + $inact" | bc)
    swap=$(swap_mb)
    read -r pcpu pram <<< "$(proc_json "$PROC" | python3 -c \
        "import json,sys; d=json.load(sys.stdin); print(d['cpu_pct'], d['rss_gb'])")"
    press=$(pressure)

    printf "$HDR\n" \
        "$(date +%H:%M:%S)" \
        "$free" "$avail" "$swap" \
        "${pcpu}%" "${pram}G" "[$press]"

    ROW=$((ROW + 1))
    sleep 1
done
