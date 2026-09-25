#!/usr/bin/env bash
# Regenerates the layers-study figures every minute while new results come
# in (only when the L=8 results file has grown, to avoid useless work).
# Also logs a one-line status (jobs done, memory) for easy tailing.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS_LOG=results_paper/watch_status.log
L8_FILE=results_paper/campaign_v2_L8_8q.jsonl
last_count=-1

while true; do
    count=0
    [ -f "$L8_FILE" ] && count=$(wc -l < "$L8_FILE")
    if [ "$count" != "$last_count" ]; then
        .venv/bin/python -m src.layers_figures results_paper/gap_vs_layers.pdf >> "$STATUS_LOG" 2>&1
        last_count=$count
    fi
    mem=$(free -h | awk '/Mem:/{print $7}')
    ts=$(date '+%H:%M:%S')
    echo "[$ts] L8 righe=$count  RAM disponibile=$mem" >> "$STATUS_LOG"
    sleep 60
done
