#!/usr/bin/env bash
# One-off: wait for the running L=4 sweep to finish, stop the main driver
# (scripts/run_layers_study.sh) before it starts L=6, then run L=8 directly.
set -euo pipefail
cd "$(dirname "$0")/.."

while ! grep -q "layers=4 done" results_paper/layers_study.log 2>/dev/null; do
    sleep 15
done

kill 393529 2>/dev/null || true
sleep 2
echo "=== driver stopped after L=4, launching L=8 directly (L=6 skipped) ===" >> results_paper/layers_study.log

TASKS="mnist45 satellite ising eurosat_fi galaxy_round_edgeon galaxy_round_spiral resisc_airport_harbor"
.venv/bin/python -m src.paper_experiments --qubits 8 --tasks $TASKS \
    --kinds sweep --layers 8 >> results_paper/layers_study_L8.log 2>&1
echo "=== layers=8 done ===" >> results_paper/layers_study.log
echo "ALL LAYERS DONE (L=6 skipped)" >> results_paper/layers_study.log
