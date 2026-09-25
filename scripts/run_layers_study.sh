#!/usr/bin/env bash
# Systematic generalization-vs-parameters study: repeats the paper's
# "sweep" jobs (train/val/val_aug accuracy vs N, per arch, 6 seeds) on all
# seven tasks at 8 qubits, for stacked-layer counts L = 2, 4, 6, 8 (L = 1
# is the already-completed main campaign, results_paper/campaign_v2_8q.jsonl).
# Each L writes to results_paper/campaign_v2_L<L>_8q.jsonl and is resumable
# (finished jobs are skipped) — safe to re-run this script if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS="mnist45 satellite ising eurosat_fi galaxy_round_edgeon galaxy_round_spiral resisc_airport_harbor"

for L in 2 4 6 8; do
    echo "=== layers=$L ===" | tee -a results_paper/layers_study.log
    .venv/bin/python -m src.paper_experiments --qubits 8 --tasks $TASKS \
        --kinds sweep --layers "$L" >> "results_paper/layers_study_L${L}.log" 2>&1
    echo "=== layers=$L done ===" | tee -a results_paper/layers_study.log
done
echo "ALL LAYERS DONE" | tee -a results_paper/layers_study.log
