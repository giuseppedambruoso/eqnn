#!/bin/bash
# One-time setup of a Linux machine for the paper campaign.
# Run from the repository root:  scripts/setup_campaign_machine.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Python 3.11 or 3.12 required"'

echo "== Python environment (.venv)"
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-campaign.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

echo "== Kaggle credentials (SATELLITE, PlanesNet)"
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "ERROR: put your Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true

echo "== Shared results and Ising data (branch campaign-results)"
scripts/sync_results.sh
mkdir -p data/ising
cp "${EQNN_RESULTS_REPO:-../eqnn-campaign-results}"/data/ising/*.npz data/ising/

echo "== Datasets (MNIST, EuroSAT, Kaggle chips): download + check"
WANDB_MODE=disabled .venv/bin/python - <<'PY'
import torchvision
torchvision.datasets.MNIST("data", train=True, download=True)
torchvision.datasets.MNIST("data", train=False, download=True)
torchvision.datasets.EuroSAT("data", download=True)
from src.paper_experiments import TASKS, loaders_for
for task in TASKS:
    loaders_for(task, 40, 1, 8)
    print("ok:", task)
PY

echo "== Equivariance tests"
WANDB_MODE=disabled .venv/bin/python -m pytest -q tests/test_multiqubit_equivariance.py

echo
echo "Setup complete. See CAMPAIGN.md for how to start the campaign."
