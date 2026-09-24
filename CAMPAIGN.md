# Paper campaign on several machines

The campaign (`src/paper_experiments.py`) compares Equiv (config6),
NonEquiv (config7) and NonEquiv-Twirled (config10) on six tasks (MNIST 4 vs 5,
SATELLITE, PlanesNet, Ising, EuroSAT Highway/River, EuroSAT Forest/Industrial).
It is split into **six rounds, one parameter seed per round**; rounds are
independent, so different machines can run different rounds at the same time.

Results are shared through the `campaign-results` branch of this repository:
each machine pushes only its own file (`results/campaign_8q.<machine>.jsonl`)
and imports the others into `results_paper/imported/`. Plots, the progress
viewer and the campaign itself (which skips jobs already done anywhere) all use
local + imported results, so **every machine sees all results**.

## One-time setup (Linux, Python 3.11 or 3.12, git with SSH access to GitHub)

```bash
git clone -b feature/paper-campaign git@github.com:giuseppedambruoso/eqnn.git eqnn-campaign
cd eqnn-campaign
# Kaggle API token (kaggle.com -> Settings -> API -> Create New Token):
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
scripts/setup_campaign_machine.sh
```

The setup creates `.venv`, installs the pinned dependencies (CPU-only torch),
downloads MNIST / EuroSAT / the two Kaggle datasets, fetches the shared results
and the Ising configurations, and runs the equivariance tests.

## Run

Always from the repository root. Pick a short, unique name for the machine
(it names its results file), e.g. `server`:

```bash
export EQNN_MACHINE=server
# 1. results sync every 10 minutes (keep it running for the whole campaign)
nohup scripts/sync_results.sh --loop 600 > results_paper/sync.log 2>&1 &
# 2. the campaign: rounds 2-6 (round 1 runs on the laptop)
nohup .venv/bin/python -m src.paper_experiments --qubits 8 --rounds 2 3 4 5 6 \
    > results_paper/campaign_8q.log 2>&1 &
```

The number of worker processes is chosen automatically (all CPUs but two,
capped at ~0.7 GB of RAM per worker); override with `--workers K`.
An interrupted campaign is resumed by relaunching the same command: finished
jobs (here or on any other machine) are skipped.

## Monitor

```bash
.venv/bin/python -m src.progress                  # live view, Ctrl+C to quit
.venv/bin/python -m src.plot_campaign results_paper/campaign_8q.jsonl results_paper/accuracy_grid_8q.pdf
tail -f results_paper/sync.log
```

## Preliminary test: output-bias initialization (run BEFORE the campaign)

Compares the default output-bias init (w = 1, b = 0) with a data-dependent
one (output z-scored on the training set) for Equiv and NonEquiv on all six
tasks, N = 80, 5 seeds (120 runs, resumable):

```bash
export EQNN_MACHINE=server
nohup scripts/sync_results.sh --loop 600 > results_paper/sync.log 2>&1 &
nohup .venv/bin/python -m src.init_comparison > results_paper/init_comparison.log 2>&1 &
tail -f results_paper/init_comparison.log          # progress; the table is printed at the end
.venv/bin/python -m src.init_comparison --summary-only   # table at any time
```
