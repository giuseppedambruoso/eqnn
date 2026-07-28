#!/bin/bash
set -e
set -o pipefail

# ==============================================================================
# CONFIGURATION AND VARIABLES
# ==============================================================================
START_TIME=$(date +%s)

# Terminal Colors
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_BLUE='\033[34m'
C_CYAN='\033[36m'
C_GREEN='\033[32m'
C_RED='\033[31m'

# ==============================================================================
# LOGGING FUNCTIONS (UI / UX)
# ==============================================================================

# UI: New step header
log_step() {
    echo -e "\n${C_BOLD}${C_CYAN}▶ [$1/6] $2${C_RESET}"
}

# UI: Ongoing info (tree branch)
log_info() {
    echo -e "  ${C_DIM}├─${C_RESET} $1"
}

# UI: Success (tree closure)
log_success() {
    echo -e "  ${C_GREEN}╰─ ✔ ${C_BOLD}$1${C_RESET}"
}

# UI: Error
log_error() {
    echo -e "  ${C_RED}╰─ ✖ $1${C_RESET}"
}

# Forced exit handling (CTRL+C)
trap 'echo -e "\n  ${C_RED}╰─ ✖ Pipeline interrupted by user.${C_RESET}"; exit 1' INT

# ==============================================================================
# PIPELINE START
# ==============================================================================
clear
echo -e "${C_CYAN}╭──────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_CYAN}│${C_RESET}${C_BOLD}             EQNN EXPERIMENT PIPELINE             ${C_RESET}${C_CYAN}│${C_RESET}"
echo -e "${C_CYAN}╰──────────────────────────────────────────────────╯${C_RESET}"

# --- PHASE 1 ---
log_step "1" "🐍 Virtual Environment Setup (venv)"

if [ ! -d ".venv" ]; then
    log_info "Creating '.venv' environment (Python 3.12)..."
    python3.12 -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
    log_error "Impossibile trovare l'ambiente virtuale. Assicurati di aver installato python3.12-venv (es. sudo apt install python3.12-venv)"
    exit 1
fi

source .venv/bin/activate

if command -v poetry &> /dev/null; then
    poetry env use "$(which python)" > /dev/null 2>&1 || true
fi

PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
log_success "Environment active (Python $PYTHON_VER)"

# --- PHASE 2 ---
log_step "2" "📦 Build System Check (Poetry)"
if ! command -v poetry &> /dev/null; then
    log_info "Poetry not found. Installing..."
    pip install poetry > /dev/null 2>&1
    poetry env use "$(which python)" > /dev/null 2>&1 || true
fi
log_success "Poetry is ready to use"

# --- PHASE 3 ---
log_step "3" "🔗 Dependency Resolution"
log_info "Installing dependencies from pyproject.toml..."
poetry install > /dev/null 2>&1
log_success "Python packages configured correctly"

# --- PHASE 4 ---
log_step "4" "⚡ Hardware Configuration (A30 GPU)"
log_info "Installing cuQuantum toolchain and Lightning GPU..."
# Redirecting to /dev/null to completely silence pip
pip install --upgrade pip > /dev/null 2>&1 || true
pip install autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 pennylane-lightning-gpu==0.44.0 cuquantum-python-cu12 > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log_success "GPU acceleration and threads configured"

# --- PHASE 5 ---
log_step "5" "🧪 Unit Testing"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/eqnn
log_info "Checking invariance and dataset balance..."

TMP_TEST_LOG=$(mktemp)

if pytest tests/test_equivariance.py -n 4 --disable-warnings > "$TMP_TEST_LOG" 2>&1; then
    log_success "All tests passed successfully"
    rm -f "$TMP_TEST_LOG"
else
    log_error "Tests failed! Ecco il report dettagliato del fallimento:"
    echo -e "${C_RED}┌────────────────────────────────────────────────────────────────────────────┐${C_RESET}"
    cat "$TMP_TEST_LOG"
    echo -e "${C_RED}└────────────────────────────────────────────────────────────────────────────┘${C_RESET}"
    rm -f "$TMP_TEST_LOG"
    exit 1
fi

# --- PHASE 6 ---
log_step "6" "🚀 Experiment Execution (Hydra)"
log_info "Starting distributed training..."
log_info "Please wait, this operation will take some time."

python src/eqnn/main.py -m \
    GENERAL.seed=1 \
    DATA.N=80 \
    QNN.equivariance=True \
    QNN.twirling=False,True \
    QNN.p_err=0 \
    QNN.reps=1 \
    TRAINING.epochs=60 \
    DATA.dataset='mnist' \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=4 \
    hydra.hydra_logging.root.level=ERROR \
    hydra.job_logging.root.level=ERROR

log_success "Training completed"

# ==============================================================================
# CONCLUSION
# ==============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo -e "\n${C_GREEN}╭──────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_GREEN}│${C_BOLD}   ✨ PIPELINE COMPLETED SUCCESSFULLY               ${C_RESET}${C_GREEN}│${C_RESET}"
echo -e "${C_GREEN}╰──────────────────────────────────────────────────╯${C_RESET}"
echo -e "    ⏱️  Total time  : ${C_BOLD}${MINUTES}m ${SECONDS}s${C_RESET}\n"
