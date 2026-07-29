#!/bin/bash
set -e
set -o pipefail

START_TIME=$(date +%s)

C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_BLUE='\033[34m'
C_CYAN='\033[36m'
C_GREEN='\033[32m'
C_RED='\033[31m'

log_step() {
    echo -e "\n${C_BOLD}${C_CYAN}==> STEP $1/6:${C_RESET} ${C_BOLD}$2${C_RESET}"
}

log_info() {
    echo -e "    ${C_DIM}-> $1${C_RESET}"
}

log_success() {
    echo -e "    ${C_GREEN}[ OK ]${C_RESET} $1"
}

log_error() {
    echo -e "    ${C_RED}[FAIL]${C_RESET} $1"
}

trap 'echo -e "\n    ${C_RED}[ABORT]${C_RESET} Pipeline interrupted by user."; exit 1' INT

clear
echo -e "${C_CYAN}==================================================${C_RESET}"
echo -e "${C_BOLD}             EQNN EXPERIMENT PIPELINE             ${C_RESET}"
echo -e "${C_CYAN}==================================================${C_RESET}"

log_step "1" "Virtual Environment Setup"

if [ ! -d ".venv" ]; then
    log_info "Installing Python 3.12 via uv..."
    
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi

    # Indentiamo l'output di uv
    uv python install 3.12 2>&1 | sed 's/^/    /'
    uv venv --python 3.12 --seed .venv 2>&1 | sed 's/^/    /'
fi

if [ ! -f ".venv/bin/activate" ]; then
    log_error "Virtual environment initialization failed."
    exit 1
fi

source .venv/bin/activate

if command -v poetry &> /dev/null; then
    poetry env use $(pwd)/.venv/bin/python > /dev/null 2>&1 || true
fi

PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
log_success "Environment active (Python $PYTHON_VER)"

log_step "2" "Build System Check"
if ! command -v poetry &> /dev/null; then
    log_info "Installing Poetry..."
    uv pip install --python .venv/bin/python poetry > /dev/null 2>&1
    poetry env use $(pwd)/.venv/bin/python > /dev/null 2>&1 || true
fi
log_success "Poetry ready"

log_step "3" "Dependency Resolution"
log_info "Installing dependencies..."
# Indentiamo l'output di Poetry
poetry install 2>&1 | sed 's/^/    /'
log_success "Dependencies installed"

log_step "4" "Hardware Configuration"
log_info "Configuring cuQuantum and Lightning GPU..."

uv pip install --python .venv/bin/python \
    autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 pennylane-lightning-gpu==0.44.0 \
    cuquantum-python-cu12 custatevec-cu12 cutensornet-cu12 > /dev/null 2>&1

# --- FIX CUDA/cuQuantum ---
# Trova dinamicamente la cartella site-packages
SITE_PKGS=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Aggiunge le librerie C++ di NVIDIA (incluse le sottocartelle 'nvidia') al path di Linux
export LD_LIBRARY_PATH=$SITE_PKGS/nvidia/custatevec/lib:$SITE_PKGS/nvidia/cutensor/lib:$SITE_PKGS/cuquantum/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
log_success "Hardware configured"

log_step "5" "Unit Testing"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/eqnn
log_info "Running test suite..."

TMP_TEST_LOG=$(mktemp)

if pytest tests/test_equivariance.py --disable-warnings > "$TMP_TEST_LOG" 2>&1; then
    log_success "All tests passed"
    rm -f "$TMP_TEST_LOG"
else
    log_error "Test suite failed. Report:"
    echo -e "    ${C_RED}--------------------------------------------------${C_RESET}"
    # Indentiamo il dump del file di log degli errori
    cat "$TMP_TEST_LOG" | sed 's/^/    /'
    echo -e "    ${C_RED}--------------------------------------------------${C_RESET}"
    rm -f "$TMP_TEST_LOG"
    exit 1
fi

log_step "6" "Experiment Execution"
log_info "Starting distributed training..."

# Indentiamo l'output di Hydra / Joblib
python src/eqnn/main.py -m \
    GENERAL.seed=1 \
    GENERAL.dev="cuda" \
    DATA.N=320 \
    QNN.equivariance=True \
    QNN.twirling=False \
    QNN.remove_cross_edge=True,False \
    QNN.p_err=0 \
    QNN.reps=1 \
    QNN.device="lightning.gpu" \
    TRAINING.epochs=40 \
    DATA.dataset='mnist' \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=2 \
    hydra.hydra_logging.root.level=ERROR \
    hydra.job_logging.root.level=ERROR 2>&1 | sed 's/^/    /'

log_success "Training completed"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo -e "\n${C_GREEN}==================================================${C_RESET}"
echo -e "${C_BOLD}    PIPELINE COMPLETED SUCCESSFULLY               ${C_RESET}"
echo -e "${C_GREEN}==================================================${C_RESET}"
echo -e "    Total time: ${C_BOLD}${MINUTES}m ${SECONDS}s${C_RESET}\n"
