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
log_step "1" "🐍 Conda Environment Setup"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda info --envs | grep -q "eqnn_env"; then
    log_info "Creating 'eqnn_env' environment..."
    conda create -n eqnn_env python=3.11 -y > /dev/null 2>&1
fi
conda activate eqnn_env
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
log_success "Environment active (Python $PYTHON_VER)"

# --- PHASE 2 ---
log_step "2" "📦 Build System Check (Poetry)"
if ! command -v poetry &> /dev/null; then
    log_info "Poetry not found. Installing..."
    pip install poetry > /dev/null 2>&1
fi
log_success "Poetry is ready to use"

# --- PHASE 3 ---
log_step "3" "🔗 Dependency Resolution"
log_info "Syncing autoray, pennylane, and matplotlib..."
poetry add autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 matplotlib Pillow > /dev/null 2>&1 || true
poetry install > /dev/null 2>&1
log_success "Python packages updated"

# --- PHASE 4 ---
log_step "4" "⚡ Hardware Configuration (A30 GPU)"
log_info "Installing cuQuantum toolchain and Lightning GPU..."
pip install autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 pennylane-lightning-gpu==0.44.0 cuquantum-python-cu12 --quiet
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log_success "GPU acceleration and threads configured"

# --- PHASE 5 ---
log_step "5" "🧪 Unit Testing"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/eqnn
log_info "Checking invariance and dataset balance..."

if pytest tests/ -n 4 --quiet --disable-warnings > /dev/null 2>&1; then
    log_success "All tests passed successfully"
else
    log_error "Tests failed. Run pytest manually for details."
    exit 1
fi

# --- PHASE 6 ---
log_step "6" "🚀 Experiment Execution (Hydra)"
log_info "Starting distributed training..."
log_info "Please wait, this operation will take some time."

python src/eqnn/main.py -m \
    GENERAL.seed=1,2,3,4,5,6,7,8,9,10 \
    DATA.N=20,40,80,160,320,640,1280,2560,5120 \
    QNN.non_equivariance=3,4 \
    QNN.p_err=0 \
    QNN.reps=1,2,3 \
    TRAINING.epochs=60 \
    DATA.dataset='mnist' \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=27 \
    hydra.hydra_logging.root.level=ERROR \
    hydra.job_logging.root.level=ERROR > /dev/null 2>&1

log_success "Training completed"

# ==============================================================================
# CONCLUSION
# ==============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo -e "\n${C_GREEN}╭──────────────────────────────────────────────────╮${C_RESET}"
echo -e "${C_GREEN}│${C_BOLD}  ✨ PIPELINE COMPLETED SUCCESSFULLY              ${C_RESET}${C_GREEN}│${C_RESET}"
echo -e "${C_GREEN}╰──────────────────────────────────────────────────╯${C_RESET}"
echo -e "   ⏱️  Total time  : ${C_BOLD}${MINUTES}m ${SECONDS}s${C_RESET}\n"
