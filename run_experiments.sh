#!/bin/bash
set -e 

# ==============================================================================
# CONFIGURAZIONE STILI E COLORI
# ==============================================================================
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_BLUE='\033[34m'
C_CYAN='\033[36m'
C_GREEN='\033[32m'
C_RED='\033[31m'

log_header() { echo -e "\n${C_BOLD}${C_CYAN}==> [$1/6] $2${C_RESET}"; }
log_info() { echo -e "  ${C_BLUE}ℹ${C_RESET} ${C_DIM}$1${C_RESET}"; }
log_success() { echo -e "  ${C_GREEN}✔${C_RESET} $1"; }
log_error() { echo -e "  ${C_RED}✖${C_RESET} $1"; }

# ==============================================================================
# INIZIO PIPELINE
# ==============================================================================
clear
echo -e "${C_BOLD}${C_BLUE}=======================================================${C_RESET}"
echo -e "${C_BOLD}             EQNN EXPERIMENT PIPELINE                  ${C_RESET}"
echo -e "${C_BOLD}${C_BLUE}=======================================================${C_RESET}"

log_header "1" "Configurazione Ambiente Conda"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda info --envs | grep -q "eqnn_env"; then
    log_info "Creazione ambiente eqnn_env..."
    conda create -n eqnn_env python=3.11 -y > /dev/null
fi
conda activate eqnn_env
log_success "Ambiente attivo: $(python --version)"

log_header "2" "Verifica Build System (Poetry)"
if ! command -v poetry &> /dev/null; then
    pip install poetry > /dev/null
fi
log_success "Poetry configurato."

log_header "3" "Risoluzione delle Dipendenze"
# Sincronizziamo autoray e pennylane per evitare i messaggi rossi di conflitto
poetry add autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 > /dev/null 2>&1 || true
poetry install > /dev/null
log_success "Dipendenze base sincronizzate."

log_header "4" "Configurazione Accelerazione Hardware (A30)"
# Installiamo le versioni specifiche che soddisfano i requisiti del progetto
pip install autoray==0.8.2 pennylane==0.44.0 pennylane-lightning==0.44.0 pennylane-lightning-gpu==0.44.0 cuquantum-python-cu12 --quiet
log_success "Hardware acceleration pronta e versioni allineate."

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

log_header "5" "Validazione Modello (Unit Testing Silenzioso)"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/eqnn
log_info "Esecuzione test p4m in parallelo..."

if pytest tests/test_equivariance.py -n 4 --quiet --disable-warnings; then
    log_success "Test superati con successo."
else
    log_error "Test falliti."
    exit 1
fi

log_header "6" "Esecuzione Esperimenti (Hydra)"
log_info "Avvio job batch..."
echo -e "${C_DIM}-------------------------------------------------------${C_RESET}\n"

python src/eqnn/main.py -m \
    GENERAL.seed=1,2,3,4,5,6,7,8,9,10 \
    DATA.N=20,40,80,160,320,640,1280 \
    QNN.non_equivariance=3 \
    QNN.p_err=0 \
    QNN.reps=1,2,3 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=30 \
    hydra.hydra_logging.root.level=ERROR \
    hydra.job_logging.root.level=ERROR

echo -e "\n${C_BOLD}${C_GREEN}=======================================================${C_RESET}"
echo -e "${C_BOLD}${C_GREEN} ✔ PIPELINE COMPLETATA SENZA ERRORI                       ${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}=======================================================${C_RESET}\n"
