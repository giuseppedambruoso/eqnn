  GNU nano 2.9.8                                                                                                                                  run_experiments_gmagnifico.sh                                                                                                                                               

#!/bin/bash

# --- Colori per l'output (Verde per OK, Blu per Info) ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Inizio configurazione e lancio esperimenti ===${NC}\n"

# --- 0. Inizializzazione Conda ---
source $HOME/miniconda3/etc/profile.d/conda.sh

# --- 1. Creazione Ambiente ---
if ! conda info --envs | grep -q "eqnn_env"; then
    echo -e "${BLUE}[1/5] Creazione ambiente 'eqnn_env' (Python 3.11)...${NC}"
    conda create -n eqnn_env python=3.11 -y
    echo -e "${GREEN}✓ Ambiente creato correttamente.${NC}\n"
else
    echo -e "${GREEN}[1/5] L'ambiente 'eqnn_env' esiste già. Salto creazione.${NC}\n"
fi

# --- 2. Attivazione ---
echo -e "${BLUE}[2/5] Attivazione ambiente...${NC}"
conda activate eqnn_env
echo -e "${GREEN}✓ Ambiente attivato: $(python --version)${NC}\n"

# --- 3. Installazione Poetry ---
if ! command -v poetry &> /dev/null; then
    echo -e "${BLUE}[3/5] Installazione Poetry...${NC}"
    pip install poetry
    echo -e "${GREEN}✓ Poetry installato.${NC}\n"
else
    echo -e "${GREEN}[3/5] Poetry è già installato.${NC}\n"
fi

# --- 4. Dipendenze ---
echo -e "${BLUE}[4/5] Installazione/Aggiornamento dipendenze del progetto...${NC}"
poetry lock
poetry install
echo -e "${GREEN}✓ Dipendenze pronte.${NC}\n"

# --- 5. Configurazione Thread ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo -e "${GREEN}✓ Variabili d'ambiente configurate.${NC}\n"

# --- 6. Esecuzione ---
echo -e "${BLUE}[6/6] Avvio Job Hydra in parallelo...${NC}"
echo "---------------------------------------------------------"

poetry run python src/eqnn/main.py -m \
    DATA.N=320,640,1280\
    QNN.non_equivariance=0,1,2,3\
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    GENERAL.seed=42,101,1234,5678,999\
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10 &

echo -e "\n${GREEN}=== TUTTI GLI ESPERIMENTI COMPLETATI ===${NC}"

