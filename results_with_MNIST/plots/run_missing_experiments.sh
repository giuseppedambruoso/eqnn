#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
echo -e "${BLUE}=== Avvio Recupero Esperimenti Mancanti ===${NC}\n"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate eqnn_env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo -e "${BLUE}Esecuzione Job Hydra...${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=20 \
    GENERAL.seed=1234 \
    QNN.non_equivariance=0 \
    QNN.p_err=0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=20 NE=0${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=20 \
    GENERAL.seed=1234 \
    QNN.non_equivariance=4 \
    QNN.p_err=0.02 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=20 NE=4${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=40 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=4 \
    QNN.p_err=0.0 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=40 NE=4${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=80 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=4 \
    QNN.p_err=0.0 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=80 NE=4${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=160 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=4 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=160 NE=4${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=320 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=0 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=320 NE=0${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=320 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=1 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=320 NE=1${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=320 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=2 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=320 NE=2${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=320 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=3 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=320 NE=3${NC}"

poetry run python src/eqnn/main.py -m \
    DATA.N=320 \
    GENERAL.seed=42,104,999,1234,5678 \
    QNN.non_equivariance=4 \
    QNN.p_err=0.0,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=10

echo -e "${GREEN}✓ Completato blocco N=320 NE=4${NC}"

echo -e "\n${GREEN}=== TUTTI GLI ESPERIMENTI COMPLETATI ===${NC}"