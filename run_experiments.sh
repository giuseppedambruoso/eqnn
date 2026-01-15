#!/bin/bash

# 1. Export environment variables for thread control
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# 2. Run the Hydra sweep using 'poetry run'
# This ensures it runs inside the environment without hanging on 'poetry shell'
poetry run python src/eqnn/main.py -m \
    QNN.non_equivariance=0,1,2,3,4 \
    QNN.p_err=0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=2
