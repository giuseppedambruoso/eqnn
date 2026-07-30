#!/bin/bash
# Interactive wizard around `docker compose run eqnn` + Hydra overrides.
set -e

echo "=== EQNN — configurazione run interattiva ==="
echo

read -rp "Modalità sweep, più configurazioni insieme? [y/N]: " SWEEP
SWEEP=${SWEEP:-n}
if [[ "$SWEEP" =~ ^[Yy] ]]; then
    echo "(In modalità sweep puoi inserire più valori separati da virgola, es: True,False)"
    echo
fi

read -rp "Equivarianza [True]: " EQ
EQ=${EQ:-True}
read -rp "Twirling [False]: " TWIRL
TWIRL=${TWIRL:-False}
read -rp "Rimuovi cross-edge [True]: " CROSS
CROSS=${CROSS:-True}
read -rp "Ripetizioni circuito (reps) [2]: " REPS
REPS=${REPS:-2}
read -rp "Numero immagini training (N) [20]: " N
N=${N:-20}
read -rp "Dataset [mnist]: " DATASET
DATASET=${DATASET:-mnist}
read -rp "Epoche [80]: " EPOCHS
EPOCHS=${EPOCHS:-80}
read -rp "Seed [1234]: " SEED
SEED=${SEED:-1234}
read -rp "Device (cpu/cuda) [cpu]: " DEV
DEV=${DEV:-cpu}
read -rp "Nome gruppo wandb per raggruppare i run (opzionale): " GROUP

MFLAG=""
if [[ "$SWEEP" =~ ^[Yy] ]]; then
    MFLAG="-m"
fi

CMD=(docker compose run --rm)
if [ -n "$GROUP" ]; then
    CMD+=(-e "WANDB_RUN_GROUP=$GROUP")
fi
CMD+=(eqnn)
if [ -n "$MFLAG" ]; then
    CMD+=("$MFLAG")
fi
CMD+=(
    "GENERAL.seed=$SEED"
    "GENERAL.dev=$DEV"
    "DATA.N=$N"
    "DATA.dataset=$DATASET"
    "QNN.equivariance=$EQ"
    "QNN.twirling=$TWIRL"
    "QNN.remove_cross_edge=$CROSS"
    "QNN.reps=$REPS"
    "TRAINING.epochs=$EPOCHS"
)

echo
echo "Comando che sto per eseguire:"
printf '%q ' "${CMD[@]}"
echo
echo

read -rp "Confermi? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-y}
if [[ "$CONFIRM" =~ ^[Yy] ]]; then
    "${CMD[@]}"
else
    echo "Annullato."
fi
