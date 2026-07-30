#!/bin/bash
# Interactive wizard around `docker compose run eqnn` + Hydra overrides.
set -e

echo "=== EQNN — configurazione run interattiva ==="
echo "(Per confrontare più valori insieme, scrivi una lista separata da"
echo " virgole senza spazi, es: True,False — la modalità sweep viene"
echo " attivata automaticamente quando serve.)"
echo

read -rp "Equivarianza [True]: " EQ
EQ=${EQ:-True}
read -rp "Twirling [False]: " TWIRL
TWIRL=${TWIRL:-False}
read -rp "Rimuovi cross-edge [True]: " CROSS
CROSS=${CROSS:-True}
read -rp "Gate di rotazione (RY/RX) [RY]: " ROTGATE
ROTGATE=${ROTGATE:-RY}
read -rp "Entangler (cnot/frozen_ryy) [cnot]: " ENTANGLER
ENTANGLER=${ENTANGLER:-cnot}
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

# Sweep mode is detected automatically: if any value contains a comma, this
# has to be a Hydra multirun (-m), otherwise Hydra rejects it as ambiguous.
MFLAG=""
for VALUE in "$EQ" "$TWIRL" "$CROSS" "$ROTGATE" "$ENTANGLER" "$REPS" "$N" "$DATASET" "$EPOCHS" "$SEED" "$DEV"; do
    if [[ "$VALUE" == *,* ]]; then
        MFLAG="-m"
        break
    fi
done
if [ -n "$MFLAG" ]; then
    echo
    echo "🔀 Sweep rilevato (valore con virgola) — aggiungo --multirun."
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
    "QNN.rotation_gate=$ROTGATE"
    "QNN.entangler=$ENTANGLER"
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
