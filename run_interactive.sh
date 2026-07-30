#!/bin/bash
# Interactive wizard around `docker compose run eqnn` + Hydra overrides.
set -e

echo "=== EQNN — configurazione run interattiva ==="
echo "(Per confrontare più valori insieme, scrivi una lista separata da"
echo " virgole senza spazi, es: True,False — la modalità sweep viene"
echo " attivata automaticamente quando serve.)"
echo

echo "Architetture disponibili:"
echo "  config1: RY + CNOT                 (non equivariante)"
echo "  config2: RY + CNOT + twirling       (equivariante)"
echo "  config3: RX + RXY                  (non equivariante)"
echo "  config4: RX + RXY + twirling        (equivariante)"
echo "  config5: RX + RYY/RYYYY + twirling  (equivariante)"
echo
read -rp "Architettura (config1-config5) [config1]: " ARCHITECTURE
ARCHITECTURE=${ARCHITECTURE:-config1}
read -rp "Ripetizioni circuito (reps) [2]: " REPS
REPS=${REPS:-2}
read -rp "Numero immagini training (N) [80]: " N
N=${N:-80}
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
for VALUE in "$ARCHITECTURE" "$REPS" "$N" "$DATASET" "$EPOCHS" "$SEED" "$DEV"; do
    if [[ "$VALUE" == *,* ]]; then
        MFLAG="-m"
        break
    fi
done
JOBLIB_ARGS=()
if [ -n "$MFLAG" ]; then
    echo
    echo "🔀 Sweep rilevato (valore con virgola) — aggiungo --multirun."
    echo

    CORES=$(nproc 2>/dev/null || echo 4)
    RECOMMENDED=$((CORES > 1 ? CORES - 1 : 1))
    echo "Il tuo computer ha $CORES core disponibili."
    echo "Ogni job usa un intero core per tutta la sua durata: non ha senso mettere"
    echo "K oltre $CORES (i job in più aspetterebbero comunque in coda), e usarli"
    echo "tutti rallenta ogni altra cosa sul computer nel frattempo."
    echo
    read -rp "Quanti job in parallelo (K)? [$RECOMMENDED]: " KJOBS
    KJOBS=${KJOBS:-$RECOMMENDED}
    if ! [[ "$KJOBS" =~ ^[0-9]+$ ]] || [ "$KJOBS" -lt 1 ]; then
        echo "Valore non valido, uso $RECOMMENDED."
        KJOBS=$RECOMMENDED
    elif [ "$KJOBS" -gt "$CORES" ]; then
        echo "⚠️  K=$KJOBS supera i $CORES core disponibili: i job oltre $CORES si"
        echo "    metteranno comunque in coda, senza guadagno di velocità."
    fi
    JOBLIB_ARGS=("hydra/launcher=joblib" "hydra.launcher.n_jobs=$KJOBS")
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
    "QNN.architecture=$ARCHITECTURE"
    "QNN.reps=$REPS"
    "TRAINING.epochs=$EPOCHS"
)
if [ "${#JOBLIB_ARGS[@]}" -gt 0 ]; then
    CMD+=("${JOBLIB_ARGS[@]}")
fi

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
