"""Interactive ansatz designer: draw an 8-qubit circuit gate-by-gate and
train it on MNIST (digit 3 vs 4) with the same pipeline used by the fixed
architectures (src/qnn.py). Run with `streamlit run src/designer_app.py`.
"""

import datetime
import json
import os

import matplotlib.pyplot as plt
import pennylane as qml
import streamlit as st
import torch

from src.ansatz_builder import (
    architecture_to_spec,
    build_qnn_from_spec,
    check_p4m_invariance,
    is_parametric_gate,
    param_labels,
    validate_spec,
)
from src.data_encoding import embedding_unitary
from src.data_loading import load_mnist_data
from src.paper_ansatzes import paper_architecture_spec
from src.qnn import ARCHITECTURES
from src.train import train_loop

NUM_QUBITS = 8
IMG_SIZE = 16  # tied to NUM_QUBITS: embedding needs img_size == 2**(NUM_QUBITS/2)
DEVICE_NAME = "default.qubit"

GATE_CATALOG: dict[str, tuple[str, int | None]] = {
    "RX — rotazione (1 qubit)": ("RX", 1),
    "RY — rotazione (1 qubit)": ("RY", 1),
    "RZ — rotazione (1 qubit)": ("RZ", 1),
    "H — Hadamard (1 qubit, fisso)": ("H", 1),
    "X — Pauli-X (1 qubit, fisso)": ("X", 1),
    "Y — Pauli-Y (1 qubit, fisso)": ("Y", 1),
    "Z — Pauli-Z (1 qubit, fisso)": ("Z", 1),
    "S (1 qubit, fisso)": ("S", 1),
    "T (1 qubit, fisso)": ("T", 1),
    "CNOT (2 qubit, fisso)": ("CNOT", 2),
    "CZ (2 qubit, fisso)": ("CZ", 2),
    "SWAP (2 qubit, fisso)": ("SWAP", 2),
    "IsingXX — rotazione (2 qubit)": ("ISINGXX", 2),
    "IsingYY — rotazione (2 qubit)": ("ISINGYY", 2),
    "IsingZZ — rotazione (2 qubit)": ("ISINGZZ", 2),
    "Toffoli/CCNOT (3 qubit, fisso)": ("TOFFOLI", 3),
    "CSWAP (3 qubit, fisso)": ("CSWAP", 3),
    "PauliRot personalizzato (2-4 qubit)": ("PAULIROT", None),
}

READOUT_LABELS = {
    "Somma Z (generico, config1-5)": "sum_z",
    "X0 + X_{n/2} (stile config6-9)": "x0_xhalf",
}
READOUT_LABELS_INV = {v: k for k, v in READOUT_LABELS.items()}


def _dummy_embedding() -> torch.Tensor:
    img = torch.rand(IMG_SIZE, IMG_SIZE)
    return embedding_unitary(img / torch.linalg.norm(img.reshape(-1)))


def _gate_summary(gate_spec: dict) -> str:
    wires = ",".join(str(w) for w in gate_spec["wires"])
    label = gate_spec["gate"]
    if label == "PAULIROT":
        label = f"PauliRot({gate_spec['pauli_word']})"
    param = gate_spec.get("param")
    if param is None:
        return f"{label} — qubit [{wires}]"
    lock = "🔒 congelato" if param["frozen"] else "🎯 allenabile"
    value = f"={param['value']:.3f}" if param["init"] == "custom" else " (casuale)"
    group = f" — gruppo:{param['group']}" if param.get("group") else ""
    return f"{label} — qubit [{wires}] — {lock}{value}{group}"


def _final_spec() -> list[dict]:
    """The circuit actually built/trained: the current gate list repeated
    `layer_reps` times (each repetition gets its own independent
    parameters, exactly like config1-5's own `reps` — see
    src.ansatz_builder.build_qnn_from_spec)."""
    return st.session_state.spec * int(st.session_state.layer_reps)


def main() -> None:
    st.set_page_config(page_title="EQNN Ansatz Designer", layout="wide")
    st.title("🔧 Ansatz Designer — circuito a 8 qubit")
    st.caption(
        "Disegna un circuito gate-per-gate e addestralo su MNIST (cifra 3 vs 4). "
        "config1-config9 sono casi particolari di quello che puoi costruire qui."
    )

    if "spec" not in st.session_state:
        st.session_state.spec = []
    st.session_state.setdefault("twirled", False)
    st.session_state.setdefault("readout", "sum_z")
    st.session_state.setdefault("layer_reps", 1)

    left, right = st.columns([1, 1])

    with left:
        st.header("1. Costruisci il circuito")

        with st.expander("Carica un punto di partenza"):
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                preset = st.selectbox("Architettura", sorted(ARCHITECTURES))
                st.caption(
                    "config1-config5: pattern rotazione+entangler generico "
                    "(twirling impostato automaticamente se previsto). "
                    "config6-config9: ansatz D4-equivarianti/non-equivarianti "
                    "a budget fisso di parametri (vedi src/paper_ansatzes.py)."
                )
            with preset_col2:
                preset_reps = st.number_input(
                    "Ripetizioni (solo config1-5)", min_value=1, value=2, step=1
                )
                if st.button("Carica come punto di partenza"):
                    meta = ARCHITECTURES[preset]
                    if meta["kind"] == "uniform":
                        st.session_state.spec = architecture_to_spec(
                            preset, NUM_QUBITS, int(preset_reps)
                        )
                        st.session_state.readout = "sum_z"
                    else:
                        st.session_state.spec = paper_architecture_spec(
                            meta["paper_ansatz"], meta["symmetry"], NUM_QUBITS
                        )
                        st.session_state.readout = "x0_xhalf"
                    st.session_state.twirled = meta["twirled"]
                    st.session_state.layer_reps = 1
                    st.rerun()

        with st.expander("Opzioni circuito", expanded=False):
            st.session_state.twirled = st.checkbox(
                "🔄 Applica twirling p4m (media su 8 elementi del gruppo)",
                value=st.session_state.twirled,
                help=(
                    "Rende il circuito ESATTAMENTE p4m-equivariante per "
                    "costruzione, indipendentemente da cosa hai disegnato — "
                    "lo stesso meccanismo di config2/config4/config5."
                ),
            )
            readout_label = st.selectbox(
                "Schema di misura",
                list(READOUT_LABELS.keys()),
                index=list(READOUT_LABELS.values()).index(st.session_state.readout),
            )
            st.session_state.readout = READOUT_LABELS[readout_label]
            st.session_state.layer_reps = st.number_input(
                "Ripeti questo layout N volte",
                min_value=1,
                value=int(st.session_state.layer_reps),
                step=1,
                help=(
                    "Il circuito disegnato sotto viene ripetuto N volte, ognuna "
                    "con parametri allenabili indipendenti (come 'reps' in "
                    "config1-config5)."
                ),
            )

        with st.form("add_gate_form", clear_on_submit=True):
            label = st.selectbox("Tipo di gate", list(GATE_CATALOG.keys()))
            gate_name, arity = GATE_CATALOG[label]

            pauli_word = None
            if gate_name == "PAULIROT":
                pauli_word = st.text_input(
                    "Stringa di Pauli (2-4 lettere tra I/X/Y/Z, es. XY, YYYY)",
                    value="XY",
                ).upper()
                arity = len(pauli_word) if pauli_word else None

            wires = st.multiselect(
                (
                    f"Qubit coinvolti (scegline esattamente {arity})"
                    if arity
                    else "Qubit coinvolti"
                ),
                options=list(range(NUM_QUBITS)),
            )

            init_mode, custom_value, frozen = "Casuale", 0.0, False
            if is_parametric_gate(gate_name):
                init_mode = st.radio(
                    "Inizializzazione parametro",
                    ["Casuale", "Personalizzata"],
                    horizontal=True,
                )
                if init_mode == "Personalizzata":
                    custom_value = st.number_input(
                        "Valore iniziale (radianti)", value=0.0, format="%.4f"
                    )
                frozen = st.checkbox("Congela questo parametro (non verrà allenato)")

            submitted = st.form_submit_button("➕ Aggiungi gate")

        if submitted:
            gate_spec: dict = {"gate": gate_name, "wires": sorted(wires)}
            if gate_name == "PAULIROT":
                gate_spec["pauli_word"] = pauli_word
            if is_parametric_gate(gate_name):
                gate_spec["param"] = {
                    "init": "custom" if init_mode == "Personalizzata" else "random",
                    "value": custom_value if init_mode == "Personalizzata" else None,
                    "frozen": frozen,
                }
            try:
                validate_spec(st.session_state.spec + [gate_spec], NUM_QUBITS)
                st.session_state.spec.append(gate_spec)
                st.success(f"Aggiunto: {_gate_summary(gate_spec)}")
            except ValueError as exc:
                st.error(str(exc))

        st.subheader("Circuito attuale (un layout)")
        if not st.session_state.spec:
            st.info("Nessun gate ancora — aggiungine uno sopra, o carica un preset.")
        else:
            for idx, gate_spec in enumerate(st.session_state.spec):
                cols = st.columns([8, 1])
                cols[0].write(f"`#{idx}` {_gate_summary(gate_spec)}")
                if cols[1].button("🗑️", key=f"del_{idx}"):
                    st.session_state.spec.pop(idx)
                    st.rerun()

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🧹 Svuota circuito"):
                    st.session_state.spec = []
                    st.rerun()
            with col_b:
                st.download_button(
                    "💾 Scarica spec (JSON)",
                    data=json.dumps(st.session_state.spec, indent=2),
                    file_name="ansatz_spec.json",
                    mime="application/json",
                )

        uploaded = st.file_uploader("📂 Carica spec (JSON)", type="json")
        if uploaded is not None:
            try:
                loaded_spec = json.load(uploaded)
                validate_spec(loaded_spec, NUM_QUBITS)
                st.session_state.spec = loaded_spec
                st.success("Spec caricata.")
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Spec non valida: {exc}")

    with right:
        st.header("Anteprima circuito")
        if st.session_state.spec:
            try:
                final_spec = _final_spec()
                qnn, initial_params, _ = build_qnn_from_spec(
                    DEVICE_NAME,
                    NUM_QUBITS,
                    0.0,
                    final_spec,
                    twirled=st.session_state.twirled,
                    readout=st.session_state.readout,
                )
                fig, _ = qml.draw_mpl(qnn.qnode, show_all_wires=True)(
                    _dummy_embedding(), initial_params, torch.tensor(0.0)
                )
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"Parametri allenabili: {len(initial_params)} — "
                    f"twirling: {'sì' if st.session_state.twirled else 'no'} — "
                    f"misura: {READOUT_LABELS_INV[st.session_state.readout]}"
                )

                if st.button("🔍 Verifica invarianza p4m"):
                    with st.spinner(
                        "Valutazione del circuito su alcune immagini "
                        "ribaltate/trasposte (qualche decina di secondi)..."
                    ):
                        is_invariant, deviation = check_p4m_invariance(
                            qnn, initial_params, IMG_SIZE
                        )
                    if is_invariant:
                        st.success(
                            f"✅ p4m-invariante (deviazione massima osservata: "
                            f"{deviation:.2e})"
                        )
                    else:
                        st.warning(
                            f"❌ NON p4m-invariante (deviazione massima osservata: "
                            f"{deviation:.2e})"
                        )
                    st.caption(
                        "Verifica numerica: l'uscita non deve cambiare se "
                        "l'immagine in ingresso viene ribaltata o trasposta. Non "
                        "è una prova formale, ma è affidabile su questo codice "
                        "(vedi tests/test_equivariance.py)."
                    )
            except Exception as exc:
                st.error(f"Impossibile disegnare il circuito: {exc}")

        st.header("2. Configura il training")
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Numero immagini (N)", min_value=10, value=100, step=10)
            epochs = st.number_input("Epoche", min_value=1, value=40)
            seed = st.number_input("Seed", value=1234, step=1)
            p_err = st.slider("Rumore depolarizzante (p_err)", 0.0, 0.5, 0.0, step=0.01)
        with col2:
            learning_rate = st.number_input("Learning rate", value=0.05, format="%.4f")
            patience = st.number_input("Patience", min_value=1, value=5)
            min_delta = st.number_input("Min delta", value=0.0001, format="%.5f")
            wandb_group = st.text_input("Gruppo wandb (opzionale)")

        can_train = bool(st.session_state.spec)
        if st.button("🚀 Avvia training", type="primary", disabled=not can_train):
            if wandb_group:
                os.environ["WANDB_RUN_GROUP"] = wandb_group

            run_dir = os.path.join(
                "outputs",
                "designer",
                datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S"),
            )
            os.makedirs(run_dir, exist_ok=True)

            with st.spinner(
                "Training in corso... segui i log sul terminale o su wandb "
                "per l'avanzamento in tempo reale."
            ):
                try:
                    batch_size = max(1, int(N) // 10)
                    train_loader, test_loader = load_mnist_data(
                        batch_size, int(N), 0, IMG_SIZE, "data", int(seed), False, False
                    )
                    _, aug_test_loader = load_mnist_data(
                        batch_size, int(N), 0, IMG_SIZE, "data", int(seed), False, True
                    )

                    twirled = st.session_state.twirled
                    readout = st.session_state.readout
                    qnn, initial_params, resolved_spec = build_qnn_from_spec(
                        DEVICE_NAME,
                        NUM_QUBITS,
                        p_err,
                        _final_spec(),
                        twirled=twirled,
                        readout=readout,
                    )
                    names = param_labels(resolved_spec)

                    old_cwd = os.getcwd()
                    os.chdir(run_dir)
                    try:
                        result = train_loop(
                            train_loader,
                            test_loader,
                            aug_test_loader,
                            epochs=int(epochs),
                            learning_rate=learning_rate,
                            patience=int(patience),
                            min_delta=min_delta,
                            dev="cpu",
                            seed=int(seed),
                            N=int(N),
                            dataset="mnist",
                            qnn=qnn,
                            initial_params=initial_params,
                            param_names=names,
                            run_name=f"custom_N={int(N)}_seed={int(seed)}",
                            checkpoint_config={
                                "device": DEVICE_NAME,
                                "num_qubits": NUM_QUBITS,
                                "p_err": p_err,
                                "circuit_spec": resolved_spec,
                                "twirled": twirled,
                                "readout": readout,
                                "img_size": IMG_SIZE,
                            },
                            wandb_extra_config={
                                "architecture": "custom",
                                "circuit_spec": resolved_spec,
                                "twirled": twirled,
                                "readout": readout,
                            },
                            verbose=False,
                        )
                    finally:
                        os.chdir(old_cwd)

                    _, _, train_loss_hist, train_acc_hist, _, val_acc, *_ = result
                    st.success(
                        f"Training completato — accuratezza di validazione: {val_acc[0]:.3f}"
                    )
                    loss_plot = os.path.join(run_dir, "loss_history.jpg")
                    if os.path.exists(loss_plot):
                        st.image(loss_plot, caption="Andamento della loss")
                    st.caption(f"Output salvato in: {run_dir}")
                except Exception as exc:
                    st.error(f"Errore durante il training: {exc}")


if __name__ == "__main__":
    main()
