from streamlit.testing.v1 import AppTest

APP_PATH = "src/designer_app.py"
TIMEOUT = 60  # first run imports torch/pennylane/matplotlib, slower than default 3s

# selectbox order: [0] architecture preset, [1] readout scheme, [2] gate type
GATE_TYPE_SELECTBOX = 2


def test_app_loads_without_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Nessun gate ancora" in i.value for i in at.info)


def test_load_config1_preset_populates_circuit():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[0].select("config1")  # architecture preset selector
    load_button = next(b for b in at.button if "Carica come punto" in b.label)
    load_button.click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    # config1, reps=2 (default), 8 qubits: 2 * (8 RY + 7 CNOT) = 30 gates
    assert len(at.markdown) == 30
    assert any("Parametri allenabili: 16" in c.value for c in at.caption)
    assert any("twirling: no" in c.value for c in at.caption)


def test_load_config2_preset_enables_twirling():
    """config2 = twirled config1 — loading it must auto-check the twirling
    toggle (see ARCHITECTURES[architecture]["twirled"] in src/qnn.py)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[0].select("config2")
    next(b for b in at.button if "Carica come punto" in b.label).click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("twirling: sì" in c.value for c in at.caption)


def test_load_config6_preset_uses_x0_xhalf_readout():
    """config6 (paper6, equivariant) needs the x0_xhalf readout to preserve
    its exact p4m-equivariance — auto-set when the preset is loaded."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[0].select("config6")
    next(b for b in at.button if "Carica come punto" in b.label).click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Parametri allenabili: 6" in c.value for c in at.caption)
    assert any("X0 + X_" in c.value for c in at.caption)


def test_layer_repetition_multiplies_gate_count():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[GATE_TYPE_SELECTBOX].select("RX — rotazione (1 qubit)")
    at.multiselect[0].select(0)
    next(b for b in at.button if "Aggiungi gate" in b.label).click()
    at.run(timeout=TIMEOUT)
    assert len(at.markdown) == 1

    reps_input = next(
        n for n in at.number_input if "Ripeti questo layout" in (n.label or "")
    )
    reps_input.set_value(3)
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Parametri allenabili: 3" in c.value for c in at.caption)


def test_add_gate_via_form():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[GATE_TYPE_SELECTBOX].select("RX — rotazione (1 qubit)")
    at.multiselect[0].select(0)
    add_button = next(b for b in at.button if "Aggiungi gate" in b.label)
    add_button.click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Aggiunto: RX" in s.value for s in at.success)
    assert any(m.value.startswith("`#0` RX") for m in at.markdown)


def test_add_gate_with_wrong_arity_shows_error():
    """CNOT needs exactly 2 wires — selecting just one must surface a
    validation error, not silently add a broken gate."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[GATE_TYPE_SELECTBOX].select("CNOT (2 qubit, fisso)")
    at.multiselect[0].select(0)  # only one wire, CNOT needs two
    add_button = next(b for b in at.button if "Aggiungi gate" in b.label)
    add_button.click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("expected 2 wire" in e.value for e in at.error)
    assert any("Nessun gate ancora" in i.value for i in at.info)


def test_remove_gate_button():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[GATE_TYPE_SELECTBOX].select("RX — rotazione (1 qubit)")
    at.multiselect[0].select(0)
    next(b for b in at.button if "Aggiungi gate" in b.label).click()
    at.run(timeout=TIMEOUT)
    assert len(at.markdown) == 1

    next(b for b in at.button if b.label == "🗑️").click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Nessun gate ancora" in i.value for i in at.info)


def test_p4m_invariance_button_reports_result():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[0].select("config6")  # exactly p4m-invariant by construction
    next(b for b in at.button if "Carica come punto" in b.label).click()
    at.run(timeout=TIMEOUT)

    next(b for b in at.button if "Verifica invarianza" in b.label).click()
    at.run(timeout=TIMEOUT * 2)

    assert not at.exception
    assert any("p4m-invariante" in s.value for s in at.success)
