from streamlit.testing.v1 import AppTest

APP_PATH = "src/designer_app.py"
TIMEOUT = 60  # first run imports torch/pennylane/matplotlib, slower than default 3s


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


def test_add_gate_via_form():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    at.selectbox[1].select("RX — rotazione (1 qubit)")  # gate-type selector
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

    at.selectbox[1].select("CNOT (2 qubit, fisso)")
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

    at.selectbox[1].select("RX — rotazione (1 qubit)")
    at.multiselect[0].select(0)
    next(b for b in at.button if "Aggiungi gate" in b.label).click()
    at.run(timeout=TIMEOUT)
    assert len(at.markdown) == 1

    next(b for b in at.button if b.label == "🗑️").click()
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert any("Nessun gate ancora" in i.value for i in at.info)
