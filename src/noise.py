"""Monte Carlo depolarizing noise: a single, reproducible noise
*realization* inserted at circuit-build time, not the analytic
qml.DepolarizingChannel (which needs default.mixed / density-matrix
simulation and would break this project's fast, batched,
diff_method="backprop" statevector pipeline everywhere it's used).

At every gate application, independently for each wire that gate
touches, a fresh coin flip (probability `noise_p`) decides whether that
site suffers an error; if it does, one of X/Y/Z is applied (chosen
uniformly) — the standard Pauli-twirl decomposition of a depolarizing
channel, sampled via Monte Carlo instead of solved analytically.

The draws come from a plain `random.Random(noise_seed)` instance created
fresh at the top of the QNode function body (see src.qnn.create_qnn and
src.ansatz_builder.build_qnn_from_spec) — NOT from Python's global
`random` module, so this never interferes with unrelated randomness
elsewhere (D4Augmentation, data sampling, ...). Because a QNode's Python
function body re-runs on every call, and the circuit's gate structure
never depends on the input data, re-seeding from the same `noise_seed`
every call deterministically reproduces the exact same noise pattern
every time — a fixed, named "noise realization", exactly like a fixed
parameter initialization is a fixed "parameter realization". `noise_seed`
must be independent from the parameter-initialization seed so the two
can be varied separately.
"""

import random

import pennylane as qml

PAULI_OPS = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def make_noise_rng(noise_seed: int, noise_p: float) -> random.Random | None:
    """None (rather than an unused Random instance) whenever noise_p<=0,
    so apply_gate_noise can skip straight past a plain identity check —
    no RNG state exists to accidentally advance or depend on."""
    if noise_p <= 0.0:
        return None
    return random.Random(noise_seed)


def apply_gate_noise(
    wires: list[int] | range, noise_rng: random.Random | None, noise_p: float
) -> None:
    """Call immediately after applying a gate to `wires`. No-op if
    noise_rng is None (i.e. noise_p<=0 at build time)."""
    if noise_rng is None:
        return
    for w in wires:
        if noise_rng.random() < noise_p:
            axis = noise_rng.choice(("X", "Y", "Z"))
            PAULI_OPS[axis](wires=w)
