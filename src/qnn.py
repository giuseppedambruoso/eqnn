import logging
import random
from typing import Any

import pennylane as qml
import torch

from src.data_encoding import as_state_vector
from src.noise import apply_gate_noise, make_noise_rng

logger = logging.getLogger(__name__)

# --- Symmetries p4m (D4) ---


def V_x(
    num_qubits: int, noise_rng: random.Random | None = None, noise_p: float = 0.0
) -> None:
    half = num_qubits // 2
    for i in range(half):
        qml.X(wires=i)
        apply_gate_noise([i], noise_rng, noise_p)


def V_y(
    num_qubits: int, noise_rng: random.Random | None = None, noise_p: float = 0.0
) -> None:
    half = num_qubits // 2
    for i in range(half, num_qubits):
        qml.X(wires=i)
        apply_gate_noise([i], noise_rng, noise_p)


def apply_group_element(
    g_idx: int,
    num_qubits: int,
    noise_rng: random.Random | None = None,
    noise_p: float = 0.0,
) -> None:
    """Applies one of the 8 p4m group elements for explicit twirling.
    noise_rng/noise_p: see src.noise's module docstring — applied after
    every X/SWAP below, same as any other gate in the ansatz."""
    half = num_qubits // 2

    def _swap_pairs() -> None:
        for i in range(half):
            qml.SWAP(wires=[i, i + half])
            apply_gate_noise([i, i + half], noise_rng, noise_p)

    if g_idx == 0:
        pass  # Identity
    elif g_idx == 1:
        V_x(num_qubits, noise_rng, noise_p)  # Reflection X
    elif g_idx == 2:
        V_y(num_qubits, noise_rng, noise_p)  # Reflection Y
    elif g_idx == 3:  # Reflection XY (180 Rotation)
        V_x(num_qubits, noise_rng, noise_p)
        V_y(num_qubits, noise_rng, noise_p)
    elif g_idx == 4:  # Transpose (x-y swap)
        _swap_pairs()
    elif g_idx == 5:  # 90 Rotation
        V_x(num_qubits, noise_rng, noise_p)
        _swap_pairs()
    elif g_idx == 6:  # -90 Rotation
        V_y(num_qubits, noise_rng, noise_p)
        _swap_pairs()
    elif g_idx == 7:  # Anti-diagonal reflection
        V_x(num_qubits, noise_rng, noise_p)
        V_y(num_qubits, noise_rng, noise_p)
        _swap_pairs()


def equiv_measure(num_qubits: int) -> None:
    for i in range(num_qubits):
        qml.H(wires=i)


# --- QNode Factory ---

# The three supported architectures (see src.paper_ansatzes): the same
# 5-block schedule with 6 tied trainable angles per layer.
#   config6:  generator-equivariant (Equiv) - p4m-equivariant by construction.
#   config7:  axis-scrambled counterpart (NonEquiv) - not equivariant.
#   config10: config7 wrapped in explicit p4m twirling (NonEquiv-Twirled):
#             output averaged over the 8 group elements, exactly invariant,
#             at 8x the circuit evaluations.
# "twirled" is the MECHANISM flag, "is_equivariant" the resulting PROPERTY.
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "config6": {
        "paper_ansatz": "6",
        "symmetry": "equivariant",
        "twirled": False,
        "is_equivariant": True,
    },
    "config7": {
        "paper_ansatz": "6",
        "symmetry": "nonequivariant",
        "twirled": False,
        "is_equivariant": False,
    },
    "config10": {
        "paper_ansatz": "6",
        "symmetry": "nonequivariant",
        "twirled": True,
        "is_equivariant": True,
    },
}


def _check_architecture(architecture: str) -> dict[str, Any]:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}"
        )
    return ARCHITECTURES[architecture]


OUTPUT_BIAS_PARAM_NAMES = ("out_scale", "out_bias")


def _with_output_bias(qnn_forward: Any) -> Any:
    """Classical affine post-processing of the measured expectation value:
    returns tanh(w <O> + b) with (w, b) = the last two entries of `params`,
    so that train.execute_batch's (1 + output) / 2 equals
    sigmoid(2 (w <O> + b)). Without it the decision threshold is pinned at
    <O> = 0, and a dataset whose <O> has the same sign for both classes
    cannot be classified at all. It acts only on the (already invariant)
    scalar output, so it never affects equivariance."""

    def forward(encoded: torch.Tensor, params: torch.Tensor) -> Any:
        raw = qnn_forward(encoded, params[:-2])
        return torch.tanh(params[-2] * raw + params[-1])

    forward.qnode = getattr(qnn_forward, "qnode", None)  # type: ignore[attr-defined]
    return forward


def initial_parameters(
    param_names: list[str], generator: torch.Generator, device: str = "cpu"
) -> torch.Tensor:
    """Uniform(-0.1, 0.1) initialization for every circuit angle; the
    output-bias parameters (if present) start at w = 1, b = 0, i.e. the
    model initially coincides with the one without output bias."""
    params = torch.empty(len(param_names), device=torch.device(device)).uniform_(
        -0.1, 0.1, generator=generator
    )
    for i, name in enumerate(param_names):
        if name == "out_scale":
            params[i] = 1.0
        elif name == "out_bias":
            params[i] = 0.0
    return params


def architecture_param_names(
    architecture: str, num_qubits: int, reps: int = 1, output_bias: bool = False,
    layers: int = 1,
) -> list[str]:
    """Names of the trainable parameters create_qnn(..., architecture)
    expects: 6 tied circuit angles per layer, plus (w, b) if output_bias.
    reps is accepted for backward compatibility and ignored; layers
    stacks copies of the circuit, each with its own angles."""
    # Local import: src.ansatz_builder imports from this module.
    from src.ansatz_builder import param_labels
    from src.paper_ansatzes import paper_architecture_spec

    spec = _check_architecture(architecture)
    gate_spec = paper_architecture_spec(spec["paper_ansatz"], spec["symmetry"], num_qubits, layers)
    extra = list(OUTPUT_BIAS_PARAM_NAMES) if output_bias else []
    return param_labels(gate_spec) + extra


def create_qnn(
    device: str,
    num_qubits: int,
    reps: int = 1,
    architecture: str = "config6",
    diff_method: str = "backprop",
    readout: str | None = None,
    noise_p: float = 0.0,
    noise_seed: int = 0,
    output_bias: bool = False,
    layers: int = 1,
) -> Any:
    """Builds config6 / config7 / config10 (see ARCHITECTURES) as a
    callable qnn(encoded_states, params).

    diff_method: "backprop" (default) is fast in simulation - src.train's
    execute_batch relies on it to run a whole batch in one vectorized call;
    "parameter-shift" is much slower but also works on real hardware.
    readout: "avg_x" (default, mean of X over every qubit) or "x0_xhalf"
    (0.5 (X_0 + X_{num_qubits/2})); both are p4m-invariant.
    noise_p/noise_seed: Monte Carlo single-qubit depolarizing noise after
    every gate (see src.noise); 0.0 disables it.
    output_bias: append the trainable affine output map (w, b).
    layers: stacked copies of the circuit, each with its own angles.
    reps is accepted for backward compatibility and ignored.
    """
    from src.ansatz_builder import build_qnn_from_spec
    from src.paper_ansatzes import paper_architecture_spec

    spec = _check_architecture(architecture)
    gate_spec = paper_architecture_spec(spec["paper_ansatz"], spec["symmetry"], num_qubits, layers)
    qnn_forward, _, _ = build_qnn_from_spec(
        device,
        num_qubits,
        gate_spec,
        twirled=spec["twirled"],
        readout=readout or "avg_x",
        diff_method=diff_method,
        noise_p=noise_p,
        noise_seed=noise_seed,
    )
    return _with_output_bias(qnn_forward) if output_bias else qnn_forward
