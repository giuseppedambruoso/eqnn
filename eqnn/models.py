"""The four models of the paper (paper: Methods, "The four models") and their training objective.

    id        name               circuit                     model output m(x)
    config6   Equiv              equivariant U(theta)        <psi(x)| U^dag O U |psi(x)>,  O = (X_0 + X_4)/2
    config7   NonEquiv           non-equivariant U'(theta)   <psi(x)| U'^dag O U' |psi(x)>
    config10  NonEquiv-Twirled   non-equivariant U'(theta)   mean over g in D4 of the NonEquiv output on g x
    config6n  Equiv-NoSym        equivariant U(theta)        <psi(x)| U^dag O_ns U |psi(x)>,  O_ns = (X_0 + Z_0 + Z_4)/(1 + sqrt 2)

The ids are those stored in the results (field "arch"). Equiv and NonEquiv-Twirled are exactly D4-invariant for every
value of the angles; NonEquiv and Equiv-NoSym are not. The prediction is the raw output: class 1 if m(x) > 0, with no
classical post-processing. Training objective: the logistic loss of BETA * t * m, t = +-1.

Simulation: without noise, exact statevector simulation (PennyLane default.qubit, backpropagation, double precision);
with the realistic noise of eqnn.noise, the exact Heisenberg-picture simulation of that module.
"""

import math
import random

import pennylane as qml
import torch

from eqnn.circuits import ANGLES_PER_LAYER, HALF, NUM_QUBITS, circuit_gates
from eqnn.data import SIDE, apply_d4

MODELS = {
    "config6": {"name": "Equiv", "equivariant": True, "twirl": False, "readout": "O"},
    "config7": {"name": "NonEquiv", "equivariant": False, "twirl": False, "readout": "O"},
    "config10": {"name": "NonEquiv-Twirled", "equivariant": False, "twirl": True, "readout": "O"},
    "config6n": {"name": "Equiv-NoSym", "equivariant": True, "twirl": False, "readout": "O_ns"},
}
NAMES = {v["name"]: k for k, v in MODELS.items()}  # Equiv -> config6, ...
BETA = 20.0  # scale of the logistic loss (the outputs are bounded in [-1, 1])
INIT_RANGE = 0.1  # initial angles uniform in [-INIT_RANGE, INIT_RANGE]


def model_id(arch: str) -> str:
    """Accepts an id (config6) or a name (Equiv)."""
    return arch if arch in MODELS else NAMES[arch]


def readout(name: str):
    """The measured observable: O = (X_0 + X_4)/2, invariant under D4; O_ns = (X_0 + Z_0 + Z_4)/(1 + sqrt 2), mapped
    into itself by no element of D4 other than the identity (t_x flips Z_0, t_y flips Z_4, the swap maps X_0 to X_4);
    both have norm 1, so that m(x) is in [-1, 1]."""
    if name == "O":
        return 0.5 * (qml.X(0) + qml.X(HALF))
    if name == "O_ns":
        return (qml.X(0) + qml.Z(0) + qml.Z(HALF)) * (1.0 / (1.0 + math.sqrt(2.0)))
    raise ValueError(name)


def d4_pixel_perms() -> torch.Tensor:
    """The eight permutations of the amplitudes (pixels) induced by the elements of D4, (8, 256): the transformed
    image is x[perms[g]] (eqnn.data.apply_d4)."""
    idx = torch.arange(SIDE * SIDE).reshape(SIDE, SIDE)
    return torch.stack([apply_d4(idx, g).reshape(-1) for g in range(8)])


def twirl_matrix(a: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """Quadratic form of the twirled model: x^T A_tw x = mean_g (g x)^T A (g x), A_tw = mean_g A[p_g][:, p_g]. The twirl
    is a classical, noiseless average over the transformed inputs."""
    return torch.stack([a[p][:, p] for p in perms]).mean(dim=0)


def build_model(arch: str, layers: int, noise_scale: float | None = None, chip_seed: int = 0,
                chip_spread: bool = True, engine: str = "heisenberg"):
    """Returns (forward, n_params); forward(states (batch, 256), params (n_params,)) -> outputs (batch,).

    noise_scale None: noiseless statevector simulation. Otherwise the realistic single-qubit noise of eqnn.noise,
    with every error rate multiplied by noise_scale and the per-qubit parameters of the chip drawn with chip_seed
    (around the typical medians; identical qubits if chip_spread is False), simulated exactly in the Heisenberg
    picture (engine "heisenberg") or with density matrices (engine "mixed", the reference used in the tests)."""
    m = MODELS[model_id(arch)]
    gates = circuit_gates(m["equivariant"], layers)
    n_params = ANGLES_PER_LAYER * layers
    if noise_scale is not None:
        from eqnn import noise
        terms = noise.READOUT_NOSYM if m["readout"] == "O_ns" else noise.READOUT_O
        return noise.build(gates, NUM_QUBITS, noise_scale, chip_seed, chip_spread, m["twirl"], engine, terms), n_params

    observable = readout(m["readout"])
    dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(states, params):
        qml.StatePrep(states, wires=range(NUM_QUBITS), normalize=True)
        for word, k in gates:
            wires = list(word)
            if len(wires) == 1:
                {"X": qml.RX, "Y": qml.RY}[word[wires[0]]](params[k], wires=wires)
            else:
                qml.PauliRot(params[k], "".join(word[w] for w in wires), wires=wires)
        return [qml.expval(observable)]

    def one(states, params):
        out = qnode(states, params)
        return torch.stack(list(out), dim=-1) if isinstance(out, (tuple, list)) else out

    def forward(states, params):
        if m["twirl"]:
            # the eight transformed copies of the images go through the circuit as one batch of 8B states
            b = states.shape[0]
            imgs = states.reshape(b, SIDE, SIDE)
            xs = torch.cat([apply_d4(imgs, g).reshape(b, -1) for g in range(8)])
            res = one(xs, params)
            res = res.reshape(8, b, *res.shape[1:]).mean(dim=0)
        else:
            res = one(states, params)
        return res.squeeze(-1) if res.shape[-1] == 1 else res

    return forward, n_params


def init_params(n_params: int, seed: int) -> torch.Tensor:
    """Initial angles, uniform in [-0.1, 0.1] (circuits close to the identity)."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(n_params, dtype=torch.float64).uniform_(-INIT_RANGE, INIT_RANGE, generator=g)


def model_loss(m: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Logistic loss of BETA * t * m, t = 2 y - 1 (paper: Methods, "Readout and loss")."""
    t = 2.0 * labels.to(torch.float64) - 1.0
    return torch.nn.functional.softplus(-BETA * t * m).mean()


def predict(m: torch.Tensor) -> torch.Tensor:
    return (m > 0).long()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
