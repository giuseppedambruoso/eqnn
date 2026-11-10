"""Realistic single-qubit noise (paper: Methods, "Hardware noise"), with the typical parameters of current
superconducting processors (paper: table of the noise parameters).

The noise acts on the gates of the circuit as they are (rotations exp(-i theta/2 P) on one qubit or on four qubits),
without compiling them into native gates, and it is made of single-qubit channels only. After every gate, each
qubit of the gate undergoes
  * thermal relaxation for the duration of the gate: amplitude damping with 1 - exp(-t/T1), then dephasing with the
    pure dephasing time T_phi, 1/T2 = 1/(2 T1) + 1/T_phi;
  * a single-qubit depolarizing channel whose strength brings the average infidelity of the whole noise on that
    qubit to the error of the gate (as Qiskit Aer does for the device noise models).
The gates are scheduled as soon as possible (a gate starts when all its qubits are free); a qubit that waits between
two gates, or until the readout, relaxes thermally for the waiting time. The measured X_0 and X_4 have asymmetric
readout (assignment) errors. The state preparation (the amplitude encoding) is ideal.

Parameters: typical medians (TYPICAL) with a log-normal dispersion from qubit to qubit, drawn once per "chip" seed
(or identical qubits, spread=False); a global scale s multiplies every error rate (gate errors, readout errors and the
relaxation rates 1/T1, 1/T_phi): s = 0 is the noiseless circuit, s = 1 the typical device.

Symmetry of the equivariant circuit: the gates that the swap of the registers exchanges have the same duration and
start at the same time, so the noise, including the waiting times, is the same on the two qubits of every pair
exchanged by the swap only if they have the same parameters (spread=False), and then the swap stays an exact symmetry;
the amplitude damping is not unital and does not commute with X, so it breaks the reflections.

Two exact engines run the same program: the Heisenberg picture (the observable evolved backwards, one 2^n x 2^n matrix
per call, used in training) and PennyLane default.mixed (density matrices, the reference used in the tests).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
import torch

# typical values of a current superconducting processor (medians). A single-qubit rotation by an arbitrary angle is
# two sqrt(X) pulses; a rotation on four qubits is charged, on each of its qubits, the error and the duration of the
# six two-qubit gates of its usual compilation shared among the four qubits (three two-qubit gates per qubit).
TYPICAL = {
    "gate1_error": 4e-4,  # rotation on one qubit (2 pulses of error 2e-4)
    "gate1_time": 100e-9,
    "gate4_error": 9e-3,  # rotation on four qubits, per qubit (3 two-qubit gates of error 3e-3)
    "gate4_time": 480e-9,  # 6 two-qubit layers of 80 ns
    "t1": 200e-6,
    "t2": 150e-6,
    "readout_p10": 0.01,  # P(read 1 | state 0)
    "readout_p01": 0.02,  # P(read 0 | state 1): larger, the excited state decays during the readout
}
# log-normal dispersion from qubit to qubit (standard deviation of the logarithm)
SPREAD = {"gate1_error": 0.5, "gate4_error": 0.4, "t1": 0.3, "t2": 0.3, "readout_p10": 0.5, "readout_p01": 0.5}


@dataclass
class Chip:
    """Per-qubit parameters of one sampled chip, at scale 1."""
    num_qubits: int
    seed: int
    q: list = field(default_factory=list)

    @classmethod
    def sample(cls, num_qubits: int, seed: int, spread: bool = True):
        rng = np.random.default_rng(seed)
        chip = cls(num_qubits, seed)
        for _ in range(num_qubits):
            z = {k: float(rng.standard_normal()) for k in SPREAD}
            draw = {k: TYPICAL[k] * (math.exp(SPREAD[k] * z[k]) if spread else 1.0) for k in SPREAD}
            draw["t2"] = min(draw["t2"], 2 * draw["t1"])
            chip.q.append(draw)
        return chip


# ---------------------------------------------------------------- channels
def relaxation_kraus(t: float, t1: float, t2: float, scale: float):
    """Thermal relaxation (zero temperature) for a time t, rates multiplied by `scale`: amplitude damping then pure
    dephasing, as [(name, parameter, [2x2 Kraus])] in time order (PennyLane AmplitudeDamping / PhaseDamping)."""
    if scale == 0 or t == 0:
        return []
    g1 = 1 - math.exp(-scale * t / t1)
    rate_phi = max(0.0, 1 / t2 - 1 / (2 * t1)) * scale
    gphi = 1 - math.exp(-2 * t * rate_phi)  # off-diagonal * sqrt(1 - gphi) = exp(-t / T_phi)
    ad = [np.array([[1, 0], [0, math.sqrt(1 - g1)]]), np.array([[0, math.sqrt(g1)], [0, 0]])]
    pd = [np.array([[1, 0], [0, math.sqrt(1 - gphi)]]), np.array([[0, 0], [0, math.sqrt(gphi)]])]
    return [("ad", g1, ad), ("pd", gphi, pd)]


def entanglement_fidelity(kraus_lists) -> float:
    """Entanglement fidelity of the composition of single-qubit channels (each a list of 2x2 Kraus operators)."""
    ops = [np.eye(2)]
    for kraus in kraus_lists:
        ops = [k @ o for o in ops for k in kraus]
    return float(sum(abs(np.trace(o)) ** 2 for o in ops) / 4)


def depolarizing_lambda(gate_error: float, relax_fe: float) -> float:
    """lambda of the single-qubit depolarizing channel (1 - lambda) rho + lambda I/2 Tr rho such that the average
    infidelity of depolarizing + relaxation is gate_error (to first order the infidelities add):
    r = 1 - (2 F_e + 1)/3 for the relaxation, r_dep = lambda/2."""
    r_relax = 1 - (2 * relax_fe + 1) / 3
    return max(0.0, gate_error - r_relax) * 2


# ---------------------------------------------------------------- the noisy program
# readouts: [(coefficient, wire, Pauli)]; each term is measured in its basis with the readout errors of its qubit
READOUT_O = ((0.5, 0, "X"), (0.5, 4, "X"))  # O = (X_0 + X_4)/2
READOUT_NOSYM = tuple((1 / (1 + math.sqrt(2)), w, p) for w, p in ((0, "X"), (0, "Z"), (4, "Z")))  # Equiv-NoSym


def compile_program(gates, num_qubits: int, chip: Chip, scale: float, readout_terms=READOUT_O):
    """gates: [(pauli word {wire: P}, parameter slot)] of the circuit (rotations exp(-i theta/2 P)).
    Returns (program, readout): program, in time order, with the operations
      ("rot", word, slot)                 the gate
      ("depol", wire, lambda)             single-qubit depolarizing
      ("kraus1", wire, name, g, [Kraus])  amplitude damping ("ad") or phase damping ("pd")
    and readout = [(coefficient, wire, Pauli, gain, offset)]: the measured <P_w> is gain <P_w> + offset."""
    readout_wires = sorted({w for _, w, _ in readout_terms})
    prog = []
    avail = [0.0] * num_qubits  # time at which each qubit is free

    def idle(w, t):
        for name, g, kraus in relaxation_kraus(t, chip.q[w]["t1"], chip.q[w]["t2"], scale):
            prog.append(("kraus1", w, name, g, kraus))

    for word, slot in gates:
        kind = "gate1" if len(word) == 1 else "gate4"
        wires = sorted(word)
        t0 = max(avail[w] for w in wires)
        for w in wires:  # waiting for the other qubits of the gate
            idle(w, t0 - avail[w])
            avail[w] = t0 + TYPICAL[f"{kind}_time"]
        prog.append(("rot", word, slot))
        if scale == 0:
            continue
        for w in wires:
            p = chip.q[w]
            relax = relaxation_kraus(TYPICAL[f"{kind}_time"], p["t1"], p["t2"], scale)
            lam = depolarizing_lambda(scale * p[f"{kind}_error"], entanglement_fidelity([k for *_, k in relax]))
            if lam > 0:
                prog.append(("depol", w, lam))
            for name, g, kraus in relax:
                prog.append(("kraus1", w, name, g, kraus))
    t_end = max(avail)
    for w in readout_wires:  # waiting for the end of the circuit (simultaneous readout)
        idle(w, t_end - avail[w])
    readout = []
    for coef, w, pauli in readout_terms:
        p10, p01 = scale * chip.q[w]["readout_p10"], scale * chip.q[w]["readout_p01"]
        readout.append((coef, w, pauli, 1 - p10 - p01, p01 - p10))  # measured mean = (1 - p10 - p01) <P> + p01 - p10
    return prog, readout


# ---------------------------------------------------------------- Heisenberg engine
_PAULI2 = {"X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _pauli_table(word: dict, num_qubits: int):
    """(P v)[j] = phase[j] v[perm[j]], wire 0 the most significant bit."""
    dim = 2 ** num_qubits
    j = torch.arange(dim)
    mask = sum(1 << (num_qubits - 1 - w) for w, c in word.items() if c in "XY")
    src = j ^ mask
    phase = torch.ones(dim, dtype=torch.complex128)
    for w, c in word.items():
        bit = (src >> (num_qubits - 1 - w)) & 1
        if c == "Z":
            phase = phase * (1 - 2 * bit)
        elif c == "Y":
            phase = phase * torch.where(bit == 0, 1j, -1j).to(torch.complex128)
    return src, phase


def _adjoint_superop(kraus) -> torch.Tensor:
    """4x4 matrix M of A -> sum_K K^dag A K on one qubit: vec(A')[i,j] = M[ij,kl] vec(A)[k,l]."""
    m = sum(np.einsum("ki,lj->ijkl", np.conj(k), k) for k in kraus)
    return torch.as_tensor(m.reshape(4, 4), dtype=torch.complex128)


def _depol_superop(lam) -> torch.Tensor:
    """(1 - lam) A + lam I/2 Tr A (self-adjoint)."""
    e = np.eye(2)
    m = (1 - lam) * np.einsum("ik,jl->ijkl", e, e) + lam / 2 * np.einsum("ij,kl->ijkl", e, e)
    return torch.as_tensor(m.reshape(4, 4), dtype=torch.complex128)


def _rotation_superop(pch, angle):
    """Adjoint superoperator of exp(-i angle/2 P) on one qubit (differentiable in angle): A -> U^dag A U."""
    p = torch.as_tensor(_PAULI2[pch], dtype=torch.complex128)
    u = torch.cos(angle / 2) * torch.eye(2, dtype=torch.complex128) - 1j * torch.sin(angle / 2) * p
    return torch.einsum("ki,lj->ijkl", u.conj(), u).reshape(4, 4)


def _apply_superop(a, w, m, num_qubits):
    lo, hi = 2 ** w, 2 ** (num_qubits - 1 - w)
    a6 = a.reshape(lo, 2, hi, lo, 2, hi)
    return torch.einsum("ijkl,akbcld->aibcjd", m.reshape(2, 2, 2, 2), a6).reshape(a.shape)


class HeisenbergProgram:
    """The program evolved backwards. Consecutive operations on one qubit (single-qubit rotations and noise) are
    merged into one 4x4 superoperator, until a rotation on several qubits acts on that qubit; operations on different
    qubits commute, so the merging is exact."""

    def __init__(self, program, readout, num_qubits, twirl: bool = False):
        self.n = num_qubits
        self.ops = []
        self.perms = None
        if twirl:
            from eqnn.models import d4_pixel_perms
            self.perms = d4_pixel_perms()
        pending = {}  # wire -> components in time order: constant 4x4 tensor or (pauli, slot)

        def push(w, comp):
            comps = pending.setdefault(w, [])
            if isinstance(comp, torch.Tensor) and comps and isinstance(comps[-1], torch.Tensor):
                comps[-1] = comps[-1] @ comp  # the adjoint of E1 then E2 is E1^dag o E2^dag
            else:
                comps.append(comp)

        def flush(wires):
            for w in wires:
                comps = pending.pop(w, None)
                if comps:
                    self.ops.append(("super", w, comps))

        for op in program:
            if op[0] == "rot" and len(op[1]) == 1:
                (w, pch), = op[1].items()
                push(w, (pch, op[2]))
            elif op[0] == "rot":
                flush(sorted(op[1]))
                self.ops.append(("rot", _pauli_table(op[1], num_qubits), op[2]))
            elif op[0] == "depol":
                push(op[1], _depol_superop(op[2]))
            elif op[0] == "kraus1":
                push(op[1], _adjoint_superop(op[4]))
        flush(list(pending))
        obs = 0
        for coef, w, pauli, gain, offset in readout:  # the measured observable, term by term
            m = torch.as_tensor(qml.matrix(getattr(qml, pauli)(w), wire_order=range(num_qubits)), dtype=torch.complex128)
            obs = obs + coef * (gain * m + offset * torch.eye(2 ** num_qubits, dtype=torch.complex128))
        self.obs = obs

    def observable(self, params):
        a = self.obs
        for op in reversed(self.ops):
            if op[0] == "super":
                m = None
                for comp in op[2]:
                    cm = comp if isinstance(comp, torch.Tensor) else _rotation_superop(comp[0], params[comp[1]])
                    m = cm if m is None else m @ cm
                a = _apply_superop(a, op[1], m, self.n)
            else:
                (perm, ph), slot = op[1], op[2]
                c, s = torch.cos(params[slot] / 2), torch.sin(params[slot] / 2)
                pa = ph[:, None] * a[perm, :]  # P A
                ap = a[:, perm] * ph[perm][None, :]  # A P
                pap = ph[:, None] * ap[perm, :]  # P A P
                a = c * c * a + s * s * pap + 1j * c * s * (pa - ap)  # U^dag A U
        return a

    def __call__(self, states, params):
        x = states.to(torch.float64)
        x = x / x.norm(dim=-1, keepdim=True)
        a = self.observable(params).real
        if self.perms is not None:  # twirled model: classical average over the transformed inputs (noiseless)
            from eqnn.models import twirl_matrix
            a = twirl_matrix(a, self.perms)
        return torch.einsum("bj,jk,bk->b", x, a, x)


# ---------------------------------------------------------------- PennyLane default.mixed reference
def mixed_model(program, readout, num_qubits, twirl: bool = False):
    dev = qml.device("default.mixed", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(states, params):
        qml.StatePrep(states, wires=range(num_qubits), normalize=True)
        for op in program:
            if op[0] == "rot":
                wires = sorted(op[1])
                qml.PauliRot(params[op[2]], "".join(op[1][w] for w in wires), wires=wires)
            elif op[0] == "depol":
                qml.DepolarizingChannel(3 * op[2] / 4, wires=op[1])  # PennyLane p = 3 lambda / 4
            elif op[0] == "kraus1":
                (qml.AmplitudeDamping if op[2] == "ad" else qml.PhaseDamping)(op[3], wires=op[1])
        return [qml.expval(getattr(qml, pauli)(w)) for _, w, pauli, _, _ in readout]

    def one(states, params):
        outs = circuit(states, params)
        return sum(coef * (g * x + o) for (coef, _, _, g, o), x in zip(readout, outs))

    if not twirl:
        return one
    from eqnn.models import d4_pixel_perms
    perms = d4_pixel_perms()
    return lambda states, params: torch.stack([one(states[:, p], params) for p in perms]).mean(dim=0)


def build(gates, num_qubits, scale: float, chip_seed: int = 0, spread: bool = True, twirl: bool = False,
          engine: str = "heisenberg", readout_terms=READOUT_O):
    """twirl: the twirled model, mean over the eight transformed inputs of the noisy circuit (the transformation of
    the input is a classical permutation of the amplitudes, noiseless)."""
    chip = Chip.sample(num_qubits, chip_seed, spread)
    program, readout = compile_program(gates, num_qubits, chip, scale, readout_terms=readout_terms)
    if engine == "heisenberg":
        return HeisenbergProgram(program, readout, num_qubits, twirl)
    if engine == "mixed":
        return mixed_model(program, readout, num_qubits, twirl)
    raise ValueError(engine)
