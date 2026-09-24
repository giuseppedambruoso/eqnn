"""config6 / config7 (and config10 = twirled config7): a matched
D4(p4m)-equivariant / non-equivariant ansatz
pair for 16x16 images (coordinate-aware amplitude encoding), adapted from
Chang et al., expressed as gate-by-gate specs for
src.ansatz_builder.build_qnn_from_spec.

The image is encoded as

    |psi(x)> = sum_{i,j=0}^{15} x[i,j] |i>_row |j>_col,

with row wires (0, 1, 2, 3) and column wires (4, 5, 6, 7) — exactly the
convention already used by src.qnn's p4m-twirling (apply_group_element):
flipping row/column wires with X gates complements the row/column index
(an image flip along that axis), and swapping row<->column wires transposes
the image. The finite image-symmetry group is the D4 point group generated
by those three operations.

The ansatz ("6", 3 angles per block) uses a 5-block schedule: 4 "fine"
blocks share ONE set of trainable angles (stage 0), and the final "coarse"
block gets its own (stage 1) — hence 6 trainable angles per layer, tied
via the spec's "group" mechanism, regardless of there being 5 blocks.
Several layers (each with its own angles) can be stacked.

Measured with readout="avg_x" in build_qnn_from_spec (the mean of X over
every qubit) by default; readout="x0_xhalf" (0.5*(X_0 + X_4)) is an
alternative that reads only the two "pooled-to" wires instead of all 8 —
both preserve the p4m-equivariance of config6.

The equivariant trainable blocks commute with the three D4 generators
above. Their non-equivariant counterparts are obtained by cycling the
Pauli axis on the column register only (X -> Y, Y -> Z, Z -> X). This
preserves parameter count, parameter sharing, gate arities, gate count, and
depth while deliberately misaligning the ansatz with the image symmetry —
config6 (equivariant) vs config7 (non-equivariant).

num_qubits may be 8, 10 or 12 (16x16, 32x32 or 64x64 images): see
block_schedule, which reproduces the original hand-crafted 8-qubit
schedule exactly and extends it to 5 or 6 coordinate bits per axis.
"""

from dataclasses import dataclass
from typing import Any

N_COORD_QUBITS = 4  # default (8 qubits, 16x16 images)
N_QUBITS = 2 * N_COORD_QUBITS
SUPPORTED_NUM_QUBITS = (8, 10, 12)
PARAMETER_GROUP_NAMES = ("fine_shared", "coarse")
EQUIVARIANT = "equivariant"
NONEQUIVARIANT = "nonequivariant"
SYMMETRY_CHOICES = (EQUIVARIANT, NONEQUIVARIANT)


@dataclass(frozen=True)
class OrbitBlockSpec:
    """One orbit block acting on coordinate bits ``a`` and ``b``."""

    stage: int
    name: str
    a: int
    b: int


# Stage 0 mixes all four binary coordinate bits and then stops acting on
# bits 1 and 3. Stage 1 transfers the remaining information from bit 2 to
# bit 0 — an 8 -> 4 -> 2 pooling schedule (most load-bearing when paired
# with readout="x0_xhalf", which only reads bits 0 and 4 back out).
def block_schedule(n_coord: int) -> tuple[OrbitBlockSpec, ...]:
    """Block schedule for n_coord coordinate bits per axis.

    Stage 0 ("fine", shared angles) couples every pair of neighbouring
    bits on a ring, in brick order: first (0,1), (2,3), ..., then
    (1,2), (3,4), ..., closing the ring with (n_coord-1, 0). Stage 1
    ("coarse", own angles) couples bit 0 with bit n_coord//2. For
    n_coord=4 this is exactly the original hand-crafted schedule
    (0,1), (2,3), (1,2), (3,0) + coarse (0,2).
    """
    even = [(2 * k, 2 * k + 1) for k in range(n_coord // 2)]
    odd = [(2 * k + 1, 2 * k + 2) for k in range((n_coord - 1) // 2)]
    wrap = [(n_coord - 1, 0)]
    blocks = [OrbitBlockSpec(0, f"fine-{a}{b}", a, b) for a, b in even + odd + wrap]
    blocks.append(OrbitBlockSpec(1, f"coarse-0{n_coord // 2}", 0, n_coord // 2))
    return tuple(blocks)


BLOCK_SCHEDULE: tuple[OrbitBlockSpec, ...] = block_schedule(N_COORD_QUBITS)

PAPER6_PARAMETER_NAMES = ("rx_a", "rx_b", "ryyyy_cross")
PAPER_ANSATZ_PARAMETER_NAMES = {
    "6": PAPER6_PARAMETER_NAMES,
}
PAPER_ANSATZ_CHOICES = tuple(PAPER_ANSATZ_PARAMETER_NAMES)


def get_symmetry_mode(symmetry: str) -> str:
    """Resolve a symmetry selector to its canonical string."""

    key = str(symmetry).lower().replace("-", "").replace("_", "")
    aliases = {
        "equivariant": EQUIVARIANT,
        "eq": EQUIVARIANT,
        "nonequivariant": NONEQUIVARIANT,
        "noneq": NONEQUIVARIANT,
        "generic": NONEQUIVARIANT,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        choices = ", ".join(SYMMETRY_CHOICES)
        raise ValueError(
            f"Unknown symmetry mode {symmetry!r}; choose one of: {choices}."
        ) from exc


def _param(group: str) -> dict[str, Any]:
    return {"init": "random", "value": None, "frozen": False, "group": group}


def _paired_rx_spec(stage: int, name: str, bit: int, coord: int) -> list[dict[str, Any]]:
    """Same RX angle applied to matching row/column coordinate bits."""

    group = f"stage{stage}_{name}"
    return [
        {"gate": "RX", "wires": [bit], "param": _param(group)},
        {"gate": "RX", "wires": [bit + coord], "param": _param(group)},
    ]


def _axis_scrambled_rx_spec(
    stage: int, name: str, bit: int, coord: int
) -> list[dict[str, Any]]:
    """Matched symmetry-breaking RX (row) / RY (column) pair."""

    group = f"stage{stage}_{name}"
    return [
        {"gate": "RX", "wires": [bit], "param": _param(group)},
        {"gate": "RY", "wires": [bit + coord], "param": _param(group)},
    ]


def _paper6_block_spec(
    stage: int, a: int, b: int, equivariant: bool, coord: int = N_COORD_QUBITS
) -> list[dict[str, Any]]:
    names = PAPER6_PARAMETER_NAMES
    cross_wires = [a, b, a + coord, b + coord]

    rx = _paired_rx_spec if equivariant else _axis_scrambled_rx_spec
    gates = rx(stage, names[0], a, coord) + rx(stage, names[1], b, coord)

    word = "YYYY" if equivariant else "YYZZ"
    gates.append(
        {
            "gate": "PAULIROT",
            "wires": cross_wires,
            "pauli_word": word,
            "param": _param(f"stage{stage}_{names[2]}"),
        }
    )
    return gates


def paper_architecture_spec(
    paper_ansatz: str, symmetry: str, num_qubits: int, layers: int = 1
) -> list[dict[str, Any]]:
    """Builds the config6/config7 gate-by-gate spec (see module docstring)
    for src.ansatz_builder.build_qnn_from_spec — use with readout="avg_x"
    (the default create_qnn uses) or readout="x0_xhalf" to preserve the
    intended (non-)equivariance.
    """
    if num_qubits not in SUPPORTED_NUM_QUBITS:
        raise ValueError(
            f"Paper ansatzes require num_qubits in {SUPPORTED_NUM_QUBITS}, got {num_qubits}"
        )
    if paper_ansatz not in PAPER_ANSATZ_CHOICES:
        raise ValueError(
            f"paper_ansatz must be one of {PAPER_ANSATZ_CHOICES}, got {paper_ansatz!r}"
        )
    equivariant = get_symmetry_mode(symmetry) == EQUIVARIANT
    block_fn = _paper6_block_spec

    spec: list[dict[str, Any]] = []
    coord = num_qubits // 2
    for block in block_schedule(coord):
        spec += block_fn(block.stage, block.a, block.b, equivariant, coord)
    if layers == 1:
        return spec
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    # `layers` stacked copies of the whole block schedule, each with its OWN
    # trainable angles (distinct parameter groups): layers x the parameter
    # count, same (non-)equivariance as a single layer, since every copy
    # uses the same generators.
    stacked: list[dict[str, Any]] = []
    for layer in range(layers):
        for gate in spec:
            gate = {**gate}
            if "param" in gate:
                gate["param"] = {**gate["param"], "group": f"L{layer}_{gate['param']['group']}"}
            stacked.append(gate)
    return stacked
