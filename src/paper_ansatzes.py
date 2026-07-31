"""config6-config9: a matched D4(p4m)-equivariant / non-equivariant ansatz
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

Two ansatzes are available, named by their TOTAL trainable-parameter count:
"6" (paper6, 3 angles/block) and "18" (shared18, 9 angles/block). Both use
the same 5-block schedule: 4 "fine" blocks share ONE set of trainable
angles (stage 0, "fine_shared"), and the final "coarse" block gets its own
(stage 1) — hence only 2 * parameters_per_block total trainable angles,
tied via the spec's "group" mechanism, regardless of there being 5 blocks.

Measured with readout="avg_x" in build_qnn_from_spec (the mean of X over
every qubit — the same quantity config1-config5 measure in effect, via
H-then-Z) by default; readout="x0_xhalf" (0.5*(X_0 + X_4)) is an available
alternative that reads only the two "pooled-to" wires instead of all 8 —
both preserve p4m-equivariance for config6/config8.

The equivariant trainable blocks commute with the three D4 generators
above. Their non-equivariant counterparts are obtained by cycling the
Pauli axis on the column register only (X -> Y, Y -> Z, Z -> X). This
preserves parameter count, parameter sharing, gate arities, gate count, and
depth while deliberately misaligning the ansatz with the image symmetry —
config6/config8 (equivariant) vs config7/config9 (non-equivariant).

Only num_qubits == 8 is supported: the block schedule is hand-crafted for
exactly 4 coordinate qubits per axis (16x16 images), matching this
project's fixed 8-qubit / img_size=16 default throughout.
"""

from dataclasses import dataclass
from typing import Any

N_COORD_QUBITS = 4
N_QUBITS = 2 * N_COORD_QUBITS
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
BLOCK_SCHEDULE: tuple[OrbitBlockSpec, ...] = (
    OrbitBlockSpec(0, "fine-01", 0, 1),
    OrbitBlockSpec(0, "fine-23", 2, 3),
    OrbitBlockSpec(0, "shift-12", 1, 2),
    OrbitBlockSpec(0, "wrap-30", 3, 0),
    OrbitBlockSpec(1, "coarse-02", 0, 2),
)

PAPER6_PARAMETER_NAMES = ("rx_a", "rx_b", "ryyyy_cross")
SHARED18_PARAMETER_NAMES = (
    "rx_pre_a",
    "rx_pre_b",
    "ryy_within",
    "rzz_within",
    "rx_mid_a",
    "rx_mid_b",
    "ryyyy_cross",
    "rzzzz_cross",
    "mixed_cross",
)
PAPER_ANSATZ_PARAMETER_NAMES = {
    "6": PAPER6_PARAMETER_NAMES,
    "18": SHARED18_PARAMETER_NAMES,
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


def _paired_rx_spec(stage: int, name: str, bit: int) -> list[dict[str, Any]]:
    """Same RX angle applied to matching row/column coordinate bits."""

    group = f"stage{stage}_{name}"
    return [
        {"gate": "RX", "wires": [bit], "param": _param(group)},
        {"gate": "RX", "wires": [bit + N_COORD_QUBITS], "param": _param(group)},
    ]


def _axis_scrambled_rx_spec(stage: int, name: str, bit: int) -> list[dict[str, Any]]:
    """Matched symmetry-breaking RX (row) / RY (column) pair."""

    group = f"stage{stage}_{name}"
    return [
        {"gate": "RX", "wires": [bit], "param": _param(group)},
        {"gate": "RY", "wires": [bit + N_COORD_QUBITS], "param": _param(group)},
    ]


def _paper6_block_spec(
    stage: int, a: int, b: int, equivariant: bool
) -> list[dict[str, Any]]:
    names = PAPER6_PARAMETER_NAMES
    cross_wires = [a, b, a + N_COORD_QUBITS, b + N_COORD_QUBITS]

    rx = _paired_rx_spec if equivariant else _axis_scrambled_rx_spec
    gates = rx(stage, names[0], a) + rx(stage, names[1], b)

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


def _shared18_block_spec(
    stage: int, a: int, b: int, equivariant: bool
) -> list[dict[str, Any]]:
    names = SHARED18_PARAMETER_NAMES
    coord = N_COORD_QUBITS
    cross_wires = [a, b, a + coord, b + coord]
    rx = _paired_rx_spec if equivariant else _axis_scrambled_rx_spec

    gates: list[dict[str, Any]] = []
    gates += rx(stage, names[0], a)
    gates += rx(stage, names[1], b)

    if equivariant:
        # Two identical gates together exponentiate the swap-invariant
        # generators YY_row + YY_col and ZZ_row + ZZ_col.
        for offset in (0, coord):
            gates.append(
                {
                    "gate": "ISINGYY",
                    "wires": [a + offset, b + offset],
                    "param": _param(f"stage{stage}_{names[2]}"),
                }
            )
            gates.append(
                {
                    "gate": "ISINGZZ",
                    "wires": [a + offset, b + offset],
                    "param": _param(f"stage{stage}_{names[3]}"),
                }
            )
    else:
        # Cycle the Pauli axis on the column register only.
        gates.append(
            {
                "gate": "ISINGYY",
                "wires": [a, b],
                "param": _param(f"stage{stage}_{names[2]}"),
            }
        )
        gates.append(
            {
                "gate": "ISINGZZ",
                "wires": [a + coord, b + coord],
                "param": _param(f"stage{stage}_{names[2]}"),
            }
        )
        gates.append(
            {
                "gate": "ISINGZZ",
                "wires": [a, b],
                "param": _param(f"stage{stage}_{names[3]}"),
            }
        )
        gates.append(
            {
                "gate": "ISINGXX",
                "wires": [a + coord, b + coord],
                "param": _param(f"stage{stage}_{names[3]}"),
            }
        )

    gates += rx(stage, names[4], a)
    gates += rx(stage, names[5], b)

    if equivariant:
        # These Pauli words have even Y/Z parity in each coordinate
        # register and are fixed by row/column exchange.
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "YYYY",
                "param": _param(f"stage{stage}_{names[6]}"),
            }
        )
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "ZZZZ",
                "param": _param(f"stage{stage}_{names[7]}"),
            }
        )
        # The two mixed Pauli words are exchanged by the row<->column swap;
        # tying their angle exactly exponentiates their twirled sum.
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "YYZZ",
                "param": _param(f"stage{stage}_{names[8]}"),
            }
        )
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "ZZYY",
                "param": _param(f"stage{stage}_{names[8]}"),
            }
        )
    else:
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "YYZZ",
                "param": _param(f"stage{stage}_{names[6]}"),
            }
        )
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "ZZXX",
                "param": _param(f"stage{stage}_{names[7]}"),
            }
        )
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "YYXX",
                "param": _param(f"stage{stage}_{names[8]}"),
            }
        )
        gates.append(
            {
                "gate": "PAULIROT",
                "wires": cross_wires,
                "pauli_word": "ZZZZ",
                "param": _param(f"stage{stage}_{names[8]}"),
            }
        )
    return gates


def paper_architecture_spec(
    paper_ansatz: str, symmetry: str, num_qubits: int
) -> list[dict[str, Any]]:
    """Builds the config6-config9 gate-by-gate spec (see module docstring)
    for src.ansatz_builder.build_qnn_from_spec — use with readout="avg_x"
    (the default create_qnn uses) or readout="x0_xhalf" to preserve the
    intended (non-)equivariance.
    """
    if num_qubits != N_QUBITS:
        raise ValueError(
            f"Paper ansatzes require num_qubits == {N_QUBITS}, got {num_qubits}"
        )
    if paper_ansatz not in PAPER_ANSATZ_CHOICES:
        raise ValueError(
            f"paper_ansatz must be one of {PAPER_ANSATZ_CHOICES}, got {paper_ansatz!r}"
        )
    equivariant = get_symmetry_mode(symmetry) == EQUIVARIANT
    block_fn = _paper6_block_spec if paper_ansatz == "6" else _shared18_block_spec

    spec: list[dict[str, Any]] = []
    for block in BLOCK_SCHEDULE:
        spec += block_fn(block.stage, block.a, block.b, equivariant)
    return spec
