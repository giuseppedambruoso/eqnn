"""The two circuits of the paper (paper: Methods, "The four models", and the figure of the circuits), adapted from Chang et al. (2023).

Encoding: |psi(x)> = sum_ij x_ij |i>_row |j>_col on 8 qubits, the row index on wires 0-3 and the column index on wires
4-7 (most significant bit first). D4 acts on the qubits as X layers on one register (the reflections t_x, t_y) and as
the swap of the two registers (the transposition).

One layer applies five blocks. A block on the coordinate bits (a, b) acts on the row qubits a, b and on their column
partners a' = a + 4, b' = b + 4:

    Equiv     B(theta) = exp(-i theta_3/2 Y_a Y_b Y_a' Y_b') exp(-i theta_2/2 (X_b + X_b')) exp(-i theta_1/2 (X_a + X_a'))
    NonEquiv  B(theta) = exp(-i theta_3/2 Y_a Y_b Z_a' Z_b') exp(-i theta_2/2 (X_b + Y_b')) exp(-i theta_1/2 (X_a + Y_a'))

i.e. NonEquiv cycles the Pauli axes of the column register (X -> Y, Y -> Z). The four "fine" blocks on the bit pairs
(0, 1), (2, 3), (1, 2), (3, 0) share three angles, the "coarse" block on (0, 2) has its own three angles: six angles per
layer. L layers stack L copies, each with its own angles; the angle vector is
[theta_1^(0), theta_2^(0), theta_3^(0), theta_1^(1), theta_2^(1), theta_3^(1)] for layer 0, then layer 1, and so on.
"""

NUM_QUBITS = 8
HALF = NUM_QUBITS // 2  # qubits per register
FINE_BLOCKS = ((0, 1), (2, 3), (1, 2), (3, 0))
COARSE_BLOCK = (0, 2)
ANGLES_PER_LAYER = 6


def layer_gates(equivariant: bool):
    """The gates of one layer in time order, as (Pauli word {wire: 'X' | 'Y' | 'Z'}, index of the angle in the layer);
    each gate is the rotation exp(-i theta/2 P) by its angle."""
    col_rot = "X" if equivariant else "Y"  # single-qubit rotations on the column register
    col_ent = "Y" if equivariant else "Z"  # column factors of the four-qubit rotation
    gates = []
    for stage, blocks in ((0, FINE_BLOCKS), (1, (COARSE_BLOCK,))):
        for a, b in blocks:
            for k, q in enumerate((a, b)):  # theta_1 on (a, a'), theta_2 on (b, b'): the same angle on the pair
                gates.append(({q: "X"}, 3 * stage + k))
                gates.append(({q + HALF: col_rot}, 3 * stage + k))
            gates.append(({a: "Y", b: "Y", a + HALF: col_ent, b + HALF: col_ent}, 3 * stage + 2))
    return gates


def circuit_gates(equivariant: bool, layers: int):
    """The gates of the circuit with `layers` layers, as (Pauli word, index in the angle vector)."""
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    return [(word, ANGLES_PER_LAYER * layer + k) for layer in range(layers) for word, k in layer_gates(equivariant)]

