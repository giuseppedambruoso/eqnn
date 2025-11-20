# data_encoding.py
import logging

import torch

logger = logging.getLogger(__name__)


def binary_to_01(input_int: int) -> torch.Tensor:
    if input_int == 0:
        return torch.tensor([1, 0], dtype=torch.float64)
    elif input_int == 1:
        return torch.tensor([0, 1], dtype=torch.float64)
    else:
        raise ValueError("Input must be 0 or 1.")


def binary_str_to_basis_state(binary_str: str) -> torch.Tensor | None:
    basis_state = None
    for digit in binary_str:
        qubit = binary_to_01(int(digit))
        basis_state = qubit if basis_state is None else torch.kron(basis_state, qubit)
        if basis_state is not None:
            basis_state = basis_state.reshape(-1, 1)
    return basis_state


def zero_state_bra(num_qubits: int) -> torch.Tensor:
    state = torch.tensor([1, 0], dtype=torch.float64)
    for _ in range(num_qubits - 1):
        state = torch.kron(state, torch.tensor([1, 0], dtype=torch.float64))
    return state.reshape(1, -1)


def coordinate_to_unitary(x: int, y: int, img: torch.Tensor) -> torch.Tensor | None:
    num_qubits = int(torch.log2(torch.tensor(img.shape[0], dtype=torch.float64)).item())

    initial_state = zero_state_bra(num_qubits)

    x_state = binary_str_to_basis_state(format(int(x), f"0{num_qubits}b"))
    y_state = binary_str_to_basis_state(format(int(y), f"0{num_qubits}b"))

    first = None
    second = None
    output = None
    if x_state is not None and y_state is not None:
        first = x_state @ initial_state
        second = y_state @ initial_state
        output = img[x, y].item() * torch.kron(first, second)
    return output


def embedding_unitary(image: torch.Tensor) -> torch.Tensor:
    rows, cols = image.shape
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    parts = [coordinate_to_unitary(i, j, image) for i, j in coords]

    return torch.stack(parts).sum(dim=0)
