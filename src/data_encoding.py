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
    """Matrix M with M|0...0> = sum_ij x_ij |i>|j> (row-major index i*cols+j)
    and zeros elsewhere — exactly sum_ij coordinate_to_unitary(i, j, image),
    built directly instead of as 256 Kronecker products (~0.2 s/image)."""
    rows, cols = image.shape
    out = torch.zeros(rows * cols, rows * cols, dtype=torch.float64)
    out[:, 0] = image.reshape(-1).to(torch.float64)
    return out


def embedding_state(image: torch.Tensor) -> torch.Tensor:
    """The encoded state sum_ij x_ij |i>|j> itself, as a 2^(2n) float64
    vector (row-major index i*cols+j) — the first column of
    embedding_unitary(image), without the dense 2^(2n) x 2^(2n) matrix
    (4096 x 4096 = 134 MB per image at 12 qubits)."""
    return image.reshape(-1).to(torch.float64)


def as_state_vector(encoded: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Accepts either state vectors (dim,) / (batch, dim) or legacy
    embedding_unitary matrices (dim, dim) / (batch, dim, dim), and returns
    state vectors. A legacy matrix M only ever acts on |0...0>, so its
    encoded state is exactly its first column. Only ambiguous for a batch
    of exactly `dim` state vectors, which never occurs with this project's
    batch sizes (N // 10 <= 64 < dim = 256)."""
    dim = 2**num_qubits
    is_matrix = encoded.shape[-1] == dim and encoded.ndim >= 2 and encoded.shape[-2] == dim
    return encoded[..., :, 0] if is_matrix else encoded
