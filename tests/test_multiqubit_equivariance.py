"""Exact p4m invariance of config6 (Equiv) and config10 (NonEquiv +
twirling), and its absence for config7 (NonEquiv), at 8, 10 and 12
qubits — checked on ALL 8 elements of D4 (not only the generators), with
random parameters well away from the near-identity initialization and
random signed images."""

import pytest
import torch

from src.data_encoding import embedding_state
from src.qnn import architecture_param_names, create_qnn


def d4_orbit(img: torch.Tensor) -> list[torch.Tensor]:
    rotations = [torch.rot90(img, k, dims=(-2, -1)) for k in range(4)]
    return rotations + [torch.flip(r, dims=[-1]) for r in rotations]


def max_orbit_deviation(arch: str, num_qubits: int, readout: str, seed: int = 0) -> float:
    torch.manual_seed(seed)
    side = 2 ** (num_qubits // 2)
    qnn = create_qnn("default.qubit", num_qubits, 2, arch, readout=readout)
    params = torch.empty(len(architecture_param_names(arch, num_qubits, 2))).uniform_(-3, 3)
    img = torch.randn(side, side, dtype=torch.float64)
    img = img / img.norm()
    batch = torch.stack([embedding_state(v) for v in d4_orbit(img)])
    out = qnn(batch, params)
    return (out - out[0]).abs().max().item()


@pytest.mark.parametrize("num_qubits", [8, 10, 12])
@pytest.mark.parametrize("readout", ["avg_x", "x0_xhalf"])
@pytest.mark.parametrize("arch", ["config6", "config10"])
def test_invariant_architectures(arch: str, num_qubits: int, readout: str) -> None:
    assert max_orbit_deviation(arch, num_qubits, readout) < 1e-10


@pytest.mark.parametrize("num_qubits", [8, 10, 12])
def test_nonequivariant_architecture_breaks_symmetry(num_qubits: int) -> None:
    assert max_orbit_deviation("config7", num_qubits, "avg_x") > 1e-3
