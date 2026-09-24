"""2D Ising-model configurations for the "ising" dataset (ordered vs.
disordered phase), generated natively at the model's image size so no
rescaling is ever needed.

Configurations are sampled with checkerboard Metropolis on an L x L
square lattice with periodic boundary conditions (J = 1, k_B = 1), whose
Hamiltonian is invariant under the full p4m point group (and under the
global spin flip) — so the phase label is an exactly D4-invariant
function of the configuration. Temperatures are drawn uniformly from
[t_min, t_max] and each configuration is labelled by whether T is below
the exact infinite-lattice critical temperature T_c = 2/ln(1+sqrt(2)).

All chains start from the fully ordered state (avoiding the long-lived
striped metastable states a random start produces at low T), are
thermalized for `n_sweeps` sweeps, and are finally flipped globally with
probability 1/2 so both magnetization sectors are equally represented.
The whole pool is simulated as one vectorized batch and cached to disk.
"""

import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)

T_CRITICAL = 2.0 / math.log(1.0 + math.sqrt(2.0))

# Label convention: 0 = disordered (T > T_c), 1 = ordered (T < T_c).
ISING_LABELS = {"disordered": 0, "ordered": 1}


def _metropolis_half_sweep(
    spins: np.ndarray, beta: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> None:
    neighbours = (
        np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
        + np.roll(spins, 1, axis=2)
        + np.roll(spins, -1, axis=2)
    )
    delta_e = 2.0 * spins * neighbours
    accept = rng.random(spins.shape) < np.exp(-beta[:, None, None] * delta_e)
    spins[accept & mask] *= -1


def generate_ising_configurations(
    n_configs: int,
    size: int = 16,
    t_min: float = 1.5,
    t_max: float = 3.0,
    n_sweeps: int = 2000,
    seed: int = 0,
    chunk: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (spins, temperatures): spins has shape (n_configs, size,
    size) with entries in {-1, +1}. Chains are simulated in chunks of
    `chunk` configurations to bound memory at large lattice sizes."""
    rng = np.random.default_rng(seed)
    temperatures = rng.uniform(t_min, t_max, size=n_configs)
    spins = np.ones((n_configs, size, size), dtype=np.int8)
    ii, jj = np.indices((size, size))
    black = ((ii + jj) % 2 == 0)[None, :, :]
    white = ~black
    for start in range(0, n_configs, chunk):
        block = spins[start : start + chunk]
        beta = 1.0 / temperatures[start : start + chunk]
        for _ in range(n_sweeps):
            _metropolis_half_sweep(block, beta, black, rng)
            _metropolis_half_sweep(block, beta, white, rng)
        spins[start : start + chunk] = block
    flip = rng.random(n_configs) < 0.5
    spins[flip] *= -1
    return spins, temperatures


def load_or_generate_ising(
    data_dir: str,
    n_configs: int = 6000,
    size: int = 16,
    t_min: float = 1.5,
    t_max: float = 3.0,
    n_sweeps: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached wrapper: returns (spins, temperatures, labels)."""
    path = os.path.join(
        data_dir,
        "ising",
        f"ising_L{size}_n{n_configs}_T{t_min}-{t_max}_sw{n_sweeps}_s{seed}_chunked.npz",
    )
    if os.path.exists(path):
        cached = np.load(path)
        spins, temperatures = cached["spins"], cached["temperatures"]
    else:
        logger.info(f"Generating {n_configs} Ising configurations ({path})")
        spins, temperatures = generate_ising_configurations(
            n_configs, size, t_min, t_max, n_sweeps, seed
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, spins=spins, temperatures=temperatures)
    labels = np.where(
        temperatures < T_CRITICAL, ISING_LABELS["ordered"], ISING_LABELS["disordered"]
    )
    return spins, temperatures, labels
