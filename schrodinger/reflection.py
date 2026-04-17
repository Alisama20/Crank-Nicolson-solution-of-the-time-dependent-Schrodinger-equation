"""Monte Carlo quantum-measurement simulation of the transmission coefficient.

Replicates the algorithm in ``Relfexion.cpp``:

At fixed intervals of ``num_iter`` time steps, a projective measurement is
performed on the half-space to the right of the barrier (j ≥ j_right).

1. Draw ξ ~ U(0,1).
2. If ξ ≤ P_right (probability of being found right of barrier):
       → particle detected as *transmitted*, reinitialise ψ₀.
   Else:
       → particle not detected on the right, collapse ψ onto the left
         half-space (zero out positions j ≥ j_right), renormalise.

The procedure is repeated ``m`` times; the ratio N_T/m estimates T.
"""

import numpy as np
from .solver import _step, _init_alpha
from .initial_conditions import gaussian_wavepacket, rectangular_barrier, wave_vector


def _prob_right(psi: np.ndarray, j_right: int) -> float:
    return float(np.sum(np.abs(psi[j_right:]) ** 2))


def transmission_mc(
    N: int,
    lam: float,
    m: int,
    num_iter: int,
    seed: int = 0,
) -> float:
    """Monte Carlo estimate of the transmission coefficient.

    Parameters
    ----------
    N        : grid size
    lam      : barrier height parameter (V₀ = λk₀²)
    m        : number of measurements
    num_iter : number of Crank-Nicolson steps between measurements
    seed     : RNG seed

    Returns
    -------
    T_est : float — estimated transmission coefficient
    """
    rng  = np.random.default_rng(seed)
    k0   = wave_vector(N)
    s    = 1.0 / (4.0 * k0 ** 2)
    V    = rectangular_barrier(N, lam, k0)
    V_c  = np.ascontiguousarray(V, dtype=np.float64)
    alpha = _init_alpha(V_c, s)

    j_right = int(0.4 * N)   # first grid point to the right of barrier start

    def _reset():
        psi = gaussian_wavepacket(N, k0)
        return np.ascontiguousarray(psi, dtype=np.complex128)

    psi  = _reset()
    N_T  = 0
    k    = 0    # measurement counter
    step = 0    # total step counter

    while k < m:
        step += 1
        psi = _step(psi, alpha, V_c, s)

        if step % num_iter == 0:
            k += 1
            p = _prob_right(psi, j_right)
            xi = rng.random()
            if xi <= p:
                N_T += 1
                psi = _reset()
            else:
                psi[j_right:] = 0.0
                norm = float(np.sum(np.abs(psi) ** 2))
                if norm > 1e-14:
                    psi /= np.sqrt(norm)
                else:
                    psi = _reset()

    return N_T / m
