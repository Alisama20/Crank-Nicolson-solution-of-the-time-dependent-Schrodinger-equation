"""Cayley (Crank-Nicolson) integrator for the 1D time-dependent Schrödinger equation.

Solves  i ∂ψ/∂t = Hψ,  H = −∂²/∂x² + V(x)  on a uniform grid of N+1 points
with Dirichlet BCs ψ(0) = ψ(N) = 0 and spatial step Δx = 1.

The time step is  Δt = s = 1/(4k₀²)  (same convention as the reference C++ code).

Propagator
----------
The Cayley approximation

    U = (1 − iHΔt/2)(1 + iHΔt/2)⁻¹

is unitary by construction, so the norm ‖ψ‖ is preserved to machine precision
at every step.

Setting χ = ψⁿ⁺¹ + ψⁿ the Cayley equation reduces to

    (1 + iHΔt/2) χ = 2ψⁿ

which is a complex tridiagonal system with

    diagonal    d_j = (−2 − V_j) + 2i/s
    off-diagonal     = 1  (both sides)
    RHS         b_j = 4i·ψⁿ_j / s

solved in O(N) via the Thomas algorithm.  The diagonal recursion coefficients
α_j depend only on V and are precomputed once.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Precomputation (time-independent α)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _init_alpha(V: np.ndarray, s: float) -> np.ndarray:
    """Backward sweep to compute the α coefficients of the Thomas algorithm."""
    N = len(V) - 1
    alpha = np.zeros(N + 1, dtype=np.complex128)
    for j in range(N - 1, 0, -1):
        dj = (-2.0 - V[j]) + (2.0 / s) * 1j
        alpha[j - 1] = -1.0 / (dj + alpha[j])
    return alpha


# ---------------------------------------------------------------------------
# Single time step
# ---------------------------------------------------------------------------

@njit(cache=True)
def _step(psi: np.ndarray, alpha: np.ndarray,
          V: np.ndarray, s: float) -> np.ndarray:
    """Advance ψ by one time step Δt = s."""
    N = len(psi) - 1

    # Backward pass: compute β
    beta = np.zeros(N + 1, dtype=np.complex128)
    for j in range(N, 0, -1):
        dj = (-2.0 - V[j]) + (2.0 / s) * 1j
        rhs = (4.0 / s) * 1j * psi[j]
        beta[j - 1] = (rhs - beta[j]) / (dj + alpha[j])

    # Forward pass: compute χ
    chi = np.zeros(N + 1, dtype=np.complex128)
    for j in range(N - 1):
        chi[j + 1] = alpha[j] * chi[j] + beta[j]

    # ψ^{n+1} = χ − ψ^n
    return chi - psi


# ---------------------------------------------------------------------------
# Full simulation driver
# ---------------------------------------------------------------------------

def simulate(
    psi0: np.ndarray,
    V: np.ndarray,
    k0: float,
    n_steps: int,
    save_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the 1D TDSE with the Cayley scheme.

    Parameters
    ----------
    psi0       : initial (normalised) wavefunction, shape (N+1,)
    V          : potential array, shape (N+1,)
    k0         : central wave vector (sets Δt = 1/(4k₀²))
    n_steps    : number of time steps
    save_every : record a frame every ``save_every`` steps

    Returns
    -------
    traj   : (F, N+1) complex array — ψ at each saved frame
    norms  : (F,) array           — ‖ψ‖² at each saved frame
    times  : (F,) array           — simulation time  t = k·Δt
    """
    s = 1.0 / (4.0 * k0 ** 2)
    V_c = np.ascontiguousarray(V, dtype=np.float64)
    alpha = _init_alpha(V_c, s)
    psi = np.ascontiguousarray(psi0, dtype=np.complex128)

    n_frames = n_steps // save_every + 1
    traj  = np.empty((n_frames, len(psi)), dtype=np.complex128)
    norms = np.empty(n_frames)
    times = np.empty(n_frames)

    traj[0]  = psi
    norms[0] = float(np.sum(np.abs(psi) ** 2))
    times[0] = 0.0

    frame = 1
    for k in range(1, n_steps + 1):
        psi = _step(psi, alpha, V_c, s)
        if k % save_every == 0:
            traj[frame]  = psi
            norms[frame] = float(np.sum(np.abs(psi) ** 2))
            times[frame] = k * s
            frame += 1

    return traj[:frame], norms[:frame], times[:frame]


# ---------------------------------------------------------------------------
# Convenience: transmission / reflection from a fully evolved state
# ---------------------------------------------------------------------------

def transmission(psi: np.ndarray, N: int,
                 barrier_end: float = 0.6) -> float:
    """Fraction of the norm to the right of the barrier.

    Call this after the wave packet has fully cleared the barrier.
    """
    j_right = int(barrier_end * N) + 1
    return float(np.sum(np.abs(psi[j_right:]) ** 2))


def reflection(psi: np.ndarray, N: int,
               barrier_start: float = 0.4) -> float:
    """Fraction of the norm to the left of the barrier."""
    j_left = int(barrier_start * N)
    return float(np.sum(np.abs(psi[:j_left]) ** 2))
