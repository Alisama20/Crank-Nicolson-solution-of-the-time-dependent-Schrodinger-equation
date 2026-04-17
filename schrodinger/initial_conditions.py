"""Initial conditions for the 1D TDSE on a uniform grid of N+1 points.

Grid convention: j = 0 … N, spatial step Δx = 1 (absorbed in the
wavefunction normalisation).  Dirichlet BCs: ψ(0) = ψ(N) = 0.

Units: ħ = 1, 2m = 1  →  E = k²,  T = 1/(4k₀²).
"""

import numpy as np

# Fraction of the box occupied by the barrier (matches the C++ code).
BARRIER_START = 2 / 5   # left edge at 2N/5
BARRIER_END   = 3 / 5   # right edge at 3N/5


def wave_vector(N: int) -> float:
    """Central wave-vector k₀ = 2π * (N//4) / N."""
    return 2.0 * np.pi * (N // 4) / N


def gaussian_wavepacket(N: int, k0: float | None = None) -> np.ndarray:
    """Normalised Gaussian wave packet centred at x₀ = N/4, σ = N/16.

    Returns a complex array of length N+1 with ψ(0) = ψ(N) = 0.
    """
    if k0 is None:
        k0 = wave_vector(N)
    x0    = N / 4
    sigma = N / 16
    j = np.arange(N + 1, dtype=np.float64)
    envelope = np.exp(-(j - x0) ** 2 / (2 * sigma ** 2))
    psi = (np.cos(j * k0) + 1j * np.sin(j * k0)) * envelope
    psi[0] = psi[N] = 0.0
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi / norm


def rectangular_barrier(N: int, lam: float, k0: float | None = None) -> np.ndarray:
    """Rectangular potential barrier V(x) = λ k₀² for x ∈ [2N/5, 3N/5].

    Parameters
    ----------
    N   : grid size (N+1 points)
    lam : barrier height parameter (V₀ = λ k₀²; E = k₀², so λ = V₀/E)
    k0  : wave vector (defaults to ``wave_vector(N)``)

    Returns a real array of length N+1.
    """
    if k0 is None:
        k0 = wave_vector(N)
    V = np.zeros(N + 1)
    j1 = int(BARRIER_START * N)
    j2 = int(BARRIER_END   * N)
    V[j1 : j2 + 1] = lam * k0 ** 2
    return V


def transmission_analytical(lam: float, N: int, k0: float | None = None) -> float:
    """Analytical transmission coefficient for a rectangular barrier.

    Uses the exact quantum-mechanical formula for both above-barrier
    (λ < 1) and tunnelling (λ > 1) regimes.
    """
    if k0 is None:
        k0 = wave_vector(N)
    L = (BARRIER_END - BARRIER_START) * N   # barrier width (grid units)
    if abs(lam - 1.0) < 1e-10:
        # E = V₀ limit: T = 1/(1 + k₀²L²/4)
        return 1.0 / (1.0 + k0 ** 2 * L ** 2 / 4.0)
    elif lam < 1.0:
        kappa = k0 * np.sqrt(1.0 - lam)
        return 1.0 / (1.0 + lam ** 2 * np.sin(kappa * L) ** 2 / (4.0 * (1.0 - lam)))
    else:
        kappa_p = k0 * np.sqrt(lam - 1.0)
        return 1.0 / (1.0 + lam ** 2 * np.sinh(kappa_p * L) ** 2 / (4.0 * (lam - 1.0)))
