"""Compute the quantum transmission coefficient and save figures.

Two approaches:

1. **Deterministic** — run the wave packet until it has fully cleared the
   barrier, then integrate |ψ|² on each side.  Compared with the exact
   quantum-mechanical formula.

2. **Monte Carlo** (matches ``Relfexion.cpp``) — projective measurements
   at fixed intervals reveal whether the particle is transmitted; T is
   estimated as N_T / m.  Plotted as a function of the measurement interval.

Outputs (all under ``figures/``):

    reflection_vs_lambda.png      — T(λ) deterministic + analytical
    reflection_vs_niter.png       — T vs measurement interval (MC)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import (
    simulate,
    transmission,
    rectangular_barrier,
    gaussian_wavepacket,
    wave_vector,
    transmission_analytical,
    transmission_mc,
)

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

N = 100


# ---------------------------------------------------------------------------
# Deterministic T(λ)
# ---------------------------------------------------------------------------

def _T_deterministic(lam: float) -> float:
    """Run until fully separated, return T = integral of |ψ|² on right."""
    k0   = wave_vector(N)
    psi0 = gaussian_wavepacket(N, k0)
    V    = rectangular_barrier(N, lam, k0)
    # 1500 steps is enough for all λ to let both transmitted and reflected
    # parts fully separate from the barrier.
    traj, _, _ = simulate(psi0, V, k0, n_steps=1500, save_every=1500)
    return transmission(traj[-1], N, barrier_end=0.6)


def plot_T_vs_lambda() -> None:
    lambdas = np.concatenate([
        np.linspace(0.05, 0.95, 20),  # below-barrier (classical transmission)
        np.linspace(1.05, 5.00, 20),  # above-barrier (tunnelling)
    ])
    lambdas = np.sort(lambdas)

    print("  Computing T(lam) deterministically ...")
    T_num = np.array([_T_deterministic(l) for l in lambdas])
    T_ana = np.array([transmission_analytical(l, N) for l in lambdas])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lambdas, T_ana, "k--", lw=1.5, label="analytical $T(\\lambda)$")
    ax.plot(lambdas, T_num, "o", ms=5, color="royalblue",
            label="Crank-Nicolson (numerical)")
    ax.axvline(1.0, color="gray", ls=":", lw=1.0, alpha=0.7, label="$\\lambda=1$  ($E=V_0$)")
    ax.set_xlabel(r"$\lambda = V_0 / E$", fontsize=11)
    ax.set_ylabel(r"Transmission coefficient $T$", fontsize=11)
    ax.set_title(
        f"Quantum tunnelling through a rectangular barrier  ($N={N}$, $k_0=\\pi/2$)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "reflection_vs_lambda.png"), dpi=150)
    plt.close(fig)
    print("  saved reflection_vs_lambda.png")


# ---------------------------------------------------------------------------
# Monte Carlo: T vs measurement interval (matches depiter.txt)
# ---------------------------------------------------------------------------

def plot_T_vs_niter() -> None:
    lam = 0.3
    m   = 200    # measurements per simulation
    iter_values = np.unique(np.concatenate([
        np.arange(1, 50, 3),
        np.arange(50, 200, 10),
        np.arange(200, 1001, 50),
    ])).astype(int)

    print(f"  Monte Carlo T vs num_iter  (lam={lam}, m={m}, {len(iter_values)} points) ...")
    T_mc = np.array([
        transmission_mc(N, lam, m=m, num_iter=int(ni), seed=42)
        for ni in iter_values
    ])

    T_exact = transmission_analytical(lam, N)
    print(f"    T_exact = {T_exact:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iter_values, T_mc, "-o", ms=3, lw=1.0, color="steelblue",
            label=f"MC estimate  ($m={m}$ meas.)")
    ax.axhline(T_exact, color="k", ls="--", lw=1.2,
               label=f"analytical $T = {T_exact:.3f}$")
    ax.set_xlabel("steps between measurements  $n_{\\rm iter}$", fontsize=11)
    ax.set_ylabel("$T$", fontsize=11)
    ax.set_title(
        f"Transmission via projective measurements  "
        f"($\\lambda={lam}$, $N={N}$, $m={m}$)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "reflection_vs_niter.png"), dpi=150)
    plt.close(fig)
    print("  saved reflection_vs_niter.png")


def main():
    plot_T_vs_lambda()
    plot_T_vs_niter()
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
