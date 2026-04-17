"""Evolve a Gaussian wave packet through a rectangular barrier and save figures.

Outputs (all under ``figures/``):

    wavepacket_low.png    — snapshots |ψ|² for λ=0.3 (partial tunnelling)
    wavepacket_high.png   — snapshots |ψ|² for λ=3.0 (strong reflection)
    wavepacket_norm.png   — norm conservation ‖ψ(t)‖² ≈ 1
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from schrodinger import (
    simulate,
    gaussian_wavepacket,
    rectangular_barrier,
    wave_vector,
)

FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

N       = 100
N_STEPS = 1000
SAVE_EVERY = 1   # save every step so we can pick any snapshot


def plot_snapshots(lam: float, frames: list[int], label: str, fname: str) -> None:
    """Save a grid of |ψ(x,t)|² snapshots for a given barrier height λ."""
    k0 = wave_vector(N)
    psi0 = gaussian_wavepacket(N, k0)
    V    = rectangular_barrier(N, lam, k0)
    s    = 1.0 / (4.0 * k0 ** 2)

    traj, norms, times = simulate(psi0, V, k0, n_steps=N_STEPS, save_every=SAVE_EVERY)

    x = np.arange(N + 1)
    prob_max = np.max(np.abs(psi0) ** 2) * 1.4

    nrows, ncols = 2, len(frames) // 2
    fig = plt.figure(figsize=(ncols * 3.2, nrows * 2.6))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, frame in enumerate(frames):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        prob = np.abs(traj[frame]) ** 2
        # Shade barrier region
        j1, j2 = int(0.4 * N), int(0.6 * N)
        ax.axvspan(j1, j2, color="gold", alpha=0.35, label="barrier" if idx == 0 else None)
        ax.plot(x, prob, color="royalblue", lw=1.2)
        ax.set_ylim(0, prob_max)
        ax.set_xlim(0, N)
        ax.set_title(f"$t = {frame}\\,\\Delta t$", fontsize=9)
        ax.set_xlabel("$x$ [grid]", fontsize=8)
        ax.set_ylabel("$|\\psi|^2$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Wave-packet evolution — barrier $\\lambda = {lam}$  "
        f"($V_0 = {lam}\\,k_0^2$, $E = k_0^2$)",
        fontsize=11,
    )
    if 0 in frames:
        fig.axes[0].legend(fontsize=7, loc="upper right")

    fig.savefig(os.path.join(FIGDIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def plot_norm(lam: float = 0.3) -> None:
    """Save norm(t) plot for a single simulation."""
    k0   = wave_vector(N)
    psi0 = gaussian_wavepacket(N, k0)
    V    = rectangular_barrier(N, lam, k0)

    _, norms, times = simulate(psi0, V, k0, n_steps=N_STEPS, save_every=10)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, norms, lw=1.0, color="steelblue")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5, label="exact")
    ax.set_xlabel("time  $[1/(4k_0^2)]$", fontsize=10)
    ax.set_ylabel(r"$\|\psi(t)\|^2$", fontsize=10)
    ax.set_title(f"Norm conservation — Cayley scheme  ($\\lambda = {lam}$, $N = {N}$)", fontsize=11)
    deviation = np.max(np.abs(norms - 1.0))
    ax.text(0.98, 0.05, f"max$\\,|\\|\\psi\\|^2-1| = {deviation:.2e}$",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "wavepacket_norm.png"), dpi=150)
    plt.close(fig)
    print("  saved wavepacket_norm.png")


def main():
    print(f"N = {N},  n_steps = {N_STEPS}")

    # 6 snapshots evenly spread over 1000 steps
    frames = [0, 150, 300, 500, 700, 999]

    print("Simulating lam=0.3 (partial tunnelling) ...")
    plot_snapshots(0.3, frames, label="lam=0.3", fname="wavepacket_low.png")

    print("Simulating lam=3.0 (strong reflection) ...")
    plot_snapshots(3.0, frames, label="lam=3.0", fname="wavepacket_high.png")

    print("Norm conservation plot ...")
    plot_norm(lam=0.3)

    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
