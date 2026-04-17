"""1D time-dependent Schrödinger equation — Cayley (Crank-Nicolson) solver."""

from .solver import simulate, transmission, reflection
from .initial_conditions import (
    gaussian_wavepacket,
    rectangular_barrier,
    wave_vector,
    transmission_analytical,
)
from .reflection import transmission_mc

__all__ = [
    "simulate",
    "transmission",
    "reflection",
    "gaussian_wavepacket",
    "rectangular_barrier",
    "wave_vector",
    "transmission_analytical",
    "transmission_mc",
]
