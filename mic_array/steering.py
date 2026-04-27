"""
steering.py
===========
Shared steering-vector / time-delay math for far-field DOA and beamforming.

Conventions
-----------
* Azimuth (az):    angle from +x axis in the x-y plane, positive toward +y.
* Elevation (el):  angle above the x-y plane, positive toward +z.
* Both in degrees in public APIs unless noted.
* Unit direction vector for a source AT direction (az, el):
      u = [cos(el)cos(az), cos(el)sin(az), sin(el)]
* For a far-field plane wave arriving FROM direction u, the geometric delay
  at mic m relative to the origin is  tau_m = -(r_m . u) / c.
  We then subtract the minimum so all delays are >= 0 (causal).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "unit_vector",
    "delays_far_field",
    "delays_near_field",
    "steering_matrix",
    "steering_grid",
]


def unit_vector(az_deg: float | np.ndarray, el_deg: float | np.ndarray) -> np.ndarray:
    """Unit direction vector(s) for one or more (az, el) in degrees.

    Returns shape broadcast of inputs with trailing dim 3.
    """
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    cx = np.cos(el) * np.cos(az)
    cy = np.cos(el) * np.sin(az)
    cz = np.sin(el)
    return np.stack([cx, cy, cz], axis=-1)


def delays_far_field(
    mic_pos: np.ndarray,
    az_deg: float | np.ndarray,
    el_deg: float | np.ndarray,
    c: float = 343.0,
    causal: bool = True,
) -> np.ndarray:
    """Per-mic delays (seconds) for a far-field source at (az, el).

    Parameters
    ----------
    mic_pos : (M, 3)
    az_deg, el_deg : scalar or arrays of compatible shape (..., )
    c : speed of sound, m/s
    causal : if True, subtract the minimum delay so all values are >= 0.

    Returns
    -------
    tau : shape (..., M) of delays in seconds.
    """
    u = unit_vector(az_deg, el_deg)         # (..., 3)
    # tau = -(r . u) / c
    # mic_pos (M,3); u (...,3) -> (..., M)
    tau = -np.einsum("...d,md->...m", u, mic_pos) / c
    if causal:
        tau = tau - tau.min(axis=-1, keepdims=True)
    return tau


def delays_near_field(
    mic_pos: np.ndarray,
    point: np.ndarray,
    c: float = 343.0,
    causal: bool = True,
) -> np.ndarray:
    """Per-mic delays for a near-field point source at `point` (3,) in meters."""
    point = np.asarray(point, dtype=float).reshape(3)
    dists = np.linalg.norm(mic_pos - point[None, :], axis=1)
    tau = dists / c
    if causal:
        tau = tau - tau.min()
    return tau


def steering_matrix(
    mic_pos: np.ndarray,
    freqs_hz: np.ndarray,
    az_deg: float | np.ndarray,
    el_deg: float | np.ndarray,
    c: float = 343.0,
) -> np.ndarray:
    """Steering vectors a(f, theta) = exp(-j 2 pi f tau).

    Parameters
    ----------
    mic_pos : (M, 3)
    freqs_hz : (F,)
    az_deg, el_deg : broadcast-compatible direction params

    Returns
    -------
    A : shape (..., F, M) complex steering vectors.
    """
    tau = delays_far_field(mic_pos, az_deg, el_deg, c=c, causal=True)  # (..., M)
    f = np.asarray(freqs_hz, dtype=float)                              # (F,)
    # phase shape: (..., F, M)
    phase = -2j * np.pi * f[..., :, None] * tau[..., None, :]
    return np.exp(phase)


def steering_grid(
    mic_pos: np.ndarray,
    freqs_hz: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    c: float = 343.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outer-product steering grid over az x el.

    Returns
    -------
    A : (A_n, E_n, F, M) complex
    az_grid_deg : (A_n,)
    el_grid_deg : (E_n,)
    """
    az = np.asarray(az_deg, dtype=float).reshape(-1)
    el = np.asarray(el_deg, dtype=float).reshape(-1)
    AZ, EL = np.meshgrid(az, el, indexing="ij")     # (A,E)
    A = steering_matrix(mic_pos, freqs_hz, AZ, EL, c=c)  # (A,E,F,M)
    return A, az, el
