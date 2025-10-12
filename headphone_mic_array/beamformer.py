from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Use shared geometry primitives
from .geometry import Node


@dataclass
class BeamformerConfig:
    fs: int = 48000
    hop_ms: float = 10.0
    nfft: Optional[int] = None  # if None, next pow2 ≥ Nw
    speed_of_sound: float = 343.0
    ema_alpha: float = 0.95      # covariance EMA
    diag_load: float = 0.01      # diagonal loading fraction
    update_every_n_frames: int = 1


class MVDRBeamformer:
    """Readable, streaming MVDR beamformer.

    Design goals for readability:
    - Small, well-named helpers (analysis, mvdr step, synthesis)
    - Clear state (buffers, steering, covariance)
    - Fewer premature optimizations; prefer explicit steps
    """

    def __init__(
        self,
        mic_positions: List[Node],
        target_direction: Optional[Tuple[float, float]] = None,
        target_point: Optional[Node] = None,
        config: BeamformerConfig = BeamformerConfig(),
    ) -> None:
        self.cfg = config
        self.fs = config.fs
        self.Nw = int(round(self.fs * self.cfg.hop_ms * 2 / 1000.0))
        self.Nh = int(round(self.fs * self.cfg.hop_ms / 1000.0))
        if self.Nh <= 0 or self.Nw <= 0:
            raise ValueError("window_ms and hop_ms must yield positive sample counts.")
        if self.Nh > self.Nw:
            raise ValueError("Nh must be <= Nw for overlap.")

        if self.cfg.nfft is None:
            # next power of two ≥ Nw
            n = 1
            while n < self.Nw:
                n <<= 1
            self.nfft = n
        else:
            self.nfft = int(self.cfg.nfft)
        if self.nfft < self.Nw:
            raise ValueError(
                f"nfft ({self.nfft}) must be >= analysis window length ({self.Nw})."
            )

        self.mic_pos = np.stack([m.as_array() for m in mic_positions], axis=0)  # (M,3)
        self.M = self.mic_pos.shape[0]

        # Analysis/synthesis windows (sqrt-Hann for COLA at 50% overlap)
        self.win = np.sqrt(np.hanning(self.Nw).astype(float))

        # Frequency grid (rfft)
        self.freqs = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)  # (F,)
        self.F = self.freqs.shape[0]

        # MVDR state per frequency bin
        self._Rvv = np.array([np.eye(self.M, dtype=np.complex128) for _ in range(self.F)])
        self._invRvv = np.array([np.eye(self.M, dtype=np.complex128) for _ in range(self.F)])

        # Rolling input buffers per mic for framing
        self._inbuf = np.zeros((self.M, self.Nw), dtype=float)
        self._inbuf_fill = 0

        # Output OLA buffer (fixed-size = Nw)
        self._olabuf = np.zeros(self.Nw, dtype=float)
        self._frame_idx = 0

        # Target/steering
        self._near_field = False
        azel = target_direction if target_direction is not None else (0.0, 0.0)
        self.set_target(az_deg=azel[0], el_deg=azel[1], point=target_point)


    # ----------------------------- Target control ---------------------------- #

    def set_target(
        self,
        az_deg: Optional[float] = None,
        el_deg: Optional[float] = None,
        point: Optional[Node] = None,
    ) -> None:
        """Unified target setter.

        If `point` is provided, steer to that 3D point (spherical, near-field).
        Otherwise steer far-field using (az_deg, el_deg). Defaults to az=0, el=0
        if angles are not provided.
        """
        if point is not None:
            self.set_target_point(point)
            return
        # Far-field fallback: default az=0, el=0 if not given
        az = 0.0 if az_deg is None else az_deg
        el = 0.0 if el_deg is None else el_deg
        self.set_target_direction(az, el)

    def set_target_direction(self, az_deg: float, el_deg: float) -> None:
        """Set far-field steering using azimuth/elevation in degrees."""
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        u = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)])
        self._near_field = False
        self._target_u = u / np.linalg.norm(u)
        self._target_point = None
        self._update_steering()

    def set_target_point(self, point: Node) -> None:
        """Set near-field steering to a specific 3D point (meters)."""
        self._near_field = True
        self._target_point = point
        self._target_u = None
        self._update_steering()

    # --------------------------- Steering computation ------------------------ #

    def _compute_delays(self) -> np.ndarray:
        """Return delays tau[m] in seconds for each mic relative to reference."""
        c = self.cfg.speed_of_sound
        if self._near_field:
            p = self._target_point.as_array()
            dists = np.linalg.norm(self.mic_pos - p[None, :], axis=1)  # (M,)
            tau = (dists - dists.mean()) / c
        else:
            u = self._target_u
            tau = -(self.mic_pos @ u) / c
            tau -= tau.mean()
        return tau.astype(float)

    def _update_steering(self) -> None:
        """Compute frequency-dependent steering vectors and projection matrices."""
        tau = self._compute_delays()
        phase = -2j * np.pi * self.freqs[:, None] * tau[None, :]
        d = np.exp(phase)
        # d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        self._d = d.astype(np.complex128)
        I = np.eye(self.M, dtype=np.complex128)
        self._B = np.empty((self.F, self.M, self.M), dtype=np.complex128)
        for k in range(self.F):
            dk = self._d[k][:, None]
            self._B[k] = I - dk @ dk.conj().T

    # ------------------------------- Processing ------------------------------ #

    def process_block(self, x_block: np.ndarray) -> np.ndarray:
        """Process an arbitrary-length time-domain block and return beamformed samples."""
        assert x_block.ndim == 2 and x_block.shape[0] == self.M, (
            f"x_block must be shape (M,T) with M={self.M}"
        )
        x_block = np.asarray(x_block, dtype=float)
        emitted = []
        tpos = 0
        T = x_block.shape[1]
        while tpos < T:
            need = self.Nh - self._inbuf_fill
            take = min(need, T - tpos)
            if take > 0:
                self._inbuf = np.roll(self._inbuf, -take, axis=1)
                self._inbuf[:, -take:] = x_block[:, tpos : tpos + take]
                self._inbuf_fill += take
                tpos += take
            if self._inbuf_fill >= self.Nh:
                y_hop = self._process_one_frame()
                emitted.append(y_hop)
                self._inbuf_fill -= self.Nh
        return np.concatenate(emitted, axis=0) if emitted else np.empty(0, dtype=float)

    def _process_one_frame(self) -> np.ndarray:
        """Process current buffered frame; emit exactly Nh samples via OLA."""
        # ---- Analysis (STFT per mic) ----
        X = self._analysis()
        # ---- MVDR per frequency bin ----
        Y = np.empty(self.F, dtype=np.complex128)
        update_weights = (self._frame_idx % self.cfg.update_every_n_frames) == 0
        for k in range(self.F):
            Y[k] = self._mvdr_step(k, X[k], update_weights)
        # ---- Synthesis (iFFT + window) ----
        frame_td = self._synthesis(Y)
        # ---- OLA (fixed-size buffer) ----
        self._olabuf += frame_td
        y_out = self._olabuf[: self.Nh].copy()
        self._olabuf[:-self.Nh] = self._olabuf[self.Nh:]
        self._olabuf[-self.Nh:] = 0.0
        self._frame_idx += 1
        return y_out

    # ------------------------------- Core helpers ------------------------------ #

    def _analysis(self) -> np.ndarray:
        """Window last frame and compute one-sided rFFT for each mic."""
        x_win = (self._inbuf * self.win[None, :]).astype(float)  # (M, Nw)
        X = np.fft.rfft(x_win, n=self.nfft, axis=1)  # (M, F)
        return X.T.copy()  # (F, M)

    def _mvdr_step(self, k: int, xk: np.ndarray, update_weights: bool) -> np.complex128:
        """One MVDR update/application at frequency bin k."""
        if update_weights:
            Bk = self._B[k]
            x_proj = Bk @ xk
            Rvv_k = self.cfg.ema_alpha * self._Rvv[k] + (1 - self.cfg.ema_alpha) * (
                np.outer(x_proj, x_proj.conj())
            )
            load = self.cfg.diag_load * (np.trace(Rvv_k).real / self.M + 1e-12)
            Rvv_k = Rvv_k + load * np.eye(self.M, dtype=np.complex128)
            invR = np.linalg.pinv(Rvv_k)
            self._Rvv[k] = Rvv_k
            self._invRvv[k] = invR
        invR = self._invRvv[k]
        dk = self._d[k]
        num = invR @ dk
        den = dk.conj().T @ num
        wk = num / (den + 1e-12)
        return wk.conj().T @ xk

    def _synthesis(self, Y: np.ndarray) -> np.ndarray:
        """Inverse rFFT of single-channel spectrum and apply synthesis window."""
        y_frame = np.fft.irfft(Y, n=self.nfft).real
        return y_frame[: self.Nw] * self.win

    # ----------------------------- Utility methods --------------------------- #

    def reset_covariance(self) -> None:
        """Reset Rvv and invRvv to identity (useful when retargeting)."""
        self._Rvv[:] = np.eye(self.M, dtype=np.complex128)
        self._invRvv[:] = np.eye(self.M, dtype=np.complex128)

    def set_ema_time_constant(self, tau_seconds: float) -> None:
        """Set EMA alpha given desired time constant in seconds (approx)."""
        frames_per_sec = self.fs / self.Nh
        alpha = math.exp(-1.0 / max(frames_per_sec * tau_seconds, 1e-6))
        self.cfg.ema_alpha = float(np.clip(alpha, 0.0, 0.9999))
