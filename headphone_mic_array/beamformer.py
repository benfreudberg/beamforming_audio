from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Use shared geometry primitives
from .geometry import Node

import matplotlib.pyplot as plt


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
        np.set_printoptions(linewidth=350)
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
        self.mic_pos1 = mic_positions
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

        # Update only when target is low power
        self._td_ema = np.zeros(self.F, dtype=float)   # smoothed target ratio per bin
        self._gate_state = np.zeros(self.F, dtype=bool) # True => allowed to update at this bin
        # Gate params (tweak if needed)
        self._gate_alpha = 0.7     # time smoothing of ratio (higher = smoother)
        self._gate_on  = 0.12      # allow updates when r_ema < on
        self._gate_off = 0.18      # stop updates when r_ema > off
        self._r_accum = []
        self._p_accum = []


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
            # p = self._target_point.as_array()
            # dists = np.linalg.norm(self.mic_pos - p[None, :], axis=1)  # (M,)
            dists = np.zeros(self.M)
            for i, mic in enumerate(self.mic_pos1):
                dists[i] = mic.distance_to(self._target_point)
            tau = (dists - dists.min()) / c
        else:
            u = self._target_u
            tau = -(self.mic_pos @ u) / c
            tau -= tau.min()
        return tau.astype(float)

    def _update_steering(self) -> None:
        """Compute frequency-dependent steering vectors and projection matrices."""
        tau = self._compute_delays()
        phase = -2j * np.pi * self.freqs[:, None] * tau[None, :]
        ref_mic_index = self.M//2 #center mic
        d_raw = np.exp(phase)
        d_blk = d_raw / np.linalg.norm(d_raw, axis=1, keepdims=True) + 1e-12
        d_mvdr = d_raw / (d_raw[:, [ref_mic_index]] + 1e-12)
        self._d_blk = d_blk.astype(np.complex128)
        self._d_mvdr = d_mvdr.astype(np.complex128)
        I = np.eye(self.M, dtype=np.complex128)
        self._B = np.empty((self.F, self.M, self.M), dtype=np.complex128)
        for k in range(self.F):
            dk = self._d_blk[k][:, None]
            self._B[k] = I - dk @ dk.conj().T

        # self.plot_beam_direct(k_idx=self.F//3, c=self.cfg.speed_of_sound)
        # self.plot_beam_direct(k_idx=self.F//8, c=self.cfg.speed_of_sound)
        # self.plot_beam_direct(k_idx=self.F//32, c=self.cfg.speed_of_sound)
        # self.plot_beam_direct(k_idx=self.F//64, c=self.cfg.speed_of_sound)

    def plot_beam_direct(self, k_idx, w=None, theta_deg=np.linspace(-180, 180, 721), c=343):
        """
        Plot beampattern for arbitrary mic geometry via direct array-factor evaluation.
        - theta is measured from broadside (z-axis), in the x–z plane.
        """
        # frequency / wavenumber
        f = float(self.freqs[k_idx])
        k = 2*np.pi*f/c

        # weights (default to steering vector at this frequency)
        if w is None:
            w = self._d_mvdr[k_idx]                 # shape (M,)
        w = np.asarray(w).reshape(-1)

        # mic positions (M,3)
        R = np.asarray(self.mic_pos)           # columns: x, y, z

        # scan unit vectors u(theta) = [sinθ, -cosθ, 0]
        theta = np.deg2rad(theta_deg)
        u = np.stack([np.sin(theta), -np.cos(theta), np.zeros_like(theta)], axis=1)  # (T,3)

        # steering matrix A(t, m) = exp(-j k r_m · u_t)
        A = np.exp(-1j * k * (u @ R.T))        # shape (T, M)

        # beampattern B(θ) = |w^H a(θ)| -> A @ conj(w)
        B = np.abs(A @ np.conj(w))
        B /= B.max() + 1e-15
        B_dB = 20*np.log10(np.maximum(B, 1e-6))

        # find peak
        theta_max = theta[np.argmax(B_dB)]

        # polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, B_dB)
        ax.plot([theta_max], [0.0], 'ro')
        ax.text(theta_max - 0.1, -4, f"{np.degrees(theta_max):.2f}°")

        ax.set_theta_zero_location('N')        # 0° up (broadside)
        ax.set_theta_direction(1)             # increase ccw
        ax.set_thetamin(-180); ax.set_thetamax(180)
        ax.set_ylim([-30, 1])
        ax.set_rlabel_position(55)
        ax.set_title(f"Direct Beampattern @ {int(np.round(f))} Hz")
        plt.show()

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
            if k < 9:
                Y[k] = 0.0 # filter out low frequencies
            else:
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
        # --- NEW: compute smoothed target ratio and hysteresis-gated update flag ---
        du = self._d_blk[k]
        tpow = float(np.abs(np.vdot(du, xk))**2)                   # |d^H x|^2
        upow = float((xk.conj() @ xk).real + 1e-12)                # x^H x
        r = tpow / upow                                            # instantaneous ratio
        self._r_accum.append(r)

        x_proj = self._B[k] @ xk
        p_perp = (x_proj.conj() @ x_proj).real / upow
        interferer_present = p_perp > 0.15  # tune 0.1–0.3
        self._p_accum.append(p_perp)

        a = self._gate_alpha
        self._td_ema[k] = a * self._td_ema[k] + (1.0 - a) * r      # EMA

        # hysteresis: prevents chattering
        if self._gate_state[k]:
            # currently allowed; turn OFF if ratio rises above off-threshold
            self._gate_state[k] = self._td_ema[k] > self._gate_off
        else:
            # currently blocked; turn ON if ratio falls below on-threshold
            self._gate_state[k] = self._td_ema[k] < self._gate_on

        # allow_update = update_weights and self._gate_state[k] and interferer_present
        allow_update = self._gate_state[k] and interferer_present

        if allow_update:
            Rvv_k = self.cfg.ema_alpha * self._Rvv[k] + (1 - self.cfg.ema_alpha) * (np.outer(xk, xk.conj()))
            Rvv_k = 0.5*(Rvv_k + Rvv_k.conj().T)
            lam = self.cfg.diag_load * (np.trace(Rvv_k).real / self.M + 1e-12)
            Rvv_k += lam * np.eye(self.M, dtype=np.complex128)
            self._Rvv[k] = Rvv_k
            self._invRvv[k] = np.linalg.pinv(Rvv_k)

        dk = self._d_mvdr[k]  # (ref-mic normalized steering for constraint)
        invR = self._invRvv[k]
        num = invR @ dk
        den = dk.conj().T @ num
        wk = num / (den + 1e-12)
        wk = wk / (np.vdot(wk, dk) + 1e-12)  # keep w^H d = 1 exactly

        # if self._frame_idx in [10, 100, 300, 1000, 2000]:
        if self._frame_idx == 3000:
            if k in [3, 9, 35, 60]:
            # if k == 35:
                self.plot_beam_direct(k_idx=k, w=wk, c=self.cfg.speed_of_sound)

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
