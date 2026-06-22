from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
from pathlib import Path

from .geometry import Node
from .audio_utils import (
    read_wav_float_mono, resample_to, apply_fractional_delay,
    get_output_dir, save_wav_float32, project_root
)
from .plotting import plot_scene
from .beamformer import (BeamformerConfig, MVDRBeamformer)
from .steering import steering_matrix

from scipy.signal import find_peaks

# --- Constants ---
SPEED_OF_SOUND = 340.0  # m/s


# ---------- Analysis Data Containers ----------

@dataclass
class FrequencyFrame:
    """FFT magnitude spectrum for a single time frame."""
    time_s: float           # centre time of the analysis window (seconds)
    freqs: np.ndarray       # frequency axis, shape (N_bins,)
    magnitudes: np.ndarray  # linear magnitude, shape (N_bins,)


@dataclass
class PeakEntry:
    """A frequency peak being tracked over time."""
    freq_hz: float
    magnitude: float
    track_id: int           # persistent ID assigned by PeakTracker
    mag_ema: float = 0.0    # EMA of magnitude, used for energy-continuity gate


@dataclass
class SourceDirection:
    """Estimated far-field direction of a clustered source."""
    azimuth_deg: float
    elevation_deg: float
    peak_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0
    source_id: Optional[int] = None  # assigned by SourceTracker


class PeakTracker:
    """
    Maintains a live set of frequency tracks across successive FFT frames.

    Continuity heuristic
    --------------------
    * Each existing track is matched to the nearest incoming peak within
      *freq_tolerance_hz*.
    * Unmatched tracks are held for up to *max_miss_frames* before being dropped.
    * Unmatched incoming peaks spawn new tracks.
    * When two peaks are equidistant from a track, prefer the higher-frequency one.

    Parameters
    ----------
    freq_tolerance_hz : float
        Maximum frequency deviation between frames to consider two peaks the same track.
    max_miss_frames : int
        How many consecutive frames a track may produce no peak before being dropped.
    high_freq_bias : float
        Small additive bonus for higher-frequency candidates when matching is ambiguous.
    max_mag_ratio : float
        Energy-continuity gate. A candidate's magnitude must satisfy
        ``1/max_mag_ratio <= mag / track.mag_ema <= max_mag_ratio`` to be
        re-associated with an existing track. Prevents a faint noise bin near
        the dead track's frequency from inheriting its identity. ``inf`` disables.
    mag_ema_alpha : float
        EMA smoothing factor for the per-track magnitude estimate.
    """

    def __init__(
        self,
        freq_tolerance_hz: float = 50.0,
        max_miss_frames: int = 5,
        high_freq_bias: float = 0.01,
        max_mag_ratio: float = 10.0,
        mag_ema_alpha: float = 0.5,
    ) -> None:
        self.freq_tolerance_hz = freq_tolerance_hz
        self.max_miss_frames = max_miss_frames
        self.high_freq_bias = high_freq_bias
        self.max_mag_ratio = max_mag_ratio
        self.mag_ema_alpha = mag_ema_alpha
        self._next_id: int = 0
        self._active_tracks: List[PeakEntry] = []
        self._miss_counts: Dict[int, int] = {}

    def update(self, peaks: List[Tuple[float, float]]) -> List[PeakEntry]:
        """Ingest (freq_hz, magnitude) peaks for the current frame and return
        the updated set of active PeakEntry objects."""
        # Sort incoming peaks by frequency for greedy matching consistency.
        candidates = sorted(peaks, key=lambda p: p[0])
        matched_candidate_idx: set[int] = set()
        new_active: List[PeakEntry] = []

        # Greedy: for each existing track, find best unmatched candidate within tolerance.
        for track in self._active_tracks:
            best_idx = -1
            best_score = float("inf")
            for j, (f_c, m_c) in enumerate(candidates):
                if j in matched_candidate_idx:
                    continue
                df = abs(f_c - track.freq_hz)
                if df > self.freq_tolerance_hz:
                    continue
                # Energy-continuity gate.
                if track.mag_ema > 0 and self.max_mag_ratio < float("inf"):
                    ratio = m_c / max(track.mag_ema, 1e-12)
                    if ratio > self.max_mag_ratio or ratio < 1.0 / self.max_mag_ratio:
                        continue
                # Bias score toward higher-frequency candidates (subtract bonus).
                score = df - self.high_freq_bias * f_c
                if score < best_score:
                    best_score = score
                    best_idx = j
            if best_idx >= 0:
                f_c, m_c = candidates[best_idx]
                track.freq_hz = f_c
                track.magnitude = m_c
                a = self.mag_ema_alpha
                track.mag_ema = (a * m_c + (1 - a) * track.mag_ema) if track.mag_ema > 0 else m_c
                matched_candidate_idx.add(best_idx)
                self._miss_counts[track.track_id] = 0
                new_active.append(track)
            else:
                self._miss_counts[track.track_id] = self._miss_counts.get(track.track_id, 0) + 1
                if self._miss_counts[track.track_id] <= self.max_miss_frames:
                    new_active.append(track)
                else:
                    self._miss_counts.pop(track.track_id, None)

        # Unmatched candidates spawn new tracks.
        for j, (f_c, m_c) in enumerate(candidates):
            if j in matched_candidate_idx:
                continue
            tid = self._next_id
            self._next_id += 1
            entry = PeakEntry(freq_hz=f_c, magnitude=m_c, track_id=tid, mag_ema=m_c)
            self._miss_counts[tid] = 0
            new_active.append(entry)

        self._active_tracks = new_active
        return list(self._active_tracks)

    @property
    def active_tracks(self) -> List[PeakEntry]:
        """Return a snapshot of currently active frequency tracks."""
        return list(self._active_tracks)


class SourceTracker:
    """
    Track clustered SourceDirections across frames with persistent integer IDs.

    Greedy nearest-neighbor matching by great-circle angle within
    `angle_threshold_deg`. Lost sources held for `max_miss_frames` frames.
    """

    def __init__(self, angle_threshold_deg: float = 8.0, max_miss_frames: int = 10) -> None:
        self.angle_threshold_deg = angle_threshold_deg
        self.max_miss_frames = max_miss_frames
        self._next_id: int = 0
        self._active: List[SourceDirection] = []
        self._miss: Dict[int, int] = {}
        self._last_seen: Dict[int, SourceDirection] = {}

    @staticmethod
    def _angular_distance_deg(a: SourceDirection, b: SourceDirection) -> float:
        from .steering import unit_vector
        ua = unit_vector(a.azimuth_deg, a.elevation_deg)
        ub = unit_vector(b.azimuth_deg, b.elevation_deg)
        d = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
        return float(np.degrees(np.arccos(d)))

    def update(self, sources: List[SourceDirection]) -> List[SourceDirection]:
        """Ingest sources observed this frame and return ONLY those that were
        actually observed (with persistent IDs assigned).

        Internal state still holds recently-missed sources for up to
        ``max_miss_frames`` so a brief gap doesn't fragment an ID, but missed
        sources are NOT re-emitted as ghost detections.
        """
        matched: set[int] = set()
        observed: List[SourceDirection] = []   # what we return
        kept_internal: List[SourceDirection] = []  # what stays in self._active for matching

        # Match existing IDs to incoming sources.
        for prev in self._active:
            best_j = -1
            best_d = float("inf")
            for j, src in enumerate(sources):
                if j in matched:
                    continue
                d = self._angular_distance_deg(prev, src)
                if d <= self.angle_threshold_deg and d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                src = sources[best_j]
                src.source_id = prev.source_id
                matched.add(best_j)
                self._miss[prev.source_id] = 0
                self._last_seen[prev.source_id] = src
                observed.append(src)
                kept_internal.append(src)
            else:
                self._miss[prev.source_id] = self._miss.get(prev.source_id, 0) + 1
                if self._miss[prev.source_id] <= self.max_miss_frames:
                    # Hold for matching only -- do NOT emit.
                    kept_internal.append(prev)

        # Unmatched incoming -> new IDs.
        for j, src in enumerate(sources):
            if j in matched:
                continue
            sid = self._next_id
            self._next_id += 1
            src.source_id = sid
            self._miss[sid] = 0
            self._last_seen[sid] = src
            observed.append(src)
            kept_internal.append(src)

        self._active = kept_internal
        return list(observed)

    @property
    def active(self) -> List[SourceDirection]:
        return list(self._active)


# ---------- Core Entities ----------

class SoundSource(Node):
    def __init__(self, x: float, y: float, z: float, wav_file: str | Path, *, scale: float = 1.0, sample_rate: int):
        if not all(isinstance(v, (int, float)) for v in (x, y, z, scale)):
            raise TypeError("Coordinates (x, y, z) and scale must be numeric.")

        wav_file = Path(wav_file)
        if not wav_file.is_absolute():
            wav_file = project_root() / wav_file
        if not wav_file.is_file():
            raise FileNotFoundError(f"The specified wav file does not exist: {wav_file}")
        if scale <= 0:
            raise ValueError("Scale must be greater than 0.")

        super().__init__(float(x), float(y), float(z))
        self.wav_file_name = wav_file
        self.sample_rate = sample_rate
        sr_in, data = read_wav_float_mono(wav_file)
        # Apply the user scale and a scale for distance. The scale for distance will be undone later by the Microphone,
        # we just want the freedom to move the source anywhere without changing the absolute volume at the origin.
        data = data * float(scale) * self.distance_to(Node(0, 0, 0))
        self.data_resampled = resample_to(data, sr_in, sample_rate)

    def __repr__(self) -> str:
        return f"SoundSource(pos=({self.x}, {self.y}, {self.z}), file='{self.wav_file.name}', scale={self.scale})"


class Microphone(Node):
    def __init__(self, name: str, x: float, y: float, z: float, *, sample_rate: int):
        if not isinstance(name, str):
            raise TypeError("Microphone name must be a string.")
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Coordinates must be numeric.")
        super().__init__(float(x), float(y), float(z))
        self.name = name
        self.sample_rate = sample_rate
        self.track: np.ndarray = np.zeros(0, dtype=np.float32)

    def listen_to(self, source: SoundSource) -> None:
        """Listen to a source; optional preloaded_48k bypasses disk I/O & resampling."""
        if not isinstance(source, SoundSource):
            raise TypeError("source must be an instance of SoundSource.")

        # Load source audio data and scale it for distance
        data_resampled = source.data_resampled * 1/self.distance_to(source)

        # Apply delay for distance
        delay_sec = self.distance_to(source) / SPEED_OF_SOUND
        total_delay = delay_sec * self.sample_rate
        delayed = apply_fractional_delay(data_resampled, total_delay)

        # Mix (sum) into this mic's recording
        max_len = max(len(self.track), len(delayed))
        if len(self.track) < max_len:
            self.track = np.pad(self.track, (0, max_len - len(self.track)))
        if len(delayed) < max_len:
            delayed = np.pad(delayed, (0, max_len - len(delayed)))
        self.track += delayed

    def export(self, suffix: str = "mic_track.wav") -> None:
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string.")
        out_dir = get_output_dir()
        out_path = out_dir / f"{self.name}_{suffix}"
        save_wav_float32(out_path, self.sample_rate, self.track)
        print(f"Track saved to: {out_path}")


class VirtualEar(Microphone):
    def __init__(self, side: str, *, sample_rate: int):
        if not isinstance(side, str):
            raise TypeError("side must be a string: 'left_ear' or 'right_ear'.")
        s = side.strip().lower()
        if s == "left_ear":
            name, x, y, z = "left_ear", -0.075, 0.0, 0.0
        elif s == "right_ear":
            name, x, y, z = "right_ear", 0.075, 0.0, 0.0
        else:
            raise ValueError("side must be either 'left_ear' or 'right_ear'.")
        super().__init__(name=name, x=x, y=y, z=z, sample_rate=sample_rate)


# ---------- Target class for beamforming implementation ----------

class Target(Node):
    """
    Delay-and-sum track formed by time-aligning microphones to the latest-arriving
    microphone for waves from this target point, applying per-mic scale, and summing.
    The final sum is normalized by the sum of the per-mic scales used.
    """
    def __init__(self, name: str, x: float, y: float, z: float, *, sample_rate: int):
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Target coordinates must be numeric.")
        super().__init__(float(x), float(y), float(z))
        self._track: np.ndarray = np.zeros(0, dtype=np.float32)
        self.name = name
        self.sample_rate = sample_rate

    def create_track_DS(self, mic_entries: List[Microphone]) -> None:
        if not mic_entries:
            raise RuntimeError("No microphones provided to Target.create_track_DS().")

        # 1) Compute raw delays to each mic (seconds -> samples)
        delays: List[float] = []
        for mic in mic_entries:
            tau_s = self.distance_to(mic) / SPEED_OF_SOUND
            delays.append(tau_s * self.sample_rate)

        # 2) Align: delay each mic by (max_delay - tau_i)
        max_delay = max(delays)
        delayed_signals: List[np.ndarray] = []
        for mic, tau in zip(mic_entries, delays):
            sig = mic.track.astype(np.float32, copy=True)
            extra = max_delay - tau
            sig_d = apply_fractional_delay(sig, extra)
            delayed_signals.append(sig_d)

        # 3) Sum to a single track
        if not delayed_signals:
            self._track = np.zeros(0, dtype=np.float32)
            return

        max_len = max(len(s) for s in delayed_signals)
        stack: List[np.ndarray] = []
        for s in delayed_signals:
            if len(s) < max_len:
                s = np.pad(s, (0, max_len - len(s)))
            stack.append(s)

        summed = np.sum(np.stack(stack, axis=0), axis=0).astype(np.float32)
        scaled = summed / len(mic_entries)

        self._track = scaled

    def create_track_MVDR(self, mic_entries: List[Microphone]) -> None:
        if not mic_entries:
            raise RuntimeError("No microphones provided to Target.create_track_MVDR().")
        beam_former_config = BeamformerConfig(fs=self.sample_rate, hop_ms=3, speed_of_sound=SPEED_OF_SOUND, ema_alpha=0.9, diag_load=0.01, update_every_n_frames=1)
        # bf = MVDRBeamformer(mic_entries, target_direction=self.azel(degrees=True), config=beam_former_config)
        bf = MVDRBeamformer(mic_entries, target_point=self, config=beam_former_config)
        tracks = [mic.track for mic in mic_entries]
        max_len = max(len(track) for track in tracks)
        padded_tracks = [
            np.pad(track, (0, max_len - len(track)), mode="constant")
            for track in tracks
        ]
        x = np.vstack(padded_tracks)
        self._track = bf.process_block(x)


    def export(self, suffix: str = "") -> None:
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string.")
        if self._track is None or len(self._track) == 0:
            raise RuntimeError("No track available. Call create_track_DS() first.")
        out_dir = get_output_dir()
        out_path = out_dir / f"{self.name}_target_{suffix}.wav"
        save_wav_float32(out_path, self.sample_rate, self._track)
        print(f"Target track exported to: {out_path}")


# ---------- Simulation ----------

class MicrophoneArraySimulation:
    def __init__(self, *, sample_rate: int):
        self.sample_rate = sample_rate
        self._microphones: List[Microphone] = []
        self._sources: List[SoundSource] = []
        self._targets: List[Target] = []
        self._cache_48k_by_path: Dict[Path, np.ndarray] = {}  # resampled (48k), unscaled

        # Internal ears (not part of microphones list)
        self.left_ear = VirtualEar("left_ear", sample_rate=sample_rate)
        self.right_ear = VirtualEar("right_ear", sample_rate=sample_rate)

    # ---- Adders ----
    def add_microphone(self, mic: Microphone) -> None:
        if not isinstance(mic, Microphone):
            raise TypeError("mic must be an instance of Microphone.")
        self._microphones.append(mic)

    def add_sound_source(self, source: SoundSource) -> None:
        if not isinstance(source, SoundSource):
            raise TypeError("source must be an instance of SoundSource.")
        self._sources.append(source)

    # ---- Processing ----
    def run_recording(self) -> None:
        """Have every MICROPHONE (ears excluded) listen to every sound source."""
        if not self._microphones:
            raise RuntimeError("No microphones have been added to the simulation.")
        if not self._sources:
            raise RuntimeError("No sound sources have been added to the simulation.")

        print("Simulating recording all sources with all mics.")
        for mic in self._microphones:
            for src in self._sources:
                mic.listen_to(src)

    def apply_ambient_audio_to_ears(self) -> None:
        """
        Process all sources into both ears.
        Does NOT reset recordings.
        """
        if not self._sources:
            raise RuntimeError("No sound sources have been added to the simulation.")

        for ear in (self.left_ear, self.right_ear):
            for src in self._sources:
                ear.listen_to(src)

    def reset_recordings(self, target: Literal["ears", "mics", "all"] = "all") -> None:
        """Reset stored audio data for 'ears', 'mics', or 'all'."""
        if target not in ("ears", "mics", "all"):
            raise ValueError("target must be one of: 'ears', 'mics', 'all'")
        if target in ("ears", "all"):
            self.left_ear.track = np.zeros(0, dtype=np.float32)
            self.right_ear.track = np.zeros(0, dtype=np.float32)
        if target in ("mics", "all"):
            for mic in self._microphones:
                mic.track = np.zeros(0, dtype=np.float32)

    def export_ears_stereo(self, filename: str = "ears_stereo.wav", gain: float = 1.0) -> None:
        """Export the two internal ears as a stereo WAV (float32) to output/.

        Parameters
        ----------
        filename : str
            Output filename inside the output directory.
        gain : float
            Linear amplitude scale applied before writing. Values > 1 make the
            output louder; use with care to avoid clipping.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")

        left = self.left_ear.track.astype(np.float32) * gain
        right = self.right_ear.track.astype(np.float32) * gain
        max_len = max(len(left), len(right))
        if len(left) < max_len:
            left = np.pad(left, (0, max_len - len(left)))
        if len(right) < max_len:
            right = np.pad(right, (0, max_len - len(right)))

        stereo = np.stack([left, right], axis=1).astype(np.float32)
        out_dir = get_output_dir()
        out_path = out_dir / filename
        save_wav_float32(out_path, self.sample_rate, stereo)
        print(f"Stereo ears exported to: {out_path}")

    # ---- Target API ----
    def add_target(self, name: str, x: float, y: float, z: float) -> Target:
        """Create a Target at (x, y, z), build its track, store and return it."""
        t = Target(name, x, y, z, sample_rate=self.sample_rate)
        self._targets.append(t)
        print(f"Target created at ({x}, {y}, {z})")
        return t
    
    def create_target_tracks_DS(self):
        if not self._targets:
            raise RuntimeError("No targets have been added to the simulation.")
        for target in self._targets:
            target.create_track_DS(self._microphones)

    def create_target_tracks_MVDR(self):
        if not self._targets:
            raise RuntimeError("No targets have been added to the simulation.")
        for target in self._targets:
            target.create_track_MVDR(self._microphones)

    def export_target_tracks(self, suffix: str = ""):
        if not self._targets:
            raise RuntimeError("No targets have been added to the simulation.")
        for target in self._targets:
            target.export(suffix)

    def apply_targets_to_ears(self) -> None:
        """
        For each target:
          - compute delay to each ear
          - delay a local copy of the target track by |Δ| samples, where Δ is the delay difference
          - add the delayed copy to the farther ear and the undelayed copy to the closer ear
        """
        if not self._targets:
            raise RuntimeError("No targets have been added to the simulation.")

        for t in self._targets:
            track = t._track.astype(np.float32, copy=True)
            if track is None or len(track) == 0:
                print("Warning: target has empty track; skipping.")
                continue

            # Ear delays (seconds)
            dL_s = self.left_ear.distance_to(t) / SPEED_OF_SOUND
            dR_s = self.right_ear.distance_to(t) / SPEED_OF_SOUND

            # Difference in samples
            diff_samples = abs(dL_s - dR_s) * self.sample_rate
            delayed = apply_fractional_delay(track, diff_samples)

            # Which ear is farther?
            left_farther = dL_s > dR_s

            # Compose additions
            add_left = delayed if left_farther else track
            add_right = track if left_farther else delayed

            # Mix-add into ears with padding
            L = self.left_ear.track
            R = self.right_ear.track
            max_len = max(len(L), len(R), len(add_left), len(add_right))

            if len(L) < max_len:
                L = np.pad(L, (0, max_len - len(L)))
            if len(R) < max_len:
                R = np.pad(R, (0, max_len - len(R)))
            if len(add_left) < max_len:
                add_left = np.pad(add_left, (0, max_len - len(add_left)))
            if len(add_right) < max_len:
                add_right = np.pad(add_right, (0, max_len - len(add_right)))

            self.left_ear.track = (L + add_left).astype(np.float32)
            self.right_ear.track = (R + add_right).astype(np.float32)

            print(f"Applied target at ({t.x}, {t.y}, {t.z}) to ears "
                  f"(Δ={diff_samples:.2f} samples; farther={'left' if left_farther else 'right'}).")

    # ---- Signal Analysis ----
    @staticmethod
    def _stft_params(sample_rate: int, window_s: float) -> tuple[int, int]:
        """Return (Nw, nfft) where nfft is next pow2 >= Nw."""
        Nw = int(round(sample_rate * window_s))
        nfft = 1
        while nfft < Nw:
            nfft <<= 1
        return Nw, nfft

    def _hop_starts(self, n_samples: int, Nw: int, Nh: int) -> np.ndarray:
        """Sample indices where each analysis window starts (causal: window ends at hop time)."""
        if n_samples < Nw:
            return np.zeros(0, dtype=int)
        last_start = n_samples - Nw
        return np.arange(0, last_start + 1, Nh, dtype=int)

    def compute_rolling_fft(
        self,
        mic_index: int = 0,
        window_s: float = 4.0,
        hop_ms: float = 100.0,
    ) -> List[FrequencyFrame]:
        """Sliding-window FFT magnitude spectra over one mic's recording."""
        if not self._microphones:
            raise RuntimeError("No microphones; call run_recording() first.")
        if mic_index < 0 or mic_index >= len(self._microphones):
            raise IndexError(f"mic_index {mic_index} out of range.")
        sig = self._microphones[mic_index].track.astype(float)
        if sig.size == 0:
            raise RuntimeError("Reference mic track is empty; call run_recording() first.")

        Nw, nfft = self._stft_params(self.sample_rate, window_s)
        Nh = max(1, int(round(self.sample_rate * hop_ms / 1000.0)))
        win = np.hanning(Nw).astype(float)
        starts = self._hop_starts(sig.size, Nw, Nh)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sample_rate)

        frames: List[FrequencyFrame] = []
        for s in starts:
            seg = sig[s : s + Nw] * win
            mag = np.abs(np.fft.rfft(seg, n=nfft))
            t_centre = (s + Nw / 2.0) / self.sample_rate
            frames.append(FrequencyFrame(time_s=float(t_centre), freqs=freqs, magnitudes=mag.astype(np.float32)))
        return frames

    def compute_short_stft_all_mics(
        self,
        window_s: float = 0.1,
        hop_ms: float = 25.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute complex STFT over all mics for DOA / per-source MVDR.

        Returns
        -------
        X : (T_frames, M, F) complex64
        times_s : (T_frames,) center time of each window in seconds
        freqs_hz : (F,) frequency axis
        """
        if not self._microphones:
            raise RuntimeError("No microphones; call run_recording() first.")
        tracks = [m.track.astype(float) for m in self._microphones]
        max_len = max(len(t) for t in tracks)
        if max_len == 0:
            raise RuntimeError("All mic tracks empty; call run_recording() first.")
        # Pad to common length.
        X_time = np.stack([np.pad(t, (0, max_len - len(t))) for t in tracks], axis=0)  # (M, N)

        Nw, nfft = self._stft_params(self.sample_rate, window_s)
        Nh = max(1, int(round(self.sample_rate * hop_ms / 1000.0)))
        win = np.hanning(Nw).astype(float)
        starts = self._hop_starts(max_len, Nw, Nh)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sample_rate)

        X = np.empty((starts.size, X_time.shape[0], freqs.size), dtype=np.complex64)
        for i, s in enumerate(starts):
            seg = X_time[:, s : s + Nw] * win[None, :]
            X[i] = np.fft.rfft(seg, n=nfft, axis=1).astype(np.complex64)
        times = (starts + Nw / 2.0) / self.sample_rate
        return X, times.astype(float), freqs.astype(float)

    def find_peak_frequencies_in_frame(
        self,
        frame: FrequencyFrame,
        n_peaks: int = 20,
        min_freq_hz: float = 1000.0,
        max_freq_hz: float = 20000.0,
    ) -> List[Tuple[float, float]]:
        freqs = frame.freqs
        mag = frame.magnitudes
        band = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
        if not np.any(band):
            return []
        f_band = freqs[band]
        m_band = mag[band]

        # Noise-floor-aware prominence: median + a small margin.
        noise_floor = float(np.median(m_band))
        peak_idx, props = find_peaks(
            m_band,
            height=noise_floor * 2.0,
            prominence=noise_floor * 1.0,
        )
        if peak_idx.size == 0:
            return []
        # Sort by magnitude desc and keep top n.
        order = np.argsort(m_band[peak_idx])[::-1][:n_peaks]
        sel = peak_idx[order]
        return [(float(f_band[i]), float(m_band[i])) for i in sel]

    @staticmethod
    def whiten_spectrogram(
        mag: np.ndarray,
        freqs_hz: np.ndarray,
        min_freq_hz: float = 1000.0,
        max_freq_hz: float = 20000.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Per-bin spectral whitening across time.

        For each frequency bin, divide the time series by its time-median.
        This equalizes the noise floor so quiet sources are not masked by
        louder ones, regardless of their absolute energy.

        Parameters
        ----------
        mag : (T, F) magnitude spectrogram on the reference mic.
        freqs_hz : (F,) frequency axis.

        Returns
        -------
        (T, F) whitened magnitude. Out-of-band bins are zeroed.
        """
        per_bin_floor = np.median(mag, axis=0)            # (F,)
        # Avoid division by tiny floor in dead bins.
        per_bin_floor = np.maximum(per_bin_floor, eps + 0.05 * np.median(per_bin_floor))
        whitened = mag / per_bin_floor                     # (T, F)
        # Mask out-of-band.
        band = (freqs_hz >= min_freq_hz) & (freqs_hz <= max_freq_hz)
        whitened = whitened * band[None, :]
        return whitened

    @staticmethod
    def find_peaks_whitened_frame(
        whitened_row: np.ndarray,
        freqs_hz: np.ndarray,
        n_peaks: int = 20,
        min_height: float = 4.0,
        min_prominence: float = 2.0,
        raw_row: Optional[np.ndarray] = None,
        min_absolute_mag: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Pick peaks from one row of a whitened spectrogram.

        Heights/prominences are unitless (multiples of per-bin median noise).
        If `raw_row` and `min_absolute_mag` are given, peaks whose RAW
        magnitude is below `min_absolute_mag` are discarded -- this is the
        absolute-energy gate that keeps quiet-frame whitened noise from
        becoming a 'peak'. Up to `n_peaks` peaks are returned, but if fewer
        survive both gates, only those are returned.

        Returns list of (freq_hz, whitened_magnitude) sorted by whitened mag.
        """
        if whitened_row.size == 0 or not np.any(whitened_row > 0):
            return []
        peak_idx, _ = find_peaks(
            whitened_row,
            height=min_height,
            prominence=min_prominence,
        )
        if peak_idx.size == 0:
            return []
        if raw_row is not None and min_absolute_mag is not None:
            keep = raw_row[peak_idx] >= min_absolute_mag
            peak_idx = peak_idx[keep]
            if peak_idx.size == 0:
                return []
        order = np.argsort(whitened_row[peak_idx])[::-1][:n_peaks]
        sel = peak_idx[order]
        return [(float(freqs_hz[i]), float(whitened_row[i])) for i in sel]

    def per_source_rms_envelopes(
        self,
        times_s: np.ndarray,
        window_s: float = 0.1,
    ) -> List[Tuple[str, np.ndarray]]:
        """Per-source RMS envelopes sampled at `times_s` (using each source's
        original audio data, not the simulation results).

        Returns list of (display_name, envelope_normalised_0_1) tuples in the
        same order as `self._sources`.
        """
        out: List[Tuple[str, np.ndarray]] = []
        sr = self.sample_rate
        half = max(int(round(window_s * sr * 0.5)), 1)
        for src in self._sources:
            data = src.data_resampled.astype(np.float32)
            if data.size == 0:
                env = np.zeros_like(times_s, dtype=np.float32)
            else:
                # Precompute squared-signal cumulative sum for fast RMS windows.
                p2 = np.concatenate([[0.0], np.cumsum(data * data, dtype=np.float64)])
                centres = np.clip((times_s * sr).astype(int), 0, data.size - 1)
                lo = np.clip(centres - half, 0, data.size)
                hi = np.clip(centres + half, 0, data.size)
                ms = (p2[hi] - p2[lo]) / np.maximum(hi - lo, 1)
                env = np.sqrt(np.maximum(ms, 0.0)).astype(np.float32)
                m = float(env.max())
                if m > 0:
                    env = env / m
            name = Path(src.wav_file_name).stem
            # Wav files follow the convention "XC123 - Bird Name - Species".
            # Split on " - " (with spaces) so hyphenated names like
            # "White-necked Raven" stay intact.
            parts = [p.strip() for p in name.split(" - ")]
            if len(parts) >= 2:
                friendly = parts[1] or name
            else:
                friendly = name
            out.append((friendly, env))
        return out

    # ---- DOA helpers ----
    def _mic_pos_array(self) -> np.ndarray:
        return np.stack([m.as_array() for m in self._microphones], axis=0)

    def _peak_snr_at(
        self,
        X_short: np.ndarray,
        times_short: np.ndarray,
        freqs_short: np.ndarray,
        t_centre_s: float,
        freq_hz: float,
        ref_mic_index: int = 0,
        min_freq_hz: float = 1000.0,
        max_freq_hz: float = 20000.0,
    ) -> tuple[int, int, float]:
        """Locate the nearest short-STFT (time, freq) bin and return the SNR
        of that bin on the reference mic relative to the in-band noise floor.

        Noise floor is the 25th percentile of magnitude across the bird band
        in the same frame. Using a wide reference avoids the trap where nearby
        bins contain the same call's other harmonics.
        Returns (ti, fk, snr).
        """
        ti = int(np.argmin(np.abs(times_short - t_centre_s)))
        fk = int(np.argmin(np.abs(freqs_short - freq_hz)))
        ref_mag = np.abs(X_short[ti, ref_mic_index, :])
        band = (freqs_short >= min_freq_hz) & (freqs_short <= max_freq_hz)
        in_band = ref_mag[band]
        if in_band.size == 0:
            return ti, fk, 0.0
        nf = float(np.percentile(in_band, 25) + 1e-12)
        snr = float(ref_mag[fk] / nf)
        return ti, fk, snr

    def _bartlett_spectrum(
        self,
        snapshot_x: np.ndarray,
        freq_hz: float,
        az_deg: np.ndarray,
        el_deg: np.ndarray,
        c: float,
    ) -> np.ndarray:
        """Compute Bartlett (delay-and-sum) power spectrum |a^H x|^2 on (A, E) grid."""
        mic_pos = self._mic_pos_array()
        AZ, EL = np.meshgrid(np.asarray(az_deg, dtype=float),
                             np.asarray(el_deg, dtype=float), indexing="ij")
        A_grid = steering_matrix(mic_pos, np.array([freq_hz]), AZ, EL, c=c)[..., 0, :]  # (A,E,M)
        return np.abs(np.einsum("aem,m->ae", np.conj(A_grid), snapshot_x)) ** 2

    def localize_frequency(
        self,
        freq_hz: float,
        X_short: np.ndarray,
        times_short: np.ndarray,
        freqs_short: np.ndarray,
        t_centre_s: float,
        coarse_az_step_deg: float = 10.0,
        coarse_el_step_deg: float = 15.0,
        c: float = SPEED_OF_SOUND,
        ref_mic_index: int = 0,
        min_snr: float = 3.0,
        min_freq_hz: float = 1000.0,
        max_freq_hz: float = 20000.0,
        **_unused,
    ) -> Optional[SourceDirection]:
        """Bartlett DOA at one (freq, time) using a single short-STFT snapshot.

        Returns ``None`` if the bin is below ``min_snr`` on the reference mic
        (the peak isn't really there at this moment).
        """
        ti, fk, snr = self._peak_snr_at(X_short, times_short, freqs_short,
                                        t_centre_s, freq_hz, ref_mic_index,
                                        min_freq_hz=min_freq_hz, max_freq_hz=max_freq_hz)
        if snr < min_snr:
            return None
        x = X_short[ti, :, fk].astype(np.complex128)

        az = np.arange(-180.0, 180.0, coarse_az_step_deg)
        el = np.arange(-90.0, 90.0 + 1e-9, coarse_el_step_deg)
        P = self._bartlett_spectrum(x, freq_hz, az, el, c)
        ai, ei = np.unravel_index(int(np.argmax(P)), P.shape)
        peak = float(P[ai, ei])
        median = float(np.median(P) + 1e-12)
        # Combine spectral SNR (signal-vs-noise floor) and spatial sharpness
        # (peak-vs-median of Bartlett map) into a single confidence.
        spatial_sharpness = peak / median
        confidence = float(snr * np.sqrt(spatial_sharpness))
        return SourceDirection(
            azimuth_deg=float(az[ai]),
            elevation_deg=float(el[ei]),
            confidence=confidence,
        )

    def refine_direction(
        self,
        freq_hz: float,
        X_short: np.ndarray,
        times_short: np.ndarray,
        freqs_short: np.ndarray,
        t_centre_s: float,
        coarse: SourceDirection,
        coarse_az_step_deg: float,
        coarse_el_step_deg: float,
        fine_step_min_deg: float = 0.25,
        fine_step_max_deg: float = 3.0,
        c: float = SPEED_OF_SOUND,
        ref_mic_index: int = 0,
        **_unused,
    ) -> SourceDirection:
        """Local fine Bartlett scan around `coarse`. Step scales with wavelength/aperture."""
        mic_pos = self._mic_pos_array()
        if mic_pos.shape[0] >= 2:
            diffs = mic_pos[:, None, :] - mic_pos[None, :, :]
            D = float(np.linalg.norm(diffs, axis=-1).max())
        else:
            D = 1.0
        wavelength = c / max(freq_hz, 1.0)
        rayleigh_deg = float(np.degrees(wavelength / max(D, 1e-3)))
        step = float(np.clip(rayleigh_deg * 0.25, fine_step_min_deg, fine_step_max_deg))

        half_az = coarse_az_step_deg
        half_el = coarse_el_step_deg
        az = np.arange(coarse.azimuth_deg - half_az, coarse.azimuth_deg + half_az + 1e-9, step)
        el = np.arange(
            max(-90.0, coarse.elevation_deg - half_el),
            min(90.0, coarse.elevation_deg + half_el) + 1e-9,
            step,
        )

        ti = int(np.argmin(np.abs(times_short - t_centre_s)))
        fk = int(np.argmin(np.abs(freqs_short - freq_hz)))
        x = X_short[ti, :, fk].astype(np.complex128)
        P = self._bartlett_spectrum(x, freq_hz, az, el, c)
        ai, ei = np.unravel_index(int(np.argmax(P)), P.shape)
        return SourceDirection(
            azimuth_deg=float(az[ai]),
            elevation_deg=float(el[ei]),
            confidence=coarse.confidence,
            peak_ids=list(coarse.peak_ids),
        )

    def cluster_directions(
        self,
        directions: List[SourceDirection],
        angle_threshold_deg: float = 15.0,
    ) -> List[SourceDirection]:
        """Greedy single-link cluster by great-circle angle, weighted-mean direction."""
        if not directions:
            return []
        from .steering import unit_vector
        clusters: List[List[SourceDirection]] = []
        unit_vecs = [unit_vector(d.azimuth_deg, d.elevation_deg) for d in directions]
        cos_thr = float(np.cos(np.deg2rad(angle_threshold_deg)))

        cluster_unit_sums: List[np.ndarray] = []
        for d, u in zip(directions, unit_vecs):
            placed = False
            for ci, c_sum in enumerate(cluster_unit_sums):
                c_mean = c_sum / max(np.linalg.norm(c_sum), 1e-12)
                if float(np.dot(u, c_mean)) >= cos_thr:
                    clusters[ci].append(d)
                    cluster_unit_sums[ci] = c_sum + d.confidence * u
                    placed = True
                    break
            if not placed:
                clusters.append([d])
                cluster_unit_sums.append(d.confidence * u.copy())

        merged: List[SourceDirection] = []
        for members, c_sum in zip(clusters, cluster_unit_sums):
            n = float(np.linalg.norm(c_sum))
            if n < 1e-12:
                continue
            u = c_sum / n
            az = float(np.degrees(np.arctan2(u[1], u[0])))
            el = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
            peak_ids = []
            for m in members:
                peak_ids.extend(m.peak_ids)
            confidence = float(np.mean([m.confidence for m in members]))
            merged.append(
                SourceDirection(
                    azimuth_deg=az,
                    elevation_deg=el,
                    peak_ids=peak_ids,
                    confidence=confidence,
                )
            )
        return merged

    # ---- Visualization ----
    def plot_source_directions(
        self,
        sources_per_frame: List[List[SourceDirection]],
        times_s: List[float],
        animate: bool = False,
        output_filename: Optional[str] = None,
        audio_filename_for_mux: Optional[str] = None,
    ) -> None:
        """Static + optional animated polar plot of detected sources, with ground-truth overlay."""
        from .plotting import plot_sources_static, plot_sources_animation

        # Ground-truth source directions + friendly names for the overlay.
        times_arr = np.asarray(times_s, dtype=float)
        envelopes = self.per_source_rms_envelopes(times_arr) if times_arr.size else []
        gt_named: List[Tuple[str, float, float, np.ndarray]] = []
        for src, (name, env) in zip(self._sources, envelopes):
            try:
                az_rad, el_rad = src.azel(degrees=False)
            except ValueError:
                continue
            gt_named.append(
                (name, float(np.degrees(az_rad)), float(np.degrees(el_rad)), env)
            )

        out_dir = get_output_dir()
        if not animate:
            png = out_dir / (output_filename or "sources_timeline.png")
            plot_sources_static(sources_per_frame, times_s, gt_named, png)
            print(f"Sources timeline saved to: {png}")
        else:
            mp4 = out_dir / (output_filename or "sources_animation.mp4")
            audio_path = (out_dir / audio_filename_for_mux) if audio_filename_for_mux else None
            plot_sources_animation(
                sources_per_frame,
                times_s,
                gt_named,
                mp4,
                trail_frames=20,
                audio_path=audio_path,
            )
            print(f"Sources animation saved to: {mp4}")

    def show_scene_3d(self) -> None:
        """Display a 3D graph of microphones, sources, and ears."""
        mic_xyz = [(mic.x, mic.y, mic.z) for mic in self._microphones]
        src_xyz = [(s.x, s.y, s.z) for s in self._sources]
        left_xyz = (self.left_ear.x, self.left_ear.y, self.left_ear.z)
        right_xyz = (self.right_ear.x, self.right_ear.y, self.right_ear.z)
        plot_scene(mic_xyz, src_xyz, left_xyz, right_xyz)

    def show_scene_2d(self) -> None:
        """Display a top-down (x/y plane) view of microphones, sources, and ears."""
        from .plotting import plot_scene_2d
        mic_xyz = [(mic.x, mic.y, mic.z) for mic in self._microphones]
        src_xyz = [(s.x, s.y, s.z) for s in self._sources]
        target_xyz = [(t.x, t.y, t.z) for t in self._targets]
        left_xyz = (self.left_ear.x, self.left_ear.y, self.left_ear.z)
        right_xyz = (self.right_ear.x, self.right_ear.y, self.right_ear.z)
        plot_scene_2d(mic_xyz, src_xyz, left_xyz, right_xyz, target_xyz=target_xyz)

    def __repr__(self) -> str:
        return (f"MicrophoneArraySimulation("
                f"{len(self._microphones)} microphones, {len(self._sources)} sources, "
                f"{len(self._targets)} targets; "
                f"ears=({self.left_ear.name}, {self.right_ear.name}))")
