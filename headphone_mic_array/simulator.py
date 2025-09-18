from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import numpy as np
from pathlib import Path

from .geometry import Node
from .audio_utils import (
    INTERNAL_SR, SPEED_OF_SOUND,
    read_wav_float_mono, resample_to, apply_fractional_delay,
    get_output_dir, save_wav_float32, project_root
)
from .plotting import plot_scene


# ---------- Core Entities ----------

class SoundSource(Node):
    def __init__(self, x: float, y: float, z: float, wav_file: str | Path, scale: float = 1.0):
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
        self.wav_file: Path = wav_file
        self.scale = float(scale)

    def __repr__(self) -> str:
        return f"SoundSource(pos=({self.x}, {self.y}, {self.z}), file='{self.wav_file.name}', scale={self.scale})"


class Microphone(Node):
    def __init__(self, name: str, x: float, y: float, z: float):
        if not isinstance(name, str):
            raise TypeError("Microphone name must be a string.")
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Coordinates must be numeric.")
        super().__init__(float(x), float(y), float(z))
        self.name = name
        self.recording: np.ndarray = np.zeros(0, dtype=np.float32)

    def listen_to(self, source: SoundSource, preloaded_48k: Optional[np.ndarray] = None) -> None:
        """Listen to a source; optional preloaded_48k bypasses disk I/O & resampling."""
        if not isinstance(source, SoundSource):
            raise TypeError("source must be an instance of SoundSource.")

        # Distance-based delay
        delay_sec = self.distance_to(source) / SPEED_OF_SOUND

        # Load/normalize/resample or use cache
        if preloaded_48k is None:
            sr_in, data = read_wav_float_mono(source.wav_file)
            if source.scale != 1.0:
                data = data * source.scale
            data48 = resample_to(data, sr_in, INTERNAL_SR)
        else:
            data48 = preloaded_48k.astype(np.float32, copy=True)
            if source.scale != 1.0:
                data48 *= source.scale

        # Apply total delay (fractional ok)
        total_delay = delay_sec * INTERNAL_SR
        delayed = apply_fractional_delay(data48, total_delay)

        # Mix (sum) into this mic's recording
        max_len = max(len(self.recording), len(delayed))
        if len(self.recording) < max_len:
            self.recording = np.pad(self.recording, (0, max_len - len(self.recording)))
        if len(delayed) < max_len:
            delayed = np.pad(delayed, (0, max_len - len(delayed)))
        self.recording += delayed

    def save_recording(self, filename: str = "recording.wav") -> None:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        out_dir = get_output_dir()
        out_path = out_dir / f"{self.name}_{filename}"
        save_wav_float32(out_path, INTERNAL_SR, self.recording)
        print(f"Recording saved to: {out_path}")


class VirtualEar(Microphone):
    def __init__(self, side: str):
        if not isinstance(side, str):
            raise TypeError("side must be a string: 'left_ear' or 'right_ear'.")
        s = side.strip().lower()
        if s == "left_ear":
            name, x, y, z = "left_ear", -0.07, 0.0, 0.0
        elif s == "right_ear":
            name, x, y, z = "right_ear", 0.07, 0.0, 0.0
        else:
            raise ValueError("side must be either 'left_ear' or 'right_ear'.")
        super().__init__(name=name, x=x, y=y, z=z)


@dataclass
class MicEntry:
    mic: Microphone
    scale: float = 1.0


# ---------- Delay-and-sum helper (Target) ----------

class Target(Node):
    """
    Delay-and-sum track formed by time-aligning microphones to the latest-arriving
    microphone for waves from this target point, applying per-mic scale, and summing.
    The final sum is normalized by the sum of the per-mic scales used.
    """
    def __init__(self, x: float, y: float, z: float):
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Target coordinates must be numeric.")
        super().__init__(float(x), float(y), float(z))
        self._track: np.ndarray = np.zeros(0, dtype=np.float32)

    def create_track(self, mic_entries: List[MicEntry]) -> None:
        if not mic_entries:
            raise RuntimeError("No microphones provided to Target.create_track().")

        # 1) Compute raw delays to each mic (seconds -> samples)
        delays: List[float] = []
        for entry in mic_entries:
            tau_s = self.distance_to(entry.mic) / SPEED_OF_SOUND
            delays.append(tau_s * INTERNAL_SR)

        # 2) Align: delay each mic by (max_delay - tau_i)
        max_delay = max(delays)
        delayed_signals: List[np.ndarray] = []
        for entry, tau in zip(mic_entries, delays):
            sig = entry.mic.recording.astype(np.float32, copy=True)
            extra = max_delay - tau
            sig_d = apply_fractional_delay(sig, extra)
            sig_d *= float(entry.scale)
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

        # 4) Normalize by sum of mic scaling factors
        scale_sum = float(sum(entry.scale for entry in mic_entries))
        if scale_sum > 0.0:
            summed = (summed / scale_sum).astype(np.float32)

        self._track = summed

    def export(self, filename: str = "target_track.wav") -> None:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if self._track is None or len(self._track) == 0:
            raise RuntimeError("No track available. Call create_track() first.")
        out_dir = get_output_dir()
        out_path = out_dir / filename
        save_wav_float32(out_path, INTERNAL_SR, self._track)
        print(f"Target track exported to: {out_path}")


# ---------- Simulation ----------

class MicrophoneArraySimulation:
    def __init__(self):
        self._microphones: List[MicEntry] = []
        self._sources: List[SoundSource] = []
        self._targets: List[Target] = []
        self._cache_48k_by_path: Dict[Path, np.ndarray] = {}  # resampled (48k), unscaled

        # Internal ears (not part of microphones list)
        self.left_ear = VirtualEar("left_ear")
        self.right_ear = VirtualEar("right_ear")

    # ---- Adders ----
    def add_microphone(self, mic: Microphone, scale: float = 1.0) -> None:
        if not isinstance(mic, Microphone):
            raise TypeError("mic must be an instance of Microphone.")
        if not isinstance(scale, (int, float)):
            raise TypeError("Scaling factor for microphone must be numeric.")
        if scale <= 0:
            raise ValueError("Scaling factor for microphone must be greater than 0.")
        self._microphones.append(MicEntry(mic=mic, scale=float(scale)))

    def add_sound_source(self, source: SoundSource) -> None:
        if not isinstance(source, SoundSource):
            raise TypeError("source must be an instance of SoundSource.")
        self._sources.append(source)

    # ---- Preload/cache ----
    def preload_sources(self) -> None:
        """
        Pre-read & resample all sources to 48 kHz and cache them UN-SCALED (float32).
        Scaling by source.scale will be applied at use time.
        """
        for s in self._sources:
            path = s.wav_file
            if path in self._cache_48k_by_path:
                continue
            sr_in, data = read_wav_float_mono(path)
            data48 = resample_to(data, sr_in, INTERNAL_SR)
            self._cache_48k_by_path[path] = data48
        print(f"Preloaded {len(self._cache_48k_by_path)} source(s) into cache.")

    # ---- Processing ----
    def run_recording(self) -> None:
        """Have every MICROPHONE (ears excluded) listen to every sound source."""
        if not self._microphones:
            raise RuntimeError("No microphones have been added to the simulation.")
        if not self._sources:
            raise RuntimeError("No sound sources have been added to the simulation.")

        # Use cache for efficiency
        self.preload_sources()

        print("Running recording simulation (microphones only)...")
        for entry in self._microphones:
            mic = entry.mic
            print(f"Processing microphone '{mic.name}'...")
            for src in self._sources:
                data48 = self._cache_48k_by_path[src.wav_file]
                mic.listen_to(src, preloaded_48k=data48)
                print(f"  Source '{src.wav_file.name}' (scale={src.scale}) processed for {mic.name}.")
        print("Simulation complete (microphones).")

    def apply_ambient_audio_to_ears(self, scaling: float = 1.0) -> None:
        """
        Process all sources into both ears (ears only), applying an additional
        global 'scaling' to every source (on top of each source.scale).
        Does NOT reset recordings.
        """
        if not isinstance(scaling, (int, float)):
            raise TypeError("scaling must be numeric.")
        if not self._sources:
            raise RuntimeError("No sound sources have been added to the simulation.")

        self.preload_sources()

        print(f"Applying ambient audio to ears with scaling={scaling}...")
        for ear in (self.left_ear, self.right_ear):
            print(f"Processing ear '{ear.name}'...")
            for src in self._sources:
                # Use cached 48k (unscaled) + apply global scaling here;
                # Microphone.listen_to will further apply src.scale internally.
                data48 = self._cache_48k_by_path[src.wav_file] * float(scaling)
                ear.listen_to(src, preloaded_48k=data48)
                print(f"  Source '{src.wav_file.name}' "
                      f"(src.scale={src.scale}, global={scaling}) processed for {ear.name}.")
        print("Ambient audio application complete (ears).")

    def reset_recordings(self, target: Literal["ears", "mics", "all"] = "all") -> None:
        """Reset stored audio data for 'ears', 'mics', or 'all'."""
        if target not in ("ears", "mics", "all"):
            raise ValueError("target must be one of: 'ears', 'mics', 'all'")
        if target in ("ears", "all"):
            self.left_ear.recording = np.zeros(0, dtype=np.float32)
            self.right_ear.recording = np.zeros(0, dtype=np.float32)
            print("Reset: cleared ear recordings.")
        if target in ("mics", "all"):
            for entry in self._microphones:
                entry.mic.recording = np.zeros(0, dtype=np.float32)
            print("Reset: cleared microphone recordings.")

    def export_ears_stereo(self, filename: str = "ears_stereo.wav") -> None:
        """Export the two internal ears as a stereo WAV (float32, 48 kHz) to output/."""
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")

        left = self.left_ear.recording.astype(np.float32)
        right = self.right_ear.recording.astype(np.float32)
        max_len = max(len(left), len(right))
        if len(left) < max_len:
            left = np.pad(left, (0, max_len - len(left)))
        if len(right) < max_len:
            right = np.pad(right, (0, max_len - len(right)))

        stereo = np.stack([left, right], axis=1).astype(np.float32)
        out_dir = get_output_dir()
        out_path = out_dir / filename
        save_wav_float32(out_path, INTERNAL_SR, stereo)
        print(f"Stereo ears exported to: {out_path}")

    # ---- Target API ----
    def add_target(self, x: float, y: float, z: float) -> Target:
        """Create a Target at (x, y, z), build its track, store and return it."""
        t = Target(x, y, z)
        t.create_track(self._microphones)
        self._targets.append(t)
        print(f"Target created at ({x}, {y}, {z}) and track generated.")
        return t

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
            diff_samples = abs(dL_s - dR_s) * INTERNAL_SR
            delayed = apply_fractional_delay(track, diff_samples)

            # Which ear is farther?
            left_farther = dL_s > dR_s

            # Compose additions
            add_left = delayed if left_farther else track
            add_right = track if left_farther else delayed

            # Mix-add into ears with padding
            L = self.left_ear.recording
            R = self.right_ear.recording
            max_len = max(len(L), len(R), len(add_left), len(add_right))

            if len(L) < max_len:
                L = np.pad(L, (0, max_len - len(L)))
            if len(R) < max_len:
                R = np.pad(R, (0, max_len - len(R)))
            if len(add_left) < max_len:
                add_left = np.pad(add_left, (0, max_len - len(add_left)))
            if len(add_right) < max_len:
                add_right = np.pad(add_right, (0, max_len - len(add_right)))

            self.left_ear.recording = (L + add_left).astype(np.float32)
            self.right_ear.recording = (R + add_right).astype(np.float32)

            print(f"Applied target at ({t.x}, {t.y}, {t.z}) to ears "
                  f"(Δ={diff_samples:.2f} samples; farther={'left' if left_farther else 'right'}).")

    # ---- Visualization ----
    def show_scene_3d(self) -> None:
        """Display a 3D graph of microphones, sources, and ears."""
        mic_xyz = [(e.mic.x, e.mic.y, e.mic.z) for e in self._microphones]
        src_xyz = [(s.x, s.y, s.z) for s in self._sources]
        left_xyz = (self.left_ear.x, self.left_ear.y, self.left_ear.z)
        right_xyz = (self.right_ear.x, self.right_ear.y, self.right_ear.z)
        plot_scene(mic_xyz, src_xyz, left_xyz, right_xyz)

    def __repr__(self) -> str:
        return (f"MicrophoneArraySimulation("
                f"{len(self._microphones)} microphones, {len(self._sources)} sources, "
                f"{len(self._targets)} targets; "
                f"ears=({self.left_ear.name}, {self.right_ear.name}))")
