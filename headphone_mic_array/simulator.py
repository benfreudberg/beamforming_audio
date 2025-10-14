from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import numpy as np
from pathlib import Path

from .geometry import Node
from .audio_utils import (
    read_wav_float_mono, resample_to, apply_fractional_delay,
    get_output_dir, save_wav_float32, project_root
)
from .plotting import plot_scene
from .beamformer import (BeamformerConfig, MVDRBeamformer)

# --- Constants ---
INTERNAL_SR = 16000
SPEED_OF_SOUND = 340.0  # m/s


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
        self.wav_file_name = wav_file
        sr_in, data = read_wav_float_mono(wav_file)
        # Apply the user scale and a scale for distance. The scale for distance will be undone later by the Microphone,
        # we just want the freedom to move the source anywhere without changing the absolute volume at the origin.
        data = data * float(scale) * self.distance_to(Node(0, 0, 0))
        self.data48 = resample_to(data, sr_in, INTERNAL_SR)

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
        self.track: np.ndarray = np.zeros(0, dtype=np.float32)

    def listen_to(self, source: SoundSource) -> None:
        """Listen to a source; optional preloaded_48k bypasses disk I/O & resampling."""
        if not isinstance(source, SoundSource):
            raise TypeError("source must be an instance of SoundSource.")

        # Load source audio data and scale it for distance
        data48 = source.data48 * 1/self.distance_to(source)

        # Apply delay for distance
        delay_sec = self.distance_to(source) / SPEED_OF_SOUND
        total_delay = delay_sec * INTERNAL_SR
        delayed = apply_fractional_delay(data48, total_delay)

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
        save_wav_float32(out_path, INTERNAL_SR, self.track)
        print(f"Track saved to: {out_path}")


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


# ---------- Target class for beamforming implementation ----------

class Target(Node):
    """
    Delay-and-sum track formed by time-aligning microphones to the latest-arriving
    microphone for waves from this target point, applying per-mic scale, and summing.
    The final sum is normalized by the sum of the per-mic scales used.
    """
    def __init__(self, name: str, x: float, y: float, z: float):
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            raise TypeError("Target coordinates must be numeric.")
        super().__init__(float(x), float(y), float(z))
        self._track: np.ndarray = np.zeros(0, dtype=np.float32)
        self.name = name

    def create_track_DS(self, mic_entries: List[Microphone]) -> None:
        if not mic_entries:
            raise RuntimeError("No microphones provided to Target.create_track_DS().")

        # 1) Compute raw delays to each mic (seconds -> samples)
        delays: List[float] = []
        for mic in mic_entries:
            tau_s = self.distance_to(mic) / SPEED_OF_SOUND
            delays.append(tau_s * INTERNAL_SR)

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
        beam_former_config = BeamformerConfig(fs=INTERNAL_SR, hop_ms=3, speed_of_sound=SPEED_OF_SOUND, ema_alpha=0.9, diag_load=0.01, update_every_n_frames=1)
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
        save_wav_float32(out_path, INTERNAL_SR, self._track)
        print(f"Target track exported to: {out_path}")


# ---------- Simulation ----------

class MicrophoneArraySimulation:
    def __init__(self):
        self._microphones: List[Microphone] = []
        self._sources: List[SoundSource] = []
        self._targets: List[Target] = []
        self._cache_48k_by_path: Dict[Path, np.ndarray] = {}  # resampled (48k), unscaled

        # Internal ears (not part of microphones list)
        self.left_ear = VirtualEar("left_ear")
        self.right_ear = VirtualEar("right_ear")

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

        print(f"Applying ambient audio to ears...")
        for ear in (self.left_ear, self.right_ear):
            print(f"Processing ear '{ear.name}'...")
            for src in self._sources:
                ear.listen_to(src)
        print("Ambient audio application complete (ears).")

    def reset_recordings(self, target: Literal["ears", "mics", "all"] = "all") -> None:
        """Reset stored audio data for 'ears', 'mics', or 'all'."""
        if target not in ("ears", "mics", "all"):
            raise ValueError("target must be one of: 'ears', 'mics', 'all'")
        if target in ("ears", "all"):
            self.left_ear.track = np.zeros(0, dtype=np.float32)
            self.right_ear.track = np.zeros(0, dtype=np.float32)
            print("Reset: cleared ear recordings.")
        if target in ("mics", "all"):
            for mic in self._microphones:
                mic.track = np.zeros(0, dtype=np.float32)
            print("Reset: cleared microphone recordings.")

    def export_ears_stereo(self, filename: str = "ears_stereo.wav") -> None:
        """Export the two internal ears as a stereo WAV (float32, 48 kHz) to output/."""
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")

        left = self.left_ear.track.astype(np.float32)
        right = self.right_ear.track.astype(np.float32)
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
    def add_target(self, name: str, x: float, y: float, z: float) -> Target:
        """Create a Target at (x, y, z), build its track, store and return it."""
        t = Target(name, x, y, z)
        self._targets.append(t)
        print(f"Target created at ({x}, {y}, {z}) and track generated.")
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
            diff_samples = abs(dL_s - dR_s) * INTERNAL_SR
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

    # ---- Visualization ----
    def show_scene_3d(self) -> None:
        """Display a 3D graph of microphones, sources, and ears."""
        mic_xyz = [(mic.x, mic.y, mic.z) for mic in self._microphones]
        src_xyz = [(s.x, s.y, s.z) for s in self._sources]
        left_xyz = (self.left_ear.x, self.left_ear.y, self.left_ear.z)
        right_xyz = (self.right_ear.x, self.right_ear.y, self.right_ear.z)
        plot_scene(mic_xyz, src_xyz, left_xyz, right_xyz)

    def __repr__(self) -> str:
        return (f"MicrophoneArraySimulation("
                f"{len(self._microphones)} microphones, {len(self._sources)} sources, "
                f"{len(self._targets)} targets; "
                f"ears=({self.left_ear.name}, {self.right_ear.name}))")
