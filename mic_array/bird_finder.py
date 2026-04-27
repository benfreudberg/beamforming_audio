"""
bird_finder.py
==============
Bird-finding pipeline that orchestrates the general-purpose analysis methods
on MicrophoneArraySimulation to locate bird sound sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_utils import get_output_dir, save_wav_float32
from .beamformer import BeamformerConfig, MVDRBeamformer
from .simulator import (
    SPEED_OF_SOUND,
    MicrophoneArraySimulation,
    PeakEntry,
    PeakTracker,
    SourceDirection,
    SourceTracker,
)


@dataclass
class BirdFinderConfig:
    """Tuning knobs for the BirdFinderPipeline."""
    # Frequency band of interest.
    min_freq_hz: float = 1000.0
    max_freq_hz: float = 20000.0
    # Short STFT (used for both peak picking AND DOA snapshots).
    short_window_s: float = 0.1
    short_hop_ms: float = 25.0
    # Whitened peak picker (per-frame on the reference mic).
    n_peaks: int = 20
    whitened_min_height: float = 4.0       # multiples of per-bin median
    whitened_min_prominence: float = 2.0
    # Absolute-energy gate. The reference mic's in-band magnitudes have an
    # overall noise floor estimated as `abs_floor_percentile` percentile of
    # all in-band magnitudes; a peak's RAW magnitude must exceed
    # `abs_floor_multiple` * that floor to count. This stops whitening from
    # promoting per-bin noise to a 'peak' during silent stretches.
    abs_floor_percentile: float = 25.0
    abs_floor_multiple: float = 4.0
    # Peak tracker.
    freq_tolerance_hz: float = 50.0
    max_miss_frames: int = 8                # ~200 ms at 25 ms hop
    high_freq_bias: float = 0.01
    min_track_age_frames: int = 2           # require this many sightings before DOA
    # Energy continuity for re-associating a peak with an existing track.
    # A candidate's magnitude must lie within a factor of ``peak_max_mag_ratio``
    # of the track's EMA magnitude. Set to ``inf`` to disable.
    peak_max_mag_ratio: float = 8.0
    peak_mag_ema_alpha: float = 0.5
    # Cluster + source tracker.
    angle_threshold_deg: float = 5.0
    source_max_miss_frames: int = 80        # ~2 s at 25 ms hop
    # DOA (Bartlett, single-snapshot per (freq, time)).
    coarse_az_step_deg: float = 5.0
    coarse_el_step_deg: float = 10.0
    fine_step_min_deg: float = 0.25
    fine_step_max_deg: float = 3.0
    # SNR + confidence gates (applied to mixed reference mic).
    min_peak_snr: float = 4.0
    min_doa_confidence: float = 12.0
    # How often to compute DOAs (every N short frames). 1 = every frame.
    doa_decimation: int = 4
    # Per-source export.
    min_source_lifetime_frames: int = 80
    max_sources_to_export: int = 8
    per_source_mvdr_hop_ms: float = 5.0
    # Plot-time lifetime gate. Only source_ids that were observed in at least
    # this many frames are drawn. Doesn't affect MVDR export.
    min_plot_lifetime_frames: int = 8
    # Misc.
    reference_mic_index: int = 0
    speed_of_sound: float = SPEED_OF_SOUND


class BirdFinderPipeline:
    """Drives the full bird-finding analysis on a completed simulation."""

    def __init__(self, sim: MicrophoneArraySimulation, config: Optional[BirdFinderConfig] = None) -> None:
        self.sim = sim
        self.config = config or BirdFinderConfig()
        self.tracked_peaks_per_frame: List[List[PeakEntry]] = []
        self.sources_per_frame: List[List[SourceDirection]] = []
        self.times_s: List[float] = []
        self.source_history: Dict[int, List[Tuple[float, float, float, float]]] = {}
        self._X_short: Optional[np.ndarray] = None
        self._times_short: Optional[np.ndarray] = None
        self._freqs_short: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        cfg = self.config
        sim = self.sim

        print(f"[BirdFinder] Short STFT on all mics "
              f"(window={cfg.short_window_s}s, hop={cfg.short_hop_ms}ms)...")
        self._X_short, self._times_short, self._freqs_short = sim.compute_short_stft_all_mics(
            window_s=cfg.short_window_s, hop_ms=cfg.short_hop_ms
        )
        T = self._X_short.shape[0]
        ref = cfg.reference_mic_index
        # Whitened reference-mic spectrogram for peak picking. Whitening is
        # done across the whole recording here (not streaming) — for a future
        # real-time port, replace with a causal running-median estimate per bin.
        ref_mag = np.abs(self._X_short[:, ref, :])
        whitened = sim.whiten_spectrogram(
            ref_mag, self._freqs_short,
            min_freq_hz=cfg.min_freq_hz, max_freq_hz=cfg.max_freq_hz,
        )
        # Global absolute-magnitude floor for the in-band reference mic.
        band_mask = (self._freqs_short >= cfg.min_freq_hz) & (self._freqs_short <= cfg.max_freq_hz)
        in_band = ref_mag[:, band_mask]
        if in_band.size:
            abs_floor = float(np.percentile(in_band, cfg.abs_floor_percentile)) * cfg.abs_floor_multiple
        else:
            abs_floor = 0.0
        print(f"[BirdFinder] absolute-magnitude floor = {abs_floor:.3g} "
              f"({cfg.abs_floor_multiple}x p{cfg.abs_floor_percentile:.0f} of in-band ref-mic magnitude)")
        self.times_s = list(self._times_short.astype(float))

        peak_tracker = PeakTracker(
            freq_tolerance_hz=cfg.freq_tolerance_hz,
            max_miss_frames=cfg.max_miss_frames,
            high_freq_bias=cfg.high_freq_bias,
            max_mag_ratio=cfg.peak_max_mag_ratio,
            mag_ema_alpha=cfg.peak_mag_ema_alpha,
        )
        source_tracker = SourceTracker(
            angle_threshold_deg=cfg.angle_threshold_deg,
            max_miss_frames=cfg.source_max_miss_frames,
        )

        # Track-age accounting (for min_track_age_frames gate).
        track_ages: Dict[int, int] = {}

        print(f"[BirdFinder] Processing {T} short frames...")
        for fi in range(T):
            t_centre = float(self._times_short[fi])
            peaks = sim.find_peaks_whitened_frame(
                whitened[fi], self._freqs_short,
                n_peaks=cfg.n_peaks,
                min_height=cfg.whitened_min_height,
                min_prominence=cfg.whitened_min_prominence,
                raw_row=ref_mag[fi],
                min_absolute_mag=abs_floor,
            )
            tracked = peak_tracker.update(peaks)
            self.tracked_peaks_per_frame.append(list(tracked))

            # Update ages.
            current_ids = {pe.track_id for pe in tracked}
            for tid in list(track_ages):
                if tid not in current_ids:
                    track_ages.pop(tid, None)
            for pe in tracked:
                track_ages[pe.track_id] = track_ages.get(pe.track_id, 0) + 1

            # DOA only on every Nth frame to keep cost bounded.
            if cfg.doa_decimation > 1 and (fi % cfg.doa_decimation) != 0:
                self.sources_per_frame.append([])
                continue

            coarse_dirs: List[SourceDirection] = []
            for pe in tracked:
                if track_ages.get(pe.track_id, 0) < cfg.min_track_age_frames:
                    continue
                d = sim.localize_frequency(
                    freq_hz=pe.freq_hz,
                    X_short=self._X_short,
                    times_short=self._times_short,
                    freqs_short=self._freqs_short,
                    t_centre_s=t_centre,
                    coarse_az_step_deg=cfg.coarse_az_step_deg,
                    coarse_el_step_deg=cfg.coarse_el_step_deg,
                    c=cfg.speed_of_sound,
                    ref_mic_index=cfg.reference_mic_index,
                    min_snr=cfg.min_peak_snr,
                    min_freq_hz=cfg.min_freq_hz,
                    max_freq_hz=cfg.max_freq_hz,
                )
                if d is None:
                    continue
                if d.confidence < cfg.min_doa_confidence:
                    continue
                d.peak_ids = [pe.track_id]
                d.__dict__["_peak_freq_hz"] = pe.freq_hz
                coarse_dirs.append(d)

            clusters = sim.cluster_directions(coarse_dirs, angle_threshold_deg=cfg.angle_threshold_deg)

            refined: List[SourceDirection] = []
            peak_by_id = {d.peak_ids[0]: d for d in coarse_dirs}
            for c in clusters:
                if not c.peak_ids:
                    refined.append(c)
                    continue
                members = [peak_by_id[pid] for pid in c.peak_ids if pid in peak_by_id]
                if not members:
                    refined.append(c)
                    continue
                hi = max(members, key=lambda d: d.__dict__.get("_peak_freq_hz", 0.0))
                hi_freq = hi.__dict__.get("_peak_freq_hz", 0.0)
                fine = sim.refine_direction(
                    freq_hz=hi_freq,
                    X_short=self._X_short,
                    times_short=self._times_short,
                    freqs_short=self._freqs_short,
                    t_centre_s=t_centre,
                    coarse=c,
                    coarse_az_step_deg=cfg.coarse_az_step_deg,
                    coarse_el_step_deg=cfg.coarse_el_step_deg,
                    fine_step_min_deg=cfg.fine_step_min_deg,
                    fine_step_max_deg=cfg.fine_step_max_deg,
                    c=cfg.speed_of_sound,
                    ref_mic_index=cfg.reference_mic_index,
                )
                fine.peak_ids = list(c.peak_ids)
                refined.append(fine)

            tracked_sources = source_tracker.update(refined)
            self.sources_per_frame.append(list(tracked_sources))
            for s in tracked_sources:
                self.source_history.setdefault(s.source_id, []).append(
                    (t_centre, s.azimuth_deg, s.elevation_deg, s.confidence)
                )

            if (fi + 1) % 100 == 0 or fi == T - 1:
                print(f"  frame {fi+1}/{T}: "
                      f"{len(peaks)} peaks -> {len(tracked)} tracks -> "
                      f"{len(refined)} sources (active IDs: {len(source_tracker.active)})")

        print("[BirdFinder] Pipeline done.")

    # ---------------------------------------------------- visualization
    def plot_sources(self, animate: bool = False, audio_filename: Optional[str] = None) -> None:
        if not self.sources_per_frame:
            raise RuntimeError("Run the pipeline before plotting.")
        cfg = self.config

        # Plot-time lifetime gate: only draw source_ids observed in at least
        # `min_plot_lifetime_frames` frames. This kills one-off ghosts that
        # would otherwise clutter the plot.
        sighting_counts: Dict[int, int] = {}
        for srcs in self.sources_per_frame:
            for s in srcs:
                sid = getattr(s, "source_id", None)
                if sid is not None:
                    sighting_counts[sid] = sighting_counts.get(sid, 0) + 1
        keep_ids = {sid for sid, n in sighting_counts.items()
                    if n >= cfg.min_plot_lifetime_frames}
        filtered = [
            [s for s in srcs if getattr(s, "source_id", None) in keep_ids]
            for srcs in self.sources_per_frame
        ]
        kept_total = sum(len(f) for f in filtered)
        raw_total = sum(len(f) for f in self.sources_per_frame)
        print(f"[BirdFinder] plot lifetime gate: kept {len(keep_ids)} source IDs "
              f"({kept_total}/{raw_total} dots) at min_plot_lifetime_frames="
              f"{cfg.min_plot_lifetime_frames}")

        self.sim.plot_source_directions(
            filtered, self.times_s, animate=False,
            output_filename="sources_timeline.png",
        )
        if animate:
            self.sim.plot_source_directions(
                filtered, self.times_s, animate=True,
                output_filename="sources_animation.mp4",
                audio_filename_for_mux=audio_filename,
            )

    # --------------------------------------------- per-source MVDR export
    def export_target_tracks(self) -> None:
        if not self.sources_per_frame:
            raise RuntimeError("Run the pipeline before exporting.")
        cfg = self.config
        sr = self.sim.sample_rate
        mics = self.sim._microphones
        if not mics:
            raise RuntimeError("Simulation has no microphones.")

        eligible = {sid: hist for sid, hist in self.source_history.items()
                    if len(hist) >= cfg.min_source_lifetime_frames}
        if not eligible:
            print("[BirdFinder] No sources met min_source_lifetime_frames; nothing to export.")
            return
        # Cap to the most persistent N sources to keep export time bounded.
        eligible = dict(
            sorted(eligible.items(), key=lambda kv: -len(kv[1]))[: cfg.max_sources_to_export]
        )

        tracks = [m.track.astype(np.float32) for m in mics]
        max_len = max(len(t) for t in tracks)
        if max_len == 0:
            raise RuntimeError("All mic tracks empty.")
        x_all = np.stack([np.pad(t, (0, max_len - len(t))) for t in tracks], axis=0)

        out_dir = get_output_dir()
        for sid, hist in sorted(eligible.items()):
            print(f"[BirdFinder] Exporting source {sid} ({len(hist)} sightings)...")
            bf = MVDRBeamformer(
                mics,
                config=BeamformerConfig(
                    fs=sr,
                    hop_ms=cfg.per_source_mvdr_hop_ms,
                    speed_of_sound=cfg.speed_of_sound,
                    ema_alpha=0.95,
                    diag_load=0.01,
                ),
            )
            t0, az0, el0, _ = hist[0]
            bf.set_target_direction(az0, el0)

            schedule = [(int(round(t * sr)), az, el) for (t, az, el, _c) in hist]
            schedule.sort(key=lambda r: r[0])
            sched_idx = 0
            next_change = schedule[1][0] if len(schedule) > 1 else max_len + 1

            cursor = 0
            out_chunks: List[np.ndarray] = []
            while cursor < max_len:
                chunk_end = min(next_change, max_len)
                if chunk_end > cursor:
                    y = bf.process_block(x_all[:, cursor:chunk_end])
                    if y.size > 0:
                        out_chunks.append(y)
                    cursor = chunk_end
                if cursor >= next_change and sched_idx + 1 < len(schedule):
                    sched_idx += 1
                    _, az_n, el_n = schedule[sched_idx]
                    bf.set_target_direction(az_n, el_n)
                    next_change = (schedule[sched_idx + 1][0]
                                   if sched_idx + 1 < len(schedule) else max_len + 1)

            y_all = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
            out_path = out_dir / f"source_{sid:02d}_MVDR.wav"
            save_wav_float32(out_path, sr, y_all.astype(np.float32))
            print(f"  -> {out_path}")
