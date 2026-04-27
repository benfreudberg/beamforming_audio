"""Quantify DOA accuracy.

For each source individually, compute the strongest frequency at each
short-STFT frame from JUST that source's contribution at mic 0. Then
run Bartlett DOA on the MIXED short-STFT at that exact (t, f) and
measure the angular error against the source's true direction.

This isolates: "given perfect frequency-source attribution, how well
does the array spatial resolution actually do?"

Also reports the same metric using the pipeline's emitted detections,
so we can compare oracle accuracy vs. operational accuracy.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mic_array.simulator import (
    MicrophoneArraySimulation, SoundSource, Microphone, SPEED_OF_SOUND
)
from mic_array.steering import steering_matrix


SAMPLE_RATE = 48000
base_dir = Path(__file__).resolve().parent.parent
birds_dir = base_dir / "input_audio_files" / "birds"

bird_sources_def = [
    ("Mountain Chickadee", birds_dir / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav",     10.0,   5.0,  -3.0, 0.65),
    ("Crested Quetzal",    birds_dir / "XC1073500 - Crested Quetzal - Pharomachrus antisianus.wav", -7.0,  10.0,  0.0, 1.0),
    ("White-necked Raven", birds_dir / "XC632143 - White-necked Raven - Corvus albicollis.wav",     4.0,  20.0,  4.0, 0.75),
    ("Robin Accentor",     birds_dir / "XC767753 - Robin Accentor - Prunella rubeculoides.wav",     5.0, -17.0,  4.0, 1.0),
    ("Great Horned Owl",   birds_dir / "XC821441 - Great Horned Owl - Bubo virginianus.wav",       -5.0,  17.0,  8.0, 1.0),
]


def staged_arm_distances():
    tight, med, gap_s, gap_l = 0.0075, 0.0150, 0.05, 0.12
    distances, pos = [], 0.0
    for i in range(3):
        pos += tight * 1.2 ** i
        distances.append(pos)
    pos += gap_s
    for i in range(3):
        if i > 0:
            pos += med * 1.2 ** i
        distances.append(pos)
    pos += gap_l
    distances.append(pos)
    return distances


def build_sim() -> MicrophoneArraySimulation:
    sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)
    for _, wav, x, y, z, scale in bird_sources_def:
        sim.add_sound_source(SoundSource(x, y, z, wav, scale=scale, sample_rate=SAMPLE_RATE))
    sim.add_microphone(Microphone("center", 0, 0, 0, sample_rate=SAMPLE_RATE))
    arm_d = staged_arm_distances()
    for i, x in enumerate(arm_d):
        sim.add_microphone(Microphone(f"x_{i}", float(x), 0, 0, sample_rate=SAMPLE_RATE))
    for i, z in enumerate(arm_d):
        sim.add_microphone(Microphone(f"z_{i}", 0, 0, float(z), sample_rate=SAMPLE_RATE))
    for i, y in enumerate([-0.015, 0.0, 0.015]):
        sim.add_microphone(Microphone(f"y_{i}", 0.10, float(y), 0.10, sample_rate=SAMPLE_RATE))
    sim.run_recording()
    return sim


def per_source_mic0_track(sim: MicrophoneArraySimulation, src: SoundSource) -> np.ndarray:
    """Re-create just this source's contribution at mic 0 (origin)."""
    mic0 = sim._microphones[0]
    from mic_array.audio_utils import apply_fractional_delay
    data = src.data_resampled * (1.0 / mic0.distance_to(src))
    delay_samples = (mic0.distance_to(src) / SPEED_OF_SOUND) * sim.sample_rate
    return apply_fractional_delay(data, delay_samples).astype(np.float32)


def short_stft_of(track: np.ndarray, fs: int, window_s: float, hop_ms: float):
    Nw = int(round(window_s * fs))
    nfft = 1
    while nfft < Nw:
        nfft <<= 1
    H = max(1, int(round(hop_ms * 1e-3 * fs)))
    win = np.hanning(Nw).astype(np.float32)
    if track.size < Nw:
        return np.zeros((0, nfft // 2 + 1), dtype=np.complex64), np.zeros(0), np.zeros(0)
    starts = np.arange(0, track.size - Nw + 1, H)
    Nf = nfft // 2 + 1
    X = np.zeros((starts.size, Nf), dtype=np.complex64)
    for i, s in enumerate(starts):
        seg = track[s:s + Nw] * win
        X[i] = np.fft.rfft(seg, n=nfft)
    times = (starts + Nw / 2.0) / fs
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return X, times, freqs


def gc_angle_deg(az1, el1, az2, el2):
    u1 = np.array([np.cos(np.deg2rad(el1))*np.cos(np.deg2rad(az1)),
                   np.cos(np.deg2rad(el1))*np.sin(np.deg2rad(az1)),
                   np.sin(np.deg2rad(el1))])
    u2 = np.array([np.cos(np.deg2rad(el2))*np.cos(np.deg2rad(az2)),
                   np.cos(np.deg2rad(el2))*np.sin(np.deg2rad(az2)),
                   np.sin(np.deg2rad(el2))])
    return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))))


def main():
    print("Building simulation...")
    sim = build_sim()

    short_window_s = 0.1
    short_hop_ms = 25.0
    print(f"Short STFT (window={short_window_s}s, hop={short_hop_ms}ms) on all mics...")
    X_short, times_short, freqs_short = sim.compute_short_stft_all_mics(
        window_s=short_window_s, hop_ms=short_hop_ms,
    )

    # Per-source mic-0 spectrograms (perfect attribution).
    print("Computing per-source mic-0 STFTs...")
    src_specs = []  # list of (name, (az, el), |X|^2 magnitude on (T, F))
    for (name, _wav, x, y, z, _scale), src in zip(bird_sources_def, sim._sources):
        track = per_source_mic0_track(sim, src)
        Xs, ts, fs_ = short_stft_of(track, sim.sample_rate, short_window_s, short_hop_ms)
        # Align to times_short (should match for matching params).
        n = min(Xs.shape[0], X_short.shape[0])
        mag = (np.abs(Xs[:n]) ** 2).astype(np.float32)
        az_rad, el_rad = src.azel(degrees=False)
        gt = (float(np.degrees(az_rad)), float(np.degrees(el_rad)))
        src_specs.append((name, gt, mag))
        print(f"  {name}: gt=({gt[0]:.1f}, {gt[1]:.1f})  mic0_total_power={float(mag.sum()):.3g}")

    # Stack to (S, T, F) and compute per-(t, f) dominant source.
    S = len(src_specs)
    T = min(s[2].shape[0] for s in src_specs)
    F = src_specs[0][2].shape[1]
    P = np.stack([s[2][:T] for s in src_specs], axis=0)  # (S, T, F) per-source power at mic 0
    dominant = np.argmax(P, axis=0)                      # (T, F)
    src_max = np.max(P, axis=0)                          # (T, F)

    # Power of the actual mixed mic-0 recording at the same bins.
    mixed_mic0 = np.abs(X_short[:T, 0, :]) ** 2          # (T, F)

    # Dominance margin: chosen source must own most of the mixed-signal energy.
    share = src_max / np.maximum(mixed_mic0, 1e-12)

    # Absolute energy gate: mixed power must be well above its own band median
    # (otherwise it's just silence and DOA is noise-driven).
    band = (freqs_short >= 1000.0) & (freqs_short <= 20000.0)
    band_mask2d = np.broadcast_to(band[None, :], (T, F))
    band_median = float(np.median(mixed_mic0[:, band]))
    activity = mixed_mic0 > 25.0 * band_median           # mixed signal clearly above noise floor

    valid = band_mask2d & activity & (share > 0.8)       # this source owns >80% of energy
    print(f"Eligible (t, f) bins: {int(valid.sum())} / {T*F}  (band_median={band_median:.3g})")

    # Bartlett DOA at each eligible bin (subsample to keep cost bounded).
    mic_pos = np.stack([m.as_array() for m in sim._microphones], axis=0)
    az_grid = np.arange(-180.0, 180.0, 5.0)
    el_grid = np.arange(-50.0, 50.001, 5.0)
    AZ, EL = np.meshgrid(az_grid, el_grid, indexing="ij")

    # Subsample valid bins (cap per source).
    rng = np.random.default_rng(0)
    per_source_samples = 2000
    errors = {i: [] for i in range(S)}
    error_freqs = {i: [] for i in range(S)}
    for sid in range(S):
        idx = np.argwhere(valid & (dominant == sid))
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > per_source_samples:
            sel = rng.choice(idx.shape[0], per_source_samples, replace=False)
            idx = idx[sel]
        gt_az, gt_el = src_specs[sid][1]
        for ti, fk in idx:
            ti = int(ti); fk = int(fk)
            x = X_short[ti, :, fk].astype(np.complex128)
            f = float(freqs_short[fk])
            A = steering_matrix(mic_pos, np.array([f]), AZ, EL, c=SPEED_OF_SOUND)[..., 0, :]
            P_b = np.abs(np.einsum("aem,m->ae", np.conj(A), x)) ** 2
            ai, ei = np.unravel_index(int(np.argmax(P_b)), P_b.shape)
            err = gc_angle_deg(az_grid[ai], el_grid[ei], gt_az, gt_el)
            errors[sid].append(err)
            error_freqs[sid].append(f)

    print("\n=== ORACLE Bartlett DOA error (perfect freq-source attribution) ===")
    print(f"{'Source':<25} {'count':>6} {'median':>8} {'mean':>8} {'p90':>8} {'p99':>8}")
    for sid, (name, gt, _) in enumerate(src_specs):
        e = np.array(errors[sid])
        if e.size == 0:
            print(f"{name:<25}    n=0")
            continue
        print(f"{name:<25} {e.size:>6d} {np.median(e):>7.2f}° {np.mean(e):>7.2f}° "
              f"{np.percentile(e, 90):>7.2f}° {np.percentile(e, 99):>7.2f}°")

    print("\n=== Median DOA error per source per frequency band ===")
    bands = [(1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000)]
    print(f"{'Source':<25} " + "  ".join(f"{lo/1000:.0f}-{hi/1000:.0f}kHz".rjust(11) for lo, hi in bands))
    for sid, (name, gt, _) in enumerate(src_specs):
        e = np.array(errors[sid])
        f = np.array(error_freqs[sid])
        if e.size == 0:
            print(f"{name:<25} (no data)")
            continue
        cells = []
        for lo, hi in bands:
            m = (f >= lo) & (f < hi)
            if m.sum() < 10:
                cells.append(f"  n={m.sum():<3d}    ")
            else:
                cells.append(f"{np.median(e[m]):>5.2f}° n={m.sum():<3d}")
        print(f"{name:<25} " + "  ".join(c.rjust(11) for c in cells))

    # --- Pipeline accuracy at different window_s (peak-picking long-STFT) ---
    print("\n=== Pipeline accuracy vs. peak-picker window_s ===")
    from mic_array.bird_finder import BirdFinderPipeline, BirdFinderConfig

    for win_s in (4.0, 1.0, 0.25, 0.1):
        # Suppress per-frame chatter by silencing prints.
        import io, contextlib
        cfg = BirdFinderConfig(
            window_s=win_s, hop_ms=100.0,
            n_peaks=20, min_freq_hz=1000, max_freq_hz=20000,
            angle_threshold_deg=15.0, source_max_miss_frames=80,
            short_window_s=0.1, short_hop_ms=25.0,
            coarse_az_step_deg=5.0, coarse_el_step_deg=10.0,
            fine_step_min_deg=0.25, fine_step_max_deg=3.0,
            min_peak_snr=30.0, min_doa_confidence=80.0,
            reference_mic_index=0,
        )
        pipe = BirdFinderPipeline(sim, cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe.run()
        # Aggregate detections vs ground truth.
        per_src = {sid: [] for sid in range(S)}
        total = 0
        for srcs in pipe.sources_per_frame:
            for s in srcs:
                total += 1
                # Assign to closest GT.
                best_sid, best_err = 0, 1e9
                for sid, (_, gt, _) in enumerate(src_specs):
                    err = gc_angle_deg(s.azimuth_deg, s.elevation_deg, gt[0], gt[1])
                    if err < best_err:
                        best_err = err; best_sid = sid
                per_src[best_sid].append(best_err)
        print(f"\n  window_s = {win_s}s  (total detections: {total})")
        print(f"  {'Source':<25} {'count':>6} {'median':>8} {'mean':>8} {'p90':>8}")
        for sid, (name, _, _) in enumerate(src_specs):
            e = np.array(per_src[sid])
            if e.size == 0:
                print(f"  {name:<25}    n=0")
                continue
            print(f"  {name:<25} {e.size:>6d} {np.median(e):>7.2f}° {np.mean(e):>7.2f}° {np.percentile(e, 90):>7.2f}°")


if __name__ == "__main__":
    main()
