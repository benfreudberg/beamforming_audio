"""Array-geometry sweep for the bird-finder pipeline.

Builds a fresh `MicrophoneArraySimulation` for each candidate microphone
layout, runs the BirdFinderPipeline, and scores how cleanly the detected
sources line up with the known ground-truth bird positions.

Usage:
    python experiments/array_sweep.py            # sweep all layouts
    python experiments/array_sweep.py LABEL      # run a single layout
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from mic_array.audio_utils import (  # noqa: E402
    read_wav_float_mono,
    save_wav_float32,
)
from mic_array.bird_finder import BirdFinderConfig, BirdFinderPipeline  # noqa: E402
from mic_array.simulator import (  # noqa: E402
    MicrophoneArraySimulation,
    Microphone,
    SoundSource,
)
from mic_array.steering import unit_vector  # noqa: E402


SAMPLE_RATE = 48000
GOLDEN = math.radians(137.50776405)
BIRDS = BASE / "input_audio_files" / "birds"
CLIPS = BASE / "output" / "_clips"

WAV_CHICKADEE = BIRDS / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav"
WAV_QUETZAL   = BIRDS / "XC1073500 - Crested Quetzal - Pharomachrus antisianus.wav"
WAV_RAVEN     = BIRDS / "XC632143 - White-necked Raven - Corvus albicollis.wav"
WAV_ACCENTOR  = BIRDS / "XC767753 - Robin Accentor - Prunella rubeculoides.wav"
WAV_OWL       = BIRDS / "XC821441 - Great Horned Owl - Bubo virginianus.wav"

# Default scene: matches run_simulation_bird_finder.py.
DEFAULT_SCENE = [
    # (label,         wav_path,        x,     y,    z, scale)
    ("chickadee",     WAV_CHICKADEE,   10.0,   5.0,  -3.0, 0.5),
    ("quetzal",       WAV_QUETZAL,     -7.0,  10.0,   0.0, 1.0),
    ("raven",         WAV_RAVEN,        4.0,  20.0,   4.0, 0.75),
    ("accentor",      WAV_ACCENTOR,     5.0, -17.0,   4.0, 1.0),
    ("owl",           WAV_OWL,         -5.0,  17.0,   8.0, 1.0),
]


def _clip_wav(src: Path, start_s: float, dur_s: float) -> Path:
    """Cut a [start_s, start_s+dur_s] slice of src into output/_clips/. Cached."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    name = f"{src.stem}__{int(start_s*1000)}ms_{int(dur_s*1000)}ms.wav"
    out = CLIPS / name
    if out.is_file():
        return out
    sr, data = read_wav_float_mono(src)
    a = max(0, int(start_s * sr))
    b = min(data.size, int((start_s + dur_s) * sr))
    if b <= a:
        raise ValueError(f"Empty clip from {src.name}: [{start_s}, {start_s+dur_s}]s")
    save_wav_float32(out, sr, data[a:b].astype(np.float32))
    return out


def build_scenes() -> dict[str, list[tuple[str, Path, float, float, float, float]]]:
    """Return named scenes for robustness testing.

    Each scene is a list of (label, wav_path, x, y, z, scale) tuples.
    Scenes use deterministic offsets (no real randomness) so results are
    reproducible run-to-run.
    """
    rng = random.Random(20260101)

    scenes: dict[str, list] = {"default": list(DEFAULT_SCENE)}

    # --- scene_3: just 3 birds, well separated.
    scenes["sparse3"] = [
        ("chickadee",  WAV_CHICKADEE,  9.0,  6.0,  2.0,  0.6),
        ("raven",      WAV_RAVEN,     -8.0, 12.0, -2.0,  0.8),
        ("owl",        WAV_OWL,        2.0, -12.0, 5.0,  1.0),
    ]

    # --- scene_dense7: 7 birds, two of them clipped re-uses placed at
    #     different positions, plus a tight pair to test angular separation.
    chick_clip = _clip_wav(WAV_CHICKADEE, 5.0, 8.0)
    accent_clip = _clip_wav(WAV_ACCENTOR, 12.0, 8.0)
    scenes["dense7"] = [
        ("chickadee",       WAV_CHICKADEE,    11.0,   4.0, -1.0, 0.5),
        ("chickadee_2",     chick_clip,       -2.0,  10.0,  6.0, 0.7),  # re-use
        ("quetzal",         WAV_QUETZAL,      -8.0,   9.0,  1.0, 1.0),
        ("raven",           WAV_RAVEN,         5.0,  19.0,  4.0, 0.75),
        ("accentor",        WAV_ACCENTOR,      6.0, -16.0,  3.0, 1.0),
        ("accentor_2",      accent_clip,      14.0, -10.0, -2.0, 0.9),
        ("owl",             WAV_OWL,          -6.0,  15.0,  7.0, 1.0),
    ]

    # --- scene_close_pair: two sources only ~10 deg apart to test resolution.
    scenes["close_pair"] = [
        ("raven",       WAV_RAVEN,         8.0,  15.0,  3.0, 0.8),
        ("accentor",    WAV_ACCENTOR,      9.0,  17.0,  5.0, 1.0),
        ("owl",         WAV_OWL,         -10.0,   8.0,  4.0, 1.0),
    ]

    # --- scene_random6: deterministic-random distribution of 6 sources at
    #     varied azimuths and elevations.
    files = [WAV_CHICKADEE, WAV_QUETZAL, WAV_RAVEN, WAV_ACCENTOR, WAV_OWL]
    rand_birds = []
    for i in range(6):
        az = rng.uniform(-180.0, 180.0)
        el = rng.uniform(-25.0, 35.0)
        r = rng.uniform(8.0, 20.0)
        cx = r * math.cos(math.radians(el)) * math.cos(math.radians(az))
        cy = r * math.cos(math.radians(el)) * math.sin(math.radians(az))
        cz = r * math.sin(math.radians(el))
        wav = files[i % len(files)]
        rand_birds.append((f"rand_{i}", wav, cx, cy, cz, rng.uniform(0.6, 1.0)))
    scenes["random6"] = rand_birds

    return scenes


def gt_az_el(x, y, z):
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.atan2(z, math.hypot(x, y)))
    return az, el


def ground_truth_for(scene):
    """(label, az_deg, el_deg) per source in the scene."""
    out = []
    for label, _wav, x, y, z, _scale in scene:
        az, el = gt_az_el(x, y, z)
        out.append((label, az, el))
    return out


# Backwards-compat: keep GROUND_TRUTH usable for the original single-scene mode.
GROUND_TRUTH = ground_truth_for(DEFAULT_SCENE)


# ---------------------------------------------------------------------------
# Layout generators: each returns a list of (name, x, y, z) tuples.
# ---------------------------------------------------------------------------

def _log_radii(n, r_min, r_max):
    return [r_min * (r_max / r_min) ** (i / max(n - 1, 1)) for i in range(n)]


def layout_log_spiral_24_r45():
    """24-mic golden-angle log spiral in x/z, r 5mm..0.45m, +center +4y."""
    radii = _log_radii(24, 0.005, 0.45)
    mics = [("center", 0.0, 0.0, 0.0)]
    for i, r in enumerate(radii):
        th = i * GOLDEN
        mics.append((f"sp_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    for i, y in enumerate([-0.05, -0.005, 0.005, 0.05]):
        mics.append((f"y_{i}", 0.0, float(y), 0.0))
    return mics


def layout_log_spiral_28_r35():
    """Tighter spiral: 28 mics, r 4mm..0.35m, golden angle, +center +1y."""
    radii = _log_radii(28, 0.004, 0.35)
    mics = [("center", 0.0, 0.0, 0.0)]
    for i, r in enumerate(radii):
        th = i * GOLDEN
        mics.append((f"sp_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    mics.append(("y_pos", 0.0, 0.05, 0.0))
    return mics


def layout_log_spiral_28_r45():
    """Denser long spiral: 28 mics out to 0.45m."""
    radii = _log_radii(28, 0.004, 0.45)
    mics = [("center", 0.0, 0.0, 0.0)]
    for i, r in enumerate(radii):
        th = i * GOLDEN
        mics.append((f"sp_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    mics.append(("y_pos", 0.0, 0.05, 0.0))
    return mics


def layout_dual_arm_spiral():
    """Two interleaved log-spiral arms (opposite chirality), 14 mics each."""
    radii = _log_radii(14, 0.005, 0.45)
    mics = [("center", 0.0, 0.0, 0.0)]
    for i, r in enumerate(radii):
        th = i * GOLDEN
        mics.append((f"a_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    for i, r in enumerate(radii):
        th = math.pi - i * GOLDEN  # opposite arm, opposite rotation
        mics.append((f"b_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    mics.append(("y_pos", 0.0, 0.05, 0.0))
    return mics


def layout_concentric_rings():
    """run_simulation.py-style: 4 jittered rings in x/z + center + y stub.

    Mic counts per ring grow with radius so areal density is roughly uniform.
    Rings are rotated by a non-multiple offset to avoid aligned sidelobes.
    """
    rings = [
        # (n_mics, diameter_m, theta0_deg)
        (5,  0.020, 11.25),
        (6,  0.060, 0.0),
        (6,  0.180, 17.5),
        (6,  0.420, -8.0),
    ]
    mics = [("center", 0.0, 0.0, 0.0)]
    idx = 0
    for n, d, th0 in rings:
        r = d * 0.5
        for k in range(n):
            th = math.radians(th0) + 2 * math.pi * k / n
            mics.append((f"r{idx:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
            idx += 1
    for i, y in enumerate([-0.05, -0.005, 0.005, 0.05]):
        mics.append((f"y_{i}", 0.0, float(y), 0.0))
    return mics


def layout_hybrid_disc_28():
    """Hybrid log/sunflower disc: 28 mics, r_min=4mm, r_break=40mm, r_max=0.45m."""
    n = 28
    n_inner = 14
    n_outer = n - n_inner
    r_min, r_break, r_max = 0.004, 0.04, 0.45
    mics = [("center", 0.0, 0.0, 0.0)]
    for i in range(n):
        if i < n_inner:
            r = r_min * (r_break / r_min) ** (i / max(n_inner - 1, 1))
        else:
            j = i - n_inner
            t = j / max(n_outer - 1, 1)
            r = math.sqrt(r_break ** 2 + (r_max ** 2 - r_break ** 2) * t)
        th = i * GOLDEN
        mics.append((f"d_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    mics.append(("y_pos", 0.0, 0.05, 0.0))
    return mics


LAYOUTS: dict[str, Callable[[], list[tuple[str, float, float, float]]]] = {
    "log_spiral_24_r45":     layout_log_spiral_24_r45,
    "log_spiral_28_r35":     layout_log_spiral_28_r35,
    "log_spiral_28_r45":     layout_log_spiral_28_r45,
    "dual_arm_spiral":       layout_dual_arm_spiral,
    "concentric_rings":      layout_concentric_rings,
    "hybrid_disc_28":        layout_hybrid_disc_28,
}


# ---------------------------------------------------------------------------
# "Bang for buck" sweep: 8 kHz cap means lambda/2 = 21 mm, so the smallest
# baseline only needs to be <~20 mm. Try fewer mics with relaxed inner radii.
# ---------------------------------------------------------------------------

def _make_log_spiral(n_spiral: int, r_min: float, r_max: float,
                     y_stub: list[float]) -> list[tuple[str, float, float, float]]:
    radii = _log_radii(n_spiral, r_min, r_max)
    mics = [("center", 0.0, 0.0, 0.0)]
    for i, r in enumerate(radii):
        th = i * GOLDEN
        mics.append((f"sp_{i:02d}", r * math.cos(th), 0.0, r * math.sin(th)))
    for i, y in enumerate(y_stub):
        mics.append((f"y_{i}", 0.0, float(y), 0.0))
    return mics


def layout_spiral_16_rmin8():
    """16 spiral + center + 2 y-stub = 19 mics. Inner radius 8 mm."""
    return _make_log_spiral(16, 0.008, 0.45, [-0.04, 0.04])


def layout_spiral_16_rmin10():
    """16 spiral + center + 2 y-stub = 19 mics. Inner radius 10 mm (~lambda/2 at 8 kHz)."""
    return _make_log_spiral(16, 0.010, 0.45, [-0.04, 0.04])


def layout_spiral_20_rmin8():
    """20 spiral + center + 2 y-stub = 23 mics. Inner radius 8 mm."""
    return _make_log_spiral(20, 0.008, 0.45, [-0.04, 0.04])


def layout_spiral_20_rmin10():
    """20 spiral + center + 2 y-stub = 23 mics. Inner radius 10 mm."""
    return _make_log_spiral(20, 0.010, 0.45, [-0.04, 0.04])


def layout_spiral_20_rmin15():
    """20 spiral + center + 2 y-stub = 23 mics. Inner radius 15 mm (testing aliasing edge)."""
    return _make_log_spiral(20, 0.015, 0.45, [-0.04, 0.04])


def layout_spiral_24_rmin10():
    """24 spiral + center + 2 y-stub = 27 mics. Inner radius 10 mm."""
    return _make_log_spiral(24, 0.010, 0.45, [-0.04, 0.04])


def layout_spiral_12_rmin10():
    """Minimal-mic test: 12 spiral + center + 2 y-stub = 15 mics."""
    return _make_log_spiral(12, 0.010, 0.45, [-0.04, 0.04])


def layout_spiral_24_rmin8_r60():
    """24 spiral + center + 2 y-stub = 27 mics. Inner 8 mm, outer 0.60 m."""
    return _make_log_spiral(24, 0.008, 0.60, [-0.04, 0.04])


def layout_spiral_24_rmin8_r75():
    """24 spiral + center + 2 y-stub = 27 mics. Inner 8 mm, outer 0.75 m."""
    return _make_log_spiral(24, 0.008, 0.75, [-0.04, 0.04])


def layout_spiral_28_rmin8_r90():
    """28 spiral + center + 2 y-stub = 31 mics. Inner 8 mm, outer 0.90 m."""
    return _make_log_spiral(28, 0.008, 0.90, [-0.04, 0.04])


def layout_spiral_22_rmin8_r60():
    """22 spiral + center + 2 y-stub = 25 mics. Inner 8 mm, outer 0.60 m."""
    return _make_log_spiral(22, 0.008, 0.60, [-0.04, 0.04])


def layout_spiral_22_rmin8_r75():
    """22 spiral + center + 2 y-stub = 25 mics. Inner 8 mm, outer 0.75 m."""
    return _make_log_spiral(22, 0.008, 0.75, [-0.04, 0.04])


for fn in [
    layout_spiral_12_rmin10,
    layout_spiral_16_rmin8,
    layout_spiral_16_rmin10,
    layout_spiral_20_rmin8,
    layout_spiral_20_rmin10,
    layout_spiral_20_rmin15,
    layout_spiral_24_rmin10,
    layout_spiral_22_rmin8_r60,
    layout_spiral_22_rmin8_r75,
    layout_spiral_24_rmin8_r60,
    layout_spiral_24_rmin8_r75,
    layout_spiral_28_rmin8_r90,
]:
    LAYOUTS[fn.__name__.removeprefix("layout_")] = fn


# ---------------------------------------------------------------------------
# Build, run, score.
# ---------------------------------------------------------------------------

def make_config(min_freq_hz: float = 700.0):
    """Same config as run_simulation_bird_finder.py.

    The bird audio peaks all sit below ~6.5 kHz (see source_spectra.py), so we
    cap the analysis band at 8 kHz to suppress noise-driven detections.
    Lowering ``min_freq_hz`` helps the owl (~300 Hz) but admits more noise.
    """
    return BirdFinderConfig(
        min_freq_hz=min_freq_hz,
        max_freq_hz=8_000.0,
        short_window_s=0.1,
        short_hop_ms=25.0,
        n_peaks=20,
        whitened_min_height=4.0,
        whitened_min_prominence=2.0,
        abs_floor_percentile=25.0,
        abs_floor_multiple=4.0,
        freq_tolerance_hz=50.0,
        max_miss_frames=8,
        high_freq_bias=0.01,
        min_track_age_frames=2,
        angle_threshold_deg=6.0,
        source_max_miss_frames=80,
        coarse_az_step_deg=5.0,
        coarse_el_step_deg=10.0,
        fine_step_min_deg=0.25,
        fine_step_max_deg=3.0,
        min_peak_snr=4.0,
        min_doa_confidence=12.0,
        doa_decimation=4,
        min_source_lifetime_frames=80,
        max_sources_to_export=8,
        reference_mic_index=0,
    )


def build_sim(layout, scene):
    sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)
    for _label, wav, x, y, z, scale in scene:
        sim.add_sound_source(SoundSource(x, y, z, wav, scale=scale, sample_rate=SAMPLE_RATE))
    for name, x, y, z in layout:
        sim.add_microphone(Microphone(name, float(x), float(y), float(z), sample_rate=SAMPLE_RATE))
    return sim


def angular_distance(az1, el1, az2, el2):
    u = unit_vector(az1, el1)
    v = unit_vector(az2, el2)
    d = float(np.clip(float(np.dot(u, v)), -1.0, 1.0))
    return math.degrees(math.acos(d))


def max_baseline(layout):
    P = np.array([[x, y, z] for _, x, y, z in layout])
    if len(P) < 2:
        return 0.0
    diffs = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diffs ** 2, axis=-1))
    return float(d.max())


def score_run(pipeline: BirdFinderPipeline, ground_truth, match_tol_deg: float = 12.0):
    """Compute summary metrics from a finished pipeline run against ``ground_truth``."""
    cfg = pipeline.config
    sighting_counts = defaultdict(int)
    sums = defaultdict(lambda: np.zeros(3))
    for srcs in pipeline.sources_per_frame:
        for s in srcs:
            sid = s.source_id
            if sid is None:
                continue
            sighting_counts[sid] += 1
            sums[sid] += unit_vector(s.azimuth_deg, s.elevation_deg)
    kept_ids = [sid for sid, n in sighting_counts.items()
                if n >= cfg.min_plot_lifetime_frames]

    # Mean direction per kept source.
    src_dirs = {}
    for sid in kept_ids:
        v = sums[sid] / max(sighting_counts[sid], 1)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        v /= n
        az = math.degrees(math.atan2(v[1], v[0]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
        src_dirs[sid] = (az, el, sighting_counts[sid])

    # Match each kept source to nearest GT.
    n_gt = len(ground_truth)
    gt_to_best = {i: None for i in range(n_gt)}
    spurious = []
    per_source_assign = {}
    for sid, (az, el, n) in src_dirs.items():
        best_i, best_err = -1, float("inf")
        for i, (_, gt_az, gt_el) in enumerate(ground_truth):
            err = angular_distance(az, el, gt_az, gt_el)
            if err < best_err:
                best_err, best_i = err, i
        if best_err <= match_tol_deg:
            per_source_assign[sid] = (best_i, best_err, n)
            cur = gt_to_best[best_i]
            if cur is None or n > cur[2] or (n == cur[2] and best_err < cur[1]):
                gt_to_best[best_i] = (sid, best_err, n)
        else:
            spurious.append((sid, az, el, n))

    gts_covered = sum(1 for v in gt_to_best.values() if v is not None)
    redundant = max(0, len(per_source_assign) - gts_covered)
    return {
        "kept_ids": len(kept_ids),
        "matched_ids": len(per_source_assign),
        "spurious_ids": len(spurious),
        "redundant_ids": redundant,
        "gts_covered": gts_covered,
        "n_gt": n_gt,
        "gt_to_best": gt_to_best,
        "spurious": spurious,
    }


def composite_score(metrics):
    # Per-GT coverage so different scene sizes are comparable.
    coverage = metrics["gts_covered"] / max(metrics["n_gt"], 1)
    return (
        100.0 * coverage
        - 3.0 * metrics["spurious_ids"]
        - 0.5 * metrics["redundant_ids"]
    )


def run_one(label, layout, scene_name="default", scene=None,
            min_freq_hz=700.0, verbose=True):
    if scene is None:
        scene = DEFAULT_SCENE
    gt = ground_truth_for(scene)
    if verbose:
        print("=" * 78)
        print(f"[{label}] scene={scene_name} ({len(scene)} src) | "
              f"{len(layout)} mics, max baseline = {max_baseline(layout):.3f} m | "
              f"min_freq={min_freq_hz:.0f}Hz")
        print("=" * 78)
    sim = build_sim(layout, scene)
    sim.run_recording()
    pipe = BirdFinderPipeline(sim, config=make_config(min_freq_hz=min_freq_hz))
    pipe.run()
    m = score_run(pipe, gt)
    if verbose:
        print(f"  kept_ids = {m['kept_ids']:3d}  matched = {m['matched_ids']:3d}  "
              f"spurious = {m['spurious_ids']:3d}  redundant = {m['redundant_ids']:3d}  "
              f"GTs_covered = {m['gts_covered']}/{m['n_gt']}")
        for i, (gt_name, gt_az, gt_el) in enumerate(gt):
            best = m["gt_to_best"][i]
            if best is None:
                print(f"    GT {gt_name:<28s} ({gt_az:+7.1f}, {gt_el:+6.1f}) -- MISSED")
            else:
                sid, err, n = best
                print(f"    GT {gt_name:<28s} ({gt_az:+7.1f}, {gt_el:+6.1f}) "
                      f"-> sid={sid} err={err:5.2f} deg n_frames={n}")
        if m["spurious"]:
            print("  spurious clusters:")
            for sid, az, el, n in sorted(m["spurious"], key=lambda x: -x[3])[:6]:
                print(f"    sid={sid:3d} ({az:+7.1f}, {el:+6.1f}) n={n}")
    score = composite_score(m)
    if verbose:
        print(f"  composite_score = {score:+.1f}")
    return label, m, score


def main():
    args = sys.argv[1:]
    robust = False
    if args and args[0] == "--robust":
        robust = True
        args = args[1:]
    min_freq = 700.0
    if args and args[0].startswith("--min_freq="):
        min_freq = float(args[0].split("=", 1)[1])
        args = args[1:]
    todo = args if args else list(LAYOUTS.keys())

    if robust:
        scenes = build_scenes()
        rows = []  # (label, mics, max_b, per_scene_scores, mean_score)
        for lbl in todo:
            if lbl not in LAYOUTS:
                print(f"unknown layout {lbl}; choices: {list(LAYOUTS)}")
                continue
            layout = LAYOUTS[lbl]()
            print("#" * 78)
            print(f"# Robust sweep [{lbl}]: {len(layout)} mics, "
                  f"max_b={max_baseline(layout):.3f} m, min_freq={min_freq}")
            print("#" * 78)
            scene_scores = {}
            for sname, scene in scenes.items():
                _, m, score = run_one(lbl, layout,
                                      scene_name=sname, scene=scene,
                                      min_freq_hz=min_freq, verbose=True)
                scene_scores[sname] = (score, m)
            mean_score = float(np.mean([s for s, _ in scene_scores.values()]))
            min_score = float(np.min([s for s, _ in scene_scores.values()]))
            rows.append((lbl, len(layout), max_baseline(layout),
                         scene_scores, mean_score, min_score))
            print(f"  >>> {lbl}: mean={mean_score:+.1f} min={min_score:+.1f}")
            print()

        print()
        print("ROBUSTNESS SUMMARY")
        scene_names = list(build_scenes().keys())
        header_scenes = "  ".join(f"{s[:9]:>9}" for s in scene_names)
        print(f"{'layout':<28s} {'mics':>4} {'max_b':>6}  "
              f"{header_scenes}  {'mean':>7}  {'min':>7}")
        for lbl, n_mics, max_b, scene_scores, mean_s, min_s in rows:
            sc_str = "  ".join(f"{scene_scores[s][0]:>+9.1f}" for s in scene_names)
            print(f"{lbl:<28s} {n_mics:>4d} {max_b:>6.3f}  "
                  f"{sc_str}  {mean_s:>+7.1f}  {min_s:>+7.1f}")
        return

    results = []
    for lbl in todo:
        if lbl not in LAYOUTS:
            print(f"unknown layout {lbl}; choices: {list(LAYOUTS)}")
            continue
        results.append(run_one(lbl, LAYOUTS[lbl](), min_freq_hz=min_freq))

    print()
    print("SUMMARY")
    print(f"{'layout':<28s} {'mics':>4} {'max_b':>7} {'kept':>5} {'match':>6} "
          f"{'spur':>5} {'redun':>6} {'GTs':>5} {'score':>7}")
    for lbl, m, score in results:
        layout = LAYOUTS[lbl]()
        print(f"{lbl:<28s} {len(layout):>4d} {max_baseline(layout):>7.3f} "
              f"{m['kept_ids']:>5d} {m['matched_ids']:>6d} {m['spurious_ids']:>5d} "
              f"{m['redundant_ids']:>6d} {m['gts_covered']:>2d}/{m['n_gt']:<2d} {score:>+7.1f}")


if __name__ == "__main__":
    main()
