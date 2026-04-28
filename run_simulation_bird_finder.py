"""
run_simulation_bird_finder.py
=============================
Entry point for the bird-finding application.

This script sets up a microphone array at 48 kHz, loads bird audio files as
sound sources, runs the acoustic simulation, and then hands off to the
BirdFinderPipeline for spectral analysis, frequency tracking, direction-of-
arrival estimation, clustering, and visualisation.

Differences from run_simulation.py (headphones application)
------------------------------------------------------------
* sample_rate=48000 throughout (higher spatial resolution for bird calls).
* Microphone array geometry tuned for the higher frequency range.
* No "targets" are added up front; they emerge from the BirdFinderPipeline.
* The BirdFinderPipeline drives analysis rather than a fixed list of Targets.

Usage
-----
    python run_simulation_bird_finder.py

Outputs are written to output/ (or the path set in the OUTPUT_DIR env var).
"""

from pathlib import Path
import math

from mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
)
from mic_array.bird_finder import BirdFinderConfig, BirdFinderPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000          # internal sample rate for this application (Hz)
SPEED_OF_SOUND = 340.0    # m/s

# ---------------------------------------------------------------------------
# Audio files
# ---------------------------------------------------------------------------

base_dir  = Path(__file__).resolve().parent
birds_dir = base_dir / "input_audio_files" / "birds"

# Place each bird at a different (x, y, z) position, in metres.
bird_sources = [
    # (wav_file,                                           x,     y,    z, scale)
    (birds_dir / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav",   10.0,   5.0, -3.0, 0.5),
    (birds_dir / "XC877052 - Bald Eagle - Haliaeetus leucocephalus.wav",   -7.0,  10.0,  0.0, 1.5),
    (birds_dir / "XC632143 - White-necked Raven - Corvus albicollis.wav",   4.0,  20.0,  4.0, 0.75),
    (birds_dir / "XC767753 - Robin Accentor - Prunella rubeculoides.wav",   5.0, -17.0,  4.0, 1.0),
    (birds_dir / "XC821441 - Great Horned Owl - Bubo virginianus.wav",     -5.0,  17.0,  8.0, 1.0),
]

# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------
#
# Robustness winner of experiments/array_sweep.py --robust: spiral_22_rmin8_r60
# (best mean and best worst-case score across 5 distinct bird scenes):
#   * 22 mics on a golden-angle log spiral in the x/z plane, r 8 mm -> 0.60 m.
#       - With max_freq_hz capped at 8 kHz, lambda/2 = 21 mm, so an 8 mm
#         minimum baseline comfortably avoids spatial aliasing at the high
#         end. Sweep showed inner radii of 8..15 mm work; 8 mm scored best.
#       - The golden-angle azimuth ensures no two mic-pair baselines share
#         the same length AND direction (flat sidelobes).
#   * 1 center mic at the origin (reference for the BirdFinder pipeline).
#   * 2-mic y-stub at +/-40 mm to break the front/back ambiguity any planar
#     array has on its own. Sweep showed 4 y-mics is overkill at this band.
#   * Total 25 mics, max baseline ~1.02 m. Larger apertures (1.27 m, 1.54 m)
#     hurt close-pair resolution from sidelobe stacking.
#
# See experiments/array_sweep.py for the full A/B comparison and
# experiments/source_spectra.py for the FFTs that motivated the 8 kHz
# analysis cap (the highest significant peak across all sources is ~6.4 kHz,
# from the chickadee; everything else peaks below 2 kHz).

GOLDEN_ANGLE_RAD = math.radians(137.50776405)


def golden_log_spiral_xz(
    n: int,
    r_min: float = 0.005,
    r_max: float = 0.45,
) -> list[tuple[float, float]]:
    """N points in the x/z plane on a golden-angle log spiral, r_min..r_max."""
    if n <= 0:
        return []
    out: list[tuple[float, float]] = []
    for i in range(n):
        r = r_min * (r_max / r_min) ** (i / max(n - 1, 1))
        theta = i * GOLDEN_ANGLE_RAD
        out.append((r * math.cos(theta), r * math.sin(theta)))
    return out


# ---------------------------------------------------------------------------
# Build simulation
# ---------------------------------------------------------------------------

sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)

# --- Sound sources ---
for wav_file, x, y, z, scale in bird_sources:
    sim.add_sound_source(SoundSource(x, y, z, wav_file, scale=scale, sample_rate=SAMPLE_RATE))

# --- Microphone array ---
# Mic 0 (center) at origin; reference mic for the BirdFinder pipeline.
sim.add_microphone(Microphone("center", 0.0, 0.0, 0.0, sample_rate=SAMPLE_RATE))

# 22 mics on the golden-angle log spiral.
for i, (x, z) in enumerate(golden_log_spiral_xz(n=22, r_min=0.008, r_max=0.60)):
    sim.add_microphone(Microphone(f"sp_{i:02d}", float(x), 0.0, float(z),
                                  sample_rate=SAMPLE_RATE))

# Y-stub: 2 mics at +/-40 mm to disambiguate +y vs -y.
for i, y in enumerate([-0.04, 0.04]):
    sim.add_microphone(Microphone(f"y_stub_{i:02d}", 0.0, float(y), 0.0,
                                  sample_rate=SAMPLE_RATE))

# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

# 1) Ambient mix to ears so you can hear the unprocessed scene.
sim.apply_ambient_audio_to_ears()
sim.export_ears_stereo("bird_raw_ambient.wav")
sim.reset_recordings(target="ears")

# 2) Record at every microphone.
sim.run_recording()

# ---------------------------------------------------------------------------
# Bird-finding pipeline
# ---------------------------------------------------------------------------

# max_freq_hz tightened to 8 kHz: the highest significant spectral peak across
# all bird sources is the Mountain Chickadee at ~6.4 kHz (see
# experiments/source_spectra.py). Capping above that band suppresses noise-
# driven detections without losing any real signal.
config = BirdFinderConfig(
    min_freq_hz=700.0,
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
    reference_mic_index=0,   # center mic is index 0 in sim._microphones
)

pipeline = BirdFinderPipeline(sim, config=config)
pipeline.run()
pipeline.plot_sources(animate=True, audio_filename="bird_raw_ambient.wav")
# pipeline.export_target_tracks()  # disabled while we iterate on localization

print("Bird-finder simulation complete.")
print(f"  Microphones: {len(sim._microphones)}")
print(f"  Sources    : {len(sim._sources)}")
print(f"  Sample rate: {sim.sample_rate} Hz")

# Visualize setup
sim.show_scene_3d()