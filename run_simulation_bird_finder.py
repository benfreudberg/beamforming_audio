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

from headphone_mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
)
from headphone_mic_array.bird_finder import BirdFinderConfig, BirdFinderPipeline

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

# Place each bird at a different position in the horizontal plane (x, y, z).
# Distances are in metres; all sources are 10 m away at varying azimuths.
bird_sources = [
    # (wav_file,                                           x,     y,    z, scale)
    (birds_dir / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav",     10.0,   5.0,  -3.0, 0.65),
    (birds_dir / "XC1073500 - Crested Quetzal - Pharomachrus antisianus.wav", -7.0,  10.0, 0.0, 1.0),
    (birds_dir / "XC632143 - White-necked Raven - Corvus albicollis.wav",     4.0,  20.0,  4.0, 0.75),
    (birds_dir / "XC767753 - Robin Accentor - Prunella rubeculoides.wav",    5.0,  -17.0, 4.0, 1.0),
    (birds_dir / "XC821441 - Great Horned Owl - Bubo virginianus.wav",    -5.0,  17.0, 8.0, 1.0),
]

# ---------------------------------------------------------------------------
# Array geometry helpers
# ---------------------------------------------------------------------------

def staged_arm_distances() -> list[float]:
    """
    Distances from origin for one arm with three spacing regimes:
      - 4 tightly spaced mics
      - gap
      - 3 medium spaced mics
      - larger gap
      - 1 wider spaced mic
    """
    tight_spacing = 0.0075   # 7.5 mm, keeps spatial aliasing controlled near 22 kHz
    medium_spacing = 0.0150  # 15 mm
    wide_spacing = 0.0300    # 30 mm
    gap_small = 0.0500       # 5 cm
    gap_large = 0.1200       # 12 cm

    distances: list[float] = []

    # 3 tight mics
    pos = 0.0
    for i in range(3):
        pos += tight_spacing * 1.2 ** i
        distances.append(pos)

    # gap, then 3 medium mics
    pos += gap_small
    for i in range(3):
        if i > 0:
            pos += medium_spacing * 1.2 ** i
        distances.append(pos)

    # larger gap, then 1 wide mic
    pos += gap_large
    distances.append(pos)

    return distances

# ---------------------------------------------------------------------------
# Build simulation
# ---------------------------------------------------------------------------

sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)

# --- Sound sources ---
for wav_file, x, y, z, scale in bird_sources:
    sim.add_sound_source(SoundSource(x, y, z, wav_file, scale=scale, sample_rate=SAMPLE_RATE))

# --- Microphone array ---
# Buildable staged-arm design:
# 1) +x arm from origin: 4 tight, gap, 4 medium, larger gap, 1 wide.
# 2) +z arm from origin with the same pattern (origin mic not duplicated).
# 3) +y arm from (x=10 cm, z=10 cm): 2 tight spacings.
# 4) Short -y mirror branch from (x=10 cm, z=10 cm) (2 mics) to improve +y/-y discrimination.

center_mic = Microphone("center", 0.0, 0.0, 0.0, sample_rate=SAMPLE_RATE)
sim.add_microphone(center_mic)

arm_d = staged_arm_distances()

# +x arm
for i, x in enumerate(arm_d):
    sim.add_microphone(Microphone(f"x_arm_{i:02d}", float(x), 0.0, 0.0, sample_rate=SAMPLE_RATE))

# +z arm
for i, z in enumerate(arm_d):
    sim.add_microphone(Microphone(f"z_arm_{i:02d}", 0.0, 0.0, float(z), sample_rate=SAMPLE_RATE))

# +y arm at (x=10 cm, z=10 cm)
y_anchor_x = 0.10
y_anchor_z = 0.10
y_medium_spacing = 0.0150
y_positions = [-y_medium_spacing, 0, y_medium_spacing]

for i, y in enumerate(y_positions):
    sim.add_microphone(
        Microphone(f"y_offset_arm_{i:02d}", y_anchor_x, float(y), y_anchor_z, sample_rate=SAMPLE_RATE)
    )

# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

# sim.run_recording()

# Optional: raw ambient mix to ears so you can hear the unprocessed scene
sim.apply_ambient_audio_to_ears()
sim.export_ears_stereo("bird_raw_ambient.wav")
sim.reset_recordings(target="ears")

# ---------------------------------------------------------------------------
# Bird-finding pipeline
# ---------------------------------------------------------------------------

config = BirdFinderConfig(
    window_s=4.0,
    hop_ms=100.0,
    n_peaks=20,
    min_freq_hz=1_000.0,
    max_freq_hz=22_000.0,
    freq_tolerance_hz=50.0,
    max_miss_frames=5,
    high_freq_bias=0.01,
    angle_threshold_deg=3.0,
    reference_mic_index=0,   # center mic is index 0 in sim._microphones
)

pipeline = BirdFinderPipeline(sim, config=config)

# TODO: un-comment once pipeline steps are implemented:
# pipeline.run()
# pipeline.plot_sources(animate=True)
# pipeline.export_ear_tracks()

print("Bird-finder simulation complete.")
print(f"  Microphones: {len(sim._microphones)}")
print(f"  Sources    : {len(sim._sources)}")
print(f"  Sample rate: {sim.sample_rate} Hz")
print("Run pipeline.run() once BirdFinderPipeline is implemented.")

# Visualize setup
sim.show_scene_3d()