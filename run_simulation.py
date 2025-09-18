from pathlib import Path
from headphone_mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
    Target,
)

# project_root is where this script lives
base_dir = Path(__file__).resolve().parent
audio_dir = base_dir / "input_audio_files"

descriptive_voice_file = audio_dir / "descriptive.wav"
informative_voice_file = audio_dir / "informative.wav"

sim = MicrophoneArraySimulation()
sim.add_sound_source(SoundSource(1, 2, 0, descriptive_voice_file))
sim.add_sound_source(SoundSource(-2, 2, 0, informative_voice_file))

# --- Build a 5x5 grid of microphones (x–z plane), 2 cm spacing, centered at (0,0,0) ---
spacing = 0.02  # meters
indices = [-2, -1, 0, 1, 2]
center_mic = None

for ix in indices:
    for iz in indices:
        x = ix * spacing
        z = iz * spacing
        name = f"mic_x{int(round(x*100)):02d}cm_z{int(round(z*100)):02d}cm"
        mic = Microphone(name, x, 0.0, z)
        sim.add_microphone(mic)
        if ix == 0 and iz == 0:
            center_mic = mic  # keep a handle to the center mic

# Ambient (all sources → ears)
sim.apply_ambient_audio_to_ears()
sim.export_ears_stereo("raw_ambient.wav")
sim.reset_recordings()

# Record with the mic grid → build target → apply to ears
sim.run_recording()
target = sim.add_target(1, 2, 0)

sim.apply_targets_to_ears()
sim.export_ears_stereo("target_only.wav")

# Export center mic recording (if found)
if center_mic is not None:
    center_mic.save_recording()

# Export target track
target.export()

# Add muffled ambient background
sim.apply_ambient_audio_to_ears(scaling=0.2)
sim.export_ears_stereo("target_and_muffled_ambient.wav")

# Visualize setup
sim.show_scene_3d()
