from pathlib import Path
from headphone_mic_array.simulator import (
    MicrophoneArraySimulation,
    SoundSource,
    Microphone,
)

# project_root is where this script lives
base_dir = Path(__file__).resolve().parent
audio_dir = base_dir / "input_audio_files"

descriptive_voice_file = audio_dir / "descriptive.wav"
informative_voice_file = audio_dir / "informative.wav"

sim = MicrophoneArraySimulation()
sim.add_sound_source(SoundSource(1, 2, 0, descriptive_voice_file))
sim.add_sound_source(SoundSource(-2, 2, 0, informative_voice_file))
sim.add_target("descriptive", 1, 2, 0)
sim.add_target("informative", -2, 2, 0)

# --- Build a 5x5 grid of microphones (x–z plane), 2 cm spacing, centered at (0,0,0) ---
spacing = 0.02  # meters
indices = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
center_mic = None

iz = 0
for ix in indices:
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

# Export target tracks
sim.create_target_tracks_DS()
sim.export_target_tracks("DS")
sim.create_target_tracks_MVDR()
sim.export_target_tracks("MVDR")
# sim.apply_targets_to_ears()
# sim.export_ears_stereo("targets_applied_to_ears.wav")

# Export center mic recording (if found)
if center_mic is not None:
    center_mic.export()

# Visualize setup
sim.show_scene_3d()
