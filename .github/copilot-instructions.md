## Repo context

This repository simulates microphone-array beamforming for audio sources. The primary entry point is `run_simulation.py` which composes a `MicrophoneArraySimulation` (in `headphone_mic_array/simulator.py`) using small building blocks:

- `SoundSource`, `Microphone`, `Target`, `MicrophoneArraySimulation` (headphone_mic_array/simulator.py)
- Beamforming implementation and streaming MVDR engine: `MVDRBeamformer` and `BeamformerConfig` (headphone_mic_array/beamformer.py)
- Geometry primitives: `Node` and helpers (headphone_mic_array/geometry.py)
- Audio I/O and utility functions: read/resample/delay/save (headphone_mic_array/audio_utils.py)
- Simple plotting helpers for 3D scene and beampatterns (headphone_mic_array/plotting.py)

When editing or extending the project, read those files first — they are the canonical places for audio processing, geometry, and visualization.

## Quick developer workflows

- Run the demo simulation locally: `python run_simulation.py` (requires numpy, scipy, matplotlib).
- Outputs are written to `output/` by default. You can override by setting `OUTPUT_DIR` environment variable; `audio_utils.get_output_dir()` enforces this.
- Input audio files live in `input_audio_files/`. `SoundSource` expects a path to a mono WAV; the code will resample to the internal rate.

Example: run the main script and inspect outputs

1. Edit source/array placement in `run_simulation.py` (see geometric/circular examples).
2. Run `python run_simulation.py` and check `output/` for files like `raw_ambient.wav`, `<target>_target_DS.wav`, `<target>_target_MVDR.wav`, and `targets_applied_to_ears.wav`.

## Project-specific conventions & important details

- Internal processing sample rate: INTERNAL_SR = 16000 (see `headphone_mic_array/simulator.py`). Input WAVs are resampled to this rate via `audio_utils.resample_to()`; outputs are written at INTERNAL_SR.
- Speed-of-sound constant: defined as `SPEED_OF_SOUND` in `simulator.py` and also configurable in beamformer config (BeamformerConfig.speed_of_sound).
- Beamformer operates on short STFT frames: window length Nw computed from `cfg.hop_ms`; MVDR uses per-frequency covariance matrices. Default hop/window choices are in `BeamformerConfig` (headphone_mic_array/beamformer.py).
- Low-frequency filtering: the MVDR implementation explicitly zeroes very-low-frequency bins (k < 9) in `_process_one_frame()` — be aware when debugging listening artifacts.
- The MVDR steering supports both far-field (az/el) and near-field (3D point) steering via `set_target_direction` / `set_target_point`.

## Typical change patterns and where to modify

- To change microphone geometry examples: modify `run_simulation.py` where `geometric_linear_positions_coerced_growth()` and `circular_array_xy()` are used.
- To add new beamforming algorithms: add a class in `headphone_mic_array/beamformer.py` and follow the `process_block(x)` contract used by `Target.create_track_MVDR()` (input shape (M, T) where M=mics).
- To change output naming or format: edit `audio_utils.save_wav_float32()` and the call sites in `simulator.py`.

## Integration points and dependencies

- Python packages required: numpy, scipy, matplotlib (imported across files). There is no pinned requirements file; add one if adding CI.
- No external services or network calls. All file I/O is local.

## Small examples to reference in edits

- Creating a simulation and adding sources/mics (from `run_simulation.py`):
  - sim = MicrophoneArraySimulation(); sim.add_sound_source(SoundSource(...)); sim.add_microphone(Microphone(...)); sim.run_recording()
- Producing MVDR target tracks (from `simulator.py` Target.create_track_MVDR):
  - bf = MVDRBeamformer(mic_entries, target_point=self, config=beam_former_config)
  - self._track = bf.process_block(x)

## Notes for automated agents

- Prefer reading `headphone_mic_array/simulator.py` and `headphone_mic_array/beamformer.py` to understand data flow: SoundSource → Microphone.listen_to() → Microphone.track → Target.create_track_*() → export.
- Avoid changing the internal sample rate unless updating all resampling I/O (audio_utils). Many functions assume INTERNAL_SR=16000.
- When adding tests or CI, run at least a single short run of `run_simulation.py` with minimal sources to verify end-to-end audio files are created in `output/`.

If anything here is unclear or you'd like more examples (unit-test snippets, common refactor patterns, or CI suggestions), tell me which area to expand and I will iterate.
