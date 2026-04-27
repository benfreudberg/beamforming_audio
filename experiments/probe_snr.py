"""Probe: what SNRs are real long-STFT peaks getting in the short STFT?"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mic_array.simulator import MicrophoneArraySimulation, SoundSource, Microphone

# Reuse same setup as run_simulation_bird_finder.py
SAMPLE_RATE = 48000
base_dir = Path(__file__).resolve().parent.parent
birds_dir = base_dir / "input_audio_files" / "birds"

bird_sources = [
    (birds_dir / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav",     10.0,   5.0,  -3.0, 0.65),
    (birds_dir / "XC1073500 - Crested Quetzal - Pharomachrus antisianus.wav", -7.0,  10.0, 0.0, 1.0),
    (birds_dir / "XC632143 - White-necked Raven - Corvus albicollis.wav",     4.0,  20.0,  4.0, 0.75),
    (birds_dir / "XC767753 - Robin Accentor - Prunella rubeculoides.wav",    5.0,  -17.0, 4.0, 1.0),
    (birds_dir / "XC821441 - Great Horned Owl - Bubo virginianus.wav",    -5.0,  17.0, 8.0, 1.0),
]

def staged_arm_distances():
    tight, med, gap_s, gap_l = 0.0075, 0.0150, 0.05, 0.12
    distances = []
    pos = 0.0
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

sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)
for wav, x, y, z, scale in bird_sources:
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

frames = sim.compute_rolling_fft(mic_index=0, window_s=4.0, hop_ms=100.0)
X_short, times_short, freqs_short = sim.compute_short_stft_all_mics(window_s=0.1, hop_ms=25.0)

# Pick frame near t=2.25s (we know there's a strong peak there)
target_t = 2.25
fi = int(np.argmin([abs(f.time_s - target_t) for f in frames]))
frame = frames[fi]
print(f"Frame {fi} at t={frame.time_s:.3f}s")

peaks = sim.find_peak_frequencies_in_frame(frame, n_peaks=10, min_freq_hz=1000, max_freq_hz=20000)
print(f"Long-STFT top peaks at t={frame.time_s:.2f}s:")
for (freq_hz, long_mag) in peaks:
    ti, fk, snr = sim._peak_snr_at(X_short, times_short, freqs_short,
                                   frame.time_s, freq_hz, ref_mic_index=0,
                                   min_freq_hz=1000.0, max_freq_hz=20000.0)
    ref_mag = float(np.abs(X_short[ti, 0, fk]))
    print(f"  long_mag={long_mag:9.4g}  f={freq_hz:7.1f}Hz  short_ref_mag={ref_mag:7.3g}  snr={snr:6.2f}")

# Now find a "quiet" long frame: scan all frames and pick the one with smallest total magnitude
totals = np.array([float(np.sum(f.magnitudes)) for f in frames])
qi = int(np.argmin(totals))
print(f"\nQuietest long frame: i={qi}, t={frames[qi].time_s:.2f}s, total_mag={totals[qi]:.3g}")
qpeaks = sim.find_peak_frequencies_in_frame(frames[qi], n_peaks=10, min_freq_hz=1000, max_freq_hz=20000)
print(f"  peaks found: {len(qpeaks)}")
for (freq_hz, long_mag) in qpeaks:
    ti, fk, snr = sim._peak_snr_at(X_short, times_short, freqs_short,
                                   frames[qi].time_s, freq_hz, ref_mic_index=0,
                                   min_freq_hz=1000.0, max_freq_hz=20000.0)
    ref_mag = float(np.abs(X_short[ti, 0, fk]))
    print(f"  long_mag={long_mag:9.4g}  f={freq_hz:7.1f}Hz  short_ref_mag={ref_mag:7.3g}  snr={snr:6.2f}")
