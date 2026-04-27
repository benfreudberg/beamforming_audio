"""Diagnostic: localize multiple peaks from real multi-source recording.

For each strong peak in a chosen short-STFT frame, run Bartlett DOA and
compare to the closest ground-truth source direction.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mic_array.simulator import MicrophoneArraySimulation, SoundSource, Microphone, SPEED_OF_SOUND
from mic_array.steering import steering_matrix

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

gt = []
for s in sim._sources:
    az, el = s.azel(degrees=True)
    gt.append((float(az), float(el)))
print("Ground-truth source directions (az, el):")
for i, g in enumerate(gt):
    print(f"  src{i}: az={g[0]:6.1f}  el={g[1]:6.1f}")

X_short, times_short, freqs_short = sim.compute_short_stft_all_mics(window_s=0.1, hop_ms=25.0)
mic_pos = np.stack([m.as_array() for m in sim._microphones], axis=0)
M = mic_pos.shape[0]

az_grid = np.arange(-180.0, 180.0, 2.0)
el_grid = np.arange(-90.0, 90.001, 5.0)
AZ, EL = np.meshgrid(az_grid, el_grid, indexing="ij")

def closest_gt(az, el):
    best = None; best_d = 1e9
    for i, (gaz, gel) in enumerate(gt):
        # great-circle angle in degrees
        u1 = np.array([np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az)),
                       np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az)),
                       np.sin(np.deg2rad(el))])
        u2 = np.array([np.cos(np.deg2rad(gel))*np.cos(np.deg2rad(gaz)),
                       np.cos(np.deg2rad(gel))*np.sin(np.deg2rad(gaz)),
                       np.sin(np.deg2rad(gel))])
        d = float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))))
        if d < best_d:
            best_d = d; best = i
    return best, best_d

# Sample several time frames spread across the recording.
test_frames = np.linspace(20, X_short.shape[0]-20, 6).astype(int)
print(f"\nFor each test frame: top-5 in-band peaks (by ref-mic mag) -> Bartlett DOA -> closest GT.")

for ti in test_frames:
    t = times_short[ti]
    mag_ref = np.abs(X_short[ti, 0, :])
    band = (freqs_short >= 1000.0) & (freqs_short <= 20000.0)
    idx_band = np.where(band)[0]
    top = idx_band[np.argsort(mag_ref[idx_band])[::-1][:5]]
    print(f"\n  t={t:.2f}s  ref_total_power={float(np.sum(mag_ref**2)):.3g}")
    for fk in top:
        f = float(freqs_short[fk])
        x = X_short[ti, :, fk].astype(np.complex128)
        A = steering_matrix(mic_pos, np.array([f]), AZ, EL, c=SPEED_OF_SOUND)[..., 0, :]
        P = np.abs(np.einsum("aem,m->ae", np.conj(A), x)) ** 2
        ai, ei = np.unravel_index(int(np.argmax(P)), P.shape)
        az_p, el_p = float(az_grid[ai]), float(el_grid[ei])
        # Spectral peak/median ratio (sharpness).
        snr = float(P[ai, ei] / (np.median(P) + 1e-12))
        gtidx, ang = closest_gt(az_p, el_p)
        print(f"    f={f:6.0f}Hz  mag={mag_ref[fk]:7.3g}  "
              f"-> az={az_p:6.1f} el={el_p:6.1f}  "
              f"snr={snr:6.1f}  closest gt #{gtidx} ({gt[gtidx][0]:6.1f},{gt[gtidx][1]:6.1f}) "
              f"err={ang:5.1f}deg")
