"""Diagnostic: localization sanity check.

Builds a small simulation, picks a frame where one source is dominant,
and runs Bartlett + Capon DOA. Compares the result to the ground-truth
direction of that source.
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

# Use just ONE source to keep diagnosis clean.
src_path = birds_dir / "XC1029980 - Mountain Chickadee - Poecile gambeli.wav"
src_xyz = (10.0, 5.0, -3.0)

sim = MicrophoneArraySimulation(sample_rate=SAMPLE_RATE)
sim.add_sound_source(SoundSource(*src_xyz, src_path, scale=1.0, sample_rate=SAMPLE_RATE))

# Build the same staged-arm array as run_simulation_bird_finder.py
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

sim.add_microphone(Microphone("center", 0, 0, 0, sample_rate=SAMPLE_RATE))
arm_d = staged_arm_distances()
for i, x in enumerate(arm_d):
    sim.add_microphone(Microphone(f"x_{i}", float(x), 0, 0, sample_rate=SAMPLE_RATE))
for i, z in enumerate(arm_d):
    sim.add_microphone(Microphone(f"z_{i}", 0, 0, float(z), sample_rate=SAMPLE_RATE))
for i, y in enumerate([-0.015, 0.0, 0.015]):
    sim.add_microphone(Microphone(f"y_{i}", 0.10, float(y), 0.10, sample_rate=SAMPLE_RATE))

sim.run_recording()

# Ground-truth direction of source.
src = sim._sources[0]
az_rad, el_rad = src.azel(degrees=False)
gt_az, gt_el = float(np.degrees(az_rad)), float(np.degrees(el_rad))
print(f"Ground truth: az={gt_az:.2f} deg, el={gt_el:.2f} deg")

# Compute short STFT.
X_short, times_short, freqs_short = sim.compute_short_stft_all_mics(window_s=0.1, hop_ms=25.0)
print(f"Short STFT: {X_short.shape} (T_frames, M, F),  T_frames={X_short.shape[0]}")

# Pick the loudest moment in the reference mic's recording (use mic 0).
ref_power = np.sum(np.abs(X_short[:, 0, :]) ** 2, axis=1)
ti = int(np.argmax(ref_power))
t_centre = float(times_short[ti])
print(f"Loudest short-STFT frame: i={ti}, t={t_centre:.3f}s")

# Find the loudest frequency at that frame.
mag_ref = np.abs(X_short[ti, 0, :])
# Restrict to bird band.
band = (freqs_short >= 1000.0) & (freqs_short <= 20000.0)
mag_ref_band = np.where(band, mag_ref, 0.0)
fk = int(np.argmax(mag_ref_band))
freq_hz = float(freqs_short[fk])
print(f"Strongest in-band bin at that frame: k={fk}, f={freq_hz:.1f} Hz, mag={mag_ref[fk]:.4g}")

# ---- Bartlett DOA scan over a single bin at one time ----
mic_pos = np.stack([m.as_array() for m in sim._microphones], axis=0)
M = mic_pos.shape[0]
print(f"Mics: M={M}")

# Snapshot vector at this time, this freq:
x = X_short[ti, :, fk].astype(np.complex128)         # (M,)
print(f"snapshot |x|: {np.abs(x)}")

az_grid = np.arange(-180.0, 180.0, 2.0)
el_grid = np.arange(-90.0, 90.001, 5.0)
AZ, EL = np.meshgrid(az_grid, el_grid, indexing="ij")
A = steering_matrix(mic_pos, np.array([freq_hz]), AZ, EL, c=SPEED_OF_SOUND)[..., 0, :]   # (A, E, M)

# Bartlett: P_B = |a^H x|^2
P_bartlett = np.abs(np.einsum("aem,m->ae", np.conj(A), x)) ** 2
ai_b, ei_b = np.unravel_index(int(np.argmax(P_bartlett)), P_bartlett.shape)
print(f"Bartlett peak: az={az_grid[ai_b]:.1f}, el={el_grid[ei_b]:.1f}  (gt az={gt_az:.1f}, el={gt_el:.1f})")

# ---- Capon (with a small time/freq covariance) ----
# build covariance over [t_centre-0.5, t_centre], freq bins fk-1..fk+1
mask = (times_short >= t_centre - 0.5) & (times_short <= t_centre + 1e-9)
snaps = X_short[mask][:, :, max(0, fk-1):fk+2].astype(np.complex128)        # (T, M, K)
snaps = snaps.reshape(-1, M)
N = max(snaps.shape[0], 1)
R = (snaps.conj().T @ snaps) / N
R = 0.5 * (R + R.conj().T)
lam = 1e-2 * (np.real(np.trace(R)) / M + 1e-12)
R += lam * np.eye(M)
R_inv = np.linalg.pinv(R)
print(f"Capon: N_snapshots={N}, M={M}")

Rinv_a = np.einsum("mn,aen->aem", R_inv, A)
denom = np.einsum("aem,aem->ae", np.conj(A), Rinv_a).real
denom = np.maximum(denom, 1e-18)
P_capon = 1.0 / denom
ai_c, ei_c = np.unravel_index(int(np.argmax(P_capon)), P_capon.shape)
print(f"Capon peak:    az={az_grid[ai_c]:.1f}, el={el_grid[ei_c]:.1f}")

# ---- Sanity check: synthesize matched steering and verify peak appears at gt ----
print("\nSanity test: simulate ideal a(gt) plus noise, then run Bartlett.")
a_gt = steering_matrix(mic_pos, np.array([freq_hz]), gt_az, gt_el, c=SPEED_OF_SOUND)[0, :]
rng = np.random.default_rng(0)
x_synth = a_gt + 0.01 * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
P_synth = np.abs(np.einsum("aem,m->ae", np.conj(A), x_synth)) ** 2
ai_s, ei_s = np.unravel_index(int(np.argmax(P_synth)), P_synth.shape)
print(f"  synthetic  Bartlett peak: az={az_grid[ai_s]:.1f}, el={el_grid[ei_s]:.1f}")

# ---- Try same but using x = exp(j*2pi*f*tau) where tau = +(r.u)/c (no minus) ----
print("\nProbe: alternative phase convention (try both signs).")
def steer_alt(sign):
    u = np.array([np.cos(np.deg2rad(gt_el))*np.cos(np.deg2rad(gt_az)),
                  np.cos(np.deg2rad(gt_el))*np.sin(np.deg2rad(gt_az)),
                  np.sin(np.deg2rad(gt_el))])
    tau = sign * (mic_pos @ u) / SPEED_OF_SOUND
    return np.exp(1j * 2 * np.pi * freq_hz * tau)

for sign in (+1, -1):
    a = steer_alt(sign)
    p = np.abs(np.einsum("aem,m->ae", np.conj(A), a)) ** 2
    ai, ei = np.unravel_index(int(np.argmax(p)), p.shape)
    print(f"  sign={sign:+d}: peak az={az_grid[ai]:.1f}, el={el_grid[ei]:.1f}")
