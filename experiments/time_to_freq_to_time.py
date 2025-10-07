import sys, os
from pathlib import Path
import numpy as np

# If you need this path tweak, keep it. Otherwise prefer running with: python -m some_pkg.my_script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from headphone_mic_array.audio_utils import (
    read_wav_float_mono, resample_to, apply_fractional_delay,
    get_output_dir, save_wav_float32, project_root
)

INTERNAL_SR = 48000

base_dir = Path(__file__).resolve().parent.parent
audio_dir = base_dir / "input_audio_files"
descriptive_voice_file = audio_dir / "descriptive.wav"

sr_in, audio_raw = read_wav_float_mono(descriptive_voice_file)
audio_resampled = resample_to(audio_raw, sr_in, INTERNAL_SR)

hop_ms = 10
window_ms = hop_ms * 2
Nw = int(round(INTERNAL_SR * window_ms / 1000.0))
Nh = int(round(INTERNAL_SR * hop_ms / 1000.0))

# next power of two for FFT size (you compute it, but not used further; kept in case you expand)
n = 1
while n < Nw:
    n <<= 1
nfft = n

win = np.sqrt(np.hanning(Nw).astype(float))     # 1-D window
win_pad = np.zeros(nfft, dtype=float)
win_pad[:Nw] = win

freqs = np.fft.rfftfreq(nfft, d=1.0 / INTERNAL_SR)
F = freqs.shape[0]

inbuf = np.zeros(Nw, dtype=float)
olabuf = np.zeros(Nw, dtype=float)

pos = 0
n_samples = audio_resampled.shape[0]
inbuf_fill = 0
emitted = []

def process_one_frame() -> np.ndarray:
    global olabuf

    x_win = (inbuf * win).astype(float)
    x_win_pad = np.zeros(nfft, dtype=float)
    x_win_pad[:Nw] = x_win

    x = np.fft.rfft(x_win, n=nfft)
    y = np.fft.irfft(x, n=nfft).real

    olabuf += (y[: Nw] * win).astype(float)
    y_out = olabuf[:Nh].copy()

    olabuf = np.roll(olabuf, -Nh)
    olabuf[-Nh:] = 0.0

    return y_out

while pos < n_samples:
    need = Nh - inbuf_fill
    take = min(need, n_samples - pos)
    if take > 0:
        inbuf = np.roll(inbuf, -take)
        inbuf[-take:] = audio_resampled[pos: pos + take]
        inbuf_fill += take
        pos += take

    if inbuf_fill >= Nh:
        y_hop = process_one_frame()
        emitted.append(y_hop)
        inbuf_fill -= Nh

track = np.concatenate(emitted).astype(np.float32)

save_wav_float32("test_track.wav", INTERNAL_SR, track)
