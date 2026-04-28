"""Quick FFT analysis of the bird audio files.

For each file, compute a long-window magnitude spectrum and report the
top spectral peaks above a relative threshold so we know what the array
needs to cover at the high end.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from mic_array.audio_utils import read_wav_float_mono  # noqa: E402

BIRDS = BASE / "input_audio_files" / "birds"


def top_peaks(path: Path, n_peaks: int = 8, rel_threshold_db: float = -25.0):
    sr, x = read_wav_float_mono(path)
    if x.size == 0:
        return sr, []
    # Average magnitude spectrum (Welch-style, no overlap).
    win = 1 << 14  # 16384
    if x.size < win:
        win = 1 << int(np.floor(np.log2(max(x.size, 2))))
    n_seg = max(1, x.size // win)
    mag_acc = np.zeros(win // 2 + 1, dtype=float)
    w = np.hanning(win)
    for i in range(n_seg):
        seg = x[i * win:(i + 1) * win] * w
        mag_acc += np.abs(np.fft.rfft(seg))
    mag = mag_acc / n_seg
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    # dB normalised to peak.
    mag_db = 20.0 * np.log10(mag / (mag.max() + 1e-30) + 1e-12)
    # Find peaks above threshold.
    idx, _ = find_peaks(mag_db, height=rel_threshold_db, distance=8)
    if idx.size == 0:
        return sr, []
    # Sort by magnitude descending.
    order = np.argsort(mag_db[idx])[::-1][:n_peaks]
    chosen = idx[order]
    return sr, [(float(freqs[k]), float(mag_db[k])) for k in chosen]


def main():
    files = sorted(BIRDS.glob("*.wav"))
    print(f"{'file':<60} {'sr':>6}  highest peaks (Hz, dB rel)")
    for f in files:
        sr, peaks = top_peaks(f)
        # Sort by frequency ascending for printing.
        peaks_by_f = sorted(peaks, key=lambda p: p[0])
        # Highlight highest-frequency peak above -20 dB.
        strong_high = [p for p in peaks if p[1] >= -20.0]
        hi_f = max((p[0] for p in strong_high), default=0.0)
        peak_str = ", ".join(f"{p[0]:6.0f}({p[1]:+.0f})" for p in peaks_by_f)
        print(f"{f.name[:60]:<60} {sr:>6}  hi>=-20dB: {hi_f:6.0f} Hz | {peak_str}")


if __name__ == "__main__":
    main()
