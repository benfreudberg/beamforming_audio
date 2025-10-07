from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


# --- Paths ---
def package_dir() -> Path:
    """Return the directory where this file (audio_utils.py) lives."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the project root (one level above the package folder)."""
    return package_dir().parent


def get_output_dir() -> Path:
    """
    Ensure and return the output/ folder in the project root.
    If OUTPUT_DIR environment variable is set, use that instead.
    """
    override = os.environ.get("OUTPUT_DIR")
    base = Path(override) if override else project_root()
    out = base / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- Audio helpers ---
def read_wav_float_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV file and return (sample_rate, mono float32 data in [-1,1])."""
    path = Path(path)
    sr, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / np.iinfo(np.int32).max
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:  # already float
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data


def resample_to(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio to a new sample rate using polyphase filtering."""
    if sr_in == sr_out:
        return data.astype(np.float32)
    gcd = np.gcd(sr_in, sr_out)
    up, down = sr_out // gcd, sr_in // gcd
    return resample_poly(data, up, down).astype(np.float32)


def apply_fractional_delay(data: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Apply a fractional delay using simple linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        Input 1D signal.
    delay_samples : float
        Desired delay in samples (can be fractional, >= 0).

    Returns
    -------
    np.ndarray
        Delayed signal.
    """
    if delay_samples < 1e-9:
        return data.copy()

    integer_delay = int(np.floor(delay_samples))
    fractional_delay = delay_samples - integer_delay
    len_data = len(data)
    n = np.arange(len_data)
    y = np.zeros(len_data + integer_delay + 1)
    data = np.append(data, data[-1])
    y[integer_delay:len_data + integer_delay] = (1 - fractional_delay) * data[n] + fractional_delay * data[n + 1]

    return y.astype(np.float32)


def save_wav_float32(path: str | Path, sr: int, data: np.ndarray) -> None:
    """Save float32 data as a WAV file."""
    path = Path(path)
    wavfile.write(path, sr, data.astype(np.float32))
