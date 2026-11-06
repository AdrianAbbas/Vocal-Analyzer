"""Audio file I/O (WAV/FLAC/MP3 via libsndfile) with resampling."""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    """Return (data with shape (n, channels), sample_rate)."""
    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    if target_fs is not None and target_fs != fs:
        data = resample(data, fs, target_fs)
        fs = target_fs
    return data, int(fs)


def resample(data: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    frac = Fraction(int(fs_out), int(fs_in)).limit_denominator(1000)
    return resample_poly(data, frac.numerator, frac.denominator, axis=0)


def write_audio(path: str, data: np.ndarray, fs: int) -> None:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        warnings.warn(f"{path}: peak {20 * np.log10(peak):+.1f} dBFS; writing as float to avoid clipping")
    sf.write(str(path), data, int(fs), subtype="FLOAT" if peak > 1.0 else "PCM_24")
