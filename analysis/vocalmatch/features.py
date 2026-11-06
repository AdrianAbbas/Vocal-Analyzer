"""Shared analysis features: framing, activity gating, voicing, LTAS, smoothing."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(n,), (n, ch) or (ch, n) -> (n,) mean of channels."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.shape[0] <= 2 and a.shape[1] > 2:
        return a.mean(axis=0)
    return a.mean(axis=1)


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(n_frames, win) strided view; trailing partial frame dropped."""
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    return sliding_window_view(x, win)[::hop]


def power_db(p: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, EPS))


def frame_rms_db(x: np.ndarray, fs: float, win_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    win = max(1, int(round(win_ms * fs / 1000)))
    hop = max(1, int(round(hop_ms * fs / 1000)))
    f = frames(x, win, hop)
    return power_db(np.mean(f * f, axis=1))


def active_mask(level_db: np.ndarray, rel_db: float = 40.0, floor_db: float = -75.0, ref_pct: float = 95.0) -> np.ndarray:
    """RMS gate: frames within *rel_db* of the signal's loud level (its
    *ref_pct* percentile) and above an absolute floor."""
    ref = np.percentile(level_db, ref_pct)
    return level_db > max(ref - rel_db, floor_db)


def voicing(x: np.ndarray, fs: float, win: int, hop: int, f0_min: float = 70.0, f0_max: float = 600.0,
            threshold: float = 0.45) -> np.ndarray:
    """Per-frame voiced flag from the normalised autocorrelation peak in the
    f0 lag range."""
    f = frames(x, win, hop)
    f = f - f.mean(axis=1, keepdims=True)
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(f * np.hanning(win), n_fft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n_fft, axis=1)[:, :win]
    lo = max(1, int(fs / f0_max))
    hi = min(win - 1, int(fs / f0_min))
    peak = ac[:, lo:hi].max(axis=1)
    return peak / (ac[:, 0] + EPS) > threshold


def ltas(
    x: np.ndarray,
    fs: float,
    n_fft: int = 4096,
    hop: int = 1024,
    rel_db: float = 40.0,
    normalize_frames: bool = True,
    norm_band_hz: tuple[float, float] = (100.0, 8000.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average power spectrum over *active* frames.

    With *normalize_frames* each frame is divided by the geometric mean of its
    1/3-octave band powers in *norm_band_hz*. Filtering a frame by H or
    scaling it by a gain c multiplies that statistic by (approximately)
    the same constant exp(mean log|H|^2) * c^2 for every frame, so the
    normalised LTAS ratio between a processed and an unprocessed signal is
    unbiased by EQ and by compression (unlike unit-power normalisation,
    whose per-frame scale depends on how each frame's spectrum meets H).
    Returns (freqs_hz, psd).
    """
    x = to_mono(x)
    f = frames(x, n_fft, hop)
    window = np.hanning(n_fft)
    lev = power_db(np.mean(f * f, axis=1))
    mask = active_mask(lev, rel_db=rel_db)
    if not np.any(mask):
        mask = np.ones(len(f), bool)
    spec = np.abs(np.fft.rfft(f[mask] * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    if normalize_frames:
        bands = fractional_octave_matrix(freqs, third_octave_centres(*norm_band_hz), 3.0)
        log_norm = np.mean(np.log(spec @ bands.T + EPS), axis=1)
        spec /= np.exp(log_norm)[:, None]
    return freqs, spec.mean(axis=0)


def log_grid(f_lo: float, f_hi: float, points_per_octave: int = 12) -> np.ndarray:
    n = int(np.floor(np.log2(f_hi / f_lo) * points_per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / points_per_octave)


def third_octave_centres(f_lo: float, f_hi: float) -> np.ndarray:
    """Base-2 1/3-octave centres (referenced to 1 kHz) within [f_lo, f_hi]."""
    k = np.arange(-30, 16)
    c = 1000.0 * 2.0 ** (k / 3.0)
    # 1% slack so nominal band names (100 Hz = 99.2 Hz exact) are included
    return c[(c >= f_lo / 1.01) & (c <= f_hi * 1.01)]


def fractional_octave_matrix(bin_freqs: np.ndarray, centres: np.ndarray, fraction: float = 3.0) -> np.ndarray:
    """Row-normalised averaging matrix: row i averages bins inside
    [c_i 2^(-1/(2 fraction)), c_i 2^(1/(2 fraction))] (at least the nearest bin)."""
    half = 2.0 ** (1.0 / (2.0 * fraction))
    m = np.zeros((len(centres), len(bin_freqs)))
    for i, c in enumerate(centres):
        sel = (bin_freqs >= c / half) & (bin_freqs <= c * half)
        if not np.any(sel):
            sel = np.zeros(len(bin_freqs), bool)
            sel[np.argmin(np.abs(bin_freqs - c))] = True
        m[i, sel] = 1.0 / sel.sum()
    return m


def band_average_db(response_db_fn, centres: np.ndarray, fraction: float = 3.0, points: int = 16) -> np.ndarray:
    """Power-average a response (callable freqs -> dB) over each fractional-octave band."""
    half = 2.0 ** (1.0 / (2.0 * fraction))
    out = np.empty(len(centres))
    for i, c in enumerate(centres):
        f = np.geomspace(c / half, c * half, points)
        out[i] = power_db(np.mean(10 ** (response_db_fn(f) / 10)))
    return out
