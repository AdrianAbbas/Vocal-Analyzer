"""Feature helpers shared by the milestone-3 estimators (delay, reverb, width, saturation,
de-esser).

Every estimator follows the compressor estimator's recipe: compute level-independent
features on the reference, simulate the chain's own effect on (a proxy of) the signal that
enters it, and fit the effect's settings so that the same features match. Nothing relies on
sample alignment between the reference and the dry vocal.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt

from ..chain import to_stereo
from ..features import EPS, power_db

# Analysis bands (Hz) for the time-based effects.
BANDS_HZ = ((200.0, 1000.0), (1000.0, 3000.0), (3000.0, 6000.0), (6000.0, 12000.0))


def mid_side(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = to_stereo(x)
    return 0.5 * (s[0] + s[1]), 0.5 * (s[0] - s[1])


@lru_cache(maxsize=32)
def _band_sos(lo: float, hi: float, fs: float) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    return butter(4, [lo, hi], btype="band", fs=fs, output="sos")


def band_filter(x: np.ndarray, band: tuple[float, float], fs: float) -> np.ndarray:
    return sosfilt(_band_sos(float(band[0]), float(band[1]), float(fs)), x, axis=-1)


def autocorr_at(x: np.ndarray, lags) -> np.ndarray:
    """Normalised autocorrelation sum(x[n] x[n-L]) / sum(x^2) at the given lags (samples)."""
    e = float(np.dot(x, x)) + EPS
    out = []
    for lag in lags:
        lag = int(lag)
        out.append(float(np.dot(x[lag:], x[: len(x) - lag])) / e if 0 < lag < len(x) else 0.0)
    return np.array(out)


def autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation for lags 0..max_lag (FFT)."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n + max_lag + 1)))
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft)[: max_lag + 1]
    return ac / (ac[0] + EPS)


def envelope_db(x: np.ndarray, fs: float, hop_ms: float = 10.0, win_ms: float = 20.0) -> np.ndarray:
    """RMS level (dB) in frames of *win_ms* every *hop_ms*; frame k is centred at k*hop."""
    hop = max(1, int(round(hop_ms * fs / 1000.0)))
    win = max(hop, int(round(win_ms * fs / 1000.0)))
    e = x * x
    c = np.concatenate([[0.0], np.cumsum(e)])
    n_frames = len(x) // hop
    centres = np.arange(n_frames) * hop
    lo = np.clip(centres - win // 2, 0, len(x))
    hi = np.clip(centres + win // 2, 0, len(x))
    p = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return power_db(p)


def offsets(level_db: np.ndarray, hop_ms: float = 10.0, drop_db: float = 6.0, drop_ms: float = 60.0,
            hold_ms: float = 300.0, active_rel_db: float = 30.0, min_phrase_ms: float = 200.0) -> list[int]:
    """Frames where a phrase ends: the level was active for *min_phrase_ms*, falls by at least
    *drop_db* within *drop_ms*, and does not rise again by more than 3 dB for *hold_ms*."""
    ref = np.percentile(level_db, 95)
    active = level_db > ref - active_rel_db
    k_drop = max(1, int(round(drop_ms / hop_ms)))
    k_hold = int(round(hold_ms / hop_ms))
    k_phrase = int(round(min_phrase_ms / hop_ms))
    n = len(level_db)
    out: list[int] = []
    k = k_phrase
    while k < n - k_hold:
        if active[k] and np.all(active[k - k_phrase : k]) and level_db[k] - level_db[k + k_drop] >= drop_db:
            # the phrase's last loud frame: the local maximum just before the fall
            seg = level_db[k : k + k_drop + 1]
            start = k + int(np.argmax(seg))
            after = level_db[start + 1 : start + 1 + k_hold]
            running_min = np.minimum.accumulate(after)
            if np.all(after - running_min <= 3.0):
                out.append(start)
                k = start + k_hold
                continue
        k += 1
    return out


def decay_curves(level_db: np.ndarray, starts: list[int], times_ms: np.ndarray, hop_ms: float = 10.0) -> np.ndarray:
    """Median (over phrase ends) of level(start + t) - level(start), at the given times."""
    ks = np.round(np.asarray(times_ms) / hop_ms).astype(int)
    rows = [level_db[s + ks] - level_db[s] for s in starts if s + ks[-1] < len(level_db)]
    if not rows:
        return np.full(len(ks), np.nan)
    return np.median(np.array(rows), axis=0)


def gap_mask(level_db: np.ndarray, starts: list[int], lo_ms: float, hi_ms: float, hop_ms: float = 10.0) -> np.ndarray:
    """Frames between *lo_ms* and *hi_ms* after each phrase end."""
    m = np.zeros(len(level_db), bool)
    a, b = int(round(lo_ms / hop_ms)), int(round(hi_ms / hop_ms))
    for s in starts:
        m[s + a : min(s + b, len(m))] = True
    return m


def active_frames(level_db: np.ndarray, rel_db: float = 20.0) -> np.ndarray:
    return level_db > np.percentile(level_db, 95) - rel_db


def match_level(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale *x* so that its active-frame RMS equals *target*'s (proxies for simulations)."""
    def act_rms(v):
        m = np.max(np.abs(to_stereo(v)), axis=0) if np.ndim(v) == 2 else np.abs(v)
        p = m * m
        lev = power_db(p)
        sel = lev > np.percentile(lev, 90) - 30
        return np.sqrt(np.mean(p[sel]) + EPS)

    return x * (act_rms(target) / act_rms(x))
