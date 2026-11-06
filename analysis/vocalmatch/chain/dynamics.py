"""Compressor and de-esser.

Both use the same building blocks, defined exactly in shared/CHAIN.md:

* detector  = max(|L|, |R|) per sample, converted to dB (floored at -120 dB)
* a static soft-knee gain computer (Giannoulis, Massberg & Reiss, JAES 2012)
* log-domain "smooth branching" ballistics: a one-pole on the gain *reduction*
  that uses the attack coefficient when reduction is rising, release otherwise.
"""

from __future__ import annotations

import math

import numpy as np

from ..preset import Compressor, DeEsser
from .biquad import rbj_sos

LEVEL_FLOOR_DB = -120.0
_FLOOR_LIN = 10.0 ** (LEVEL_FLOOR_DB / 20.0)

DEESSER_ATTACK_MS = 0.5
DEESSER_RELEASE_MS = 40.0
DEESSER_Q = 0.7071


def to_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), _FLOOR_LIN))


def time_constant_coeff(time_ms: float, fs: float) -> float:
    """One-pole coefficient for time constant *time_ms* (63% step response)."""
    t = max(float(time_ms), 1e-3) * 1e-3
    return math.exp(-1.0 / (t * fs))


def gain_computer_db(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Static gain *reduction* in dB (>= 0) for a downward compressor."""
    x = np.asarray(level_db, dtype=np.float64)
    ratio = max(float(ratio), 1.0)
    slope = 1.0 / ratio - 1.0  # <= 0
    over = x - threshold_db
    y = x.copy()
    if knee_db > 0.0:
        half = knee_db / 2.0
        in_knee = np.abs(over) <= half
        y[in_knee] = x[in_knee] + slope * (over[in_knee] + half) ** 2 / (2.0 * knee_db)
        above = over > half
    else:
        above = over > 0.0
    y[above] = threshold_db + over[above] / ratio
    return x - y


def smooth_branching(gr_db: np.ndarray, attack_coeff: float, release_coeff: float, state: float = 0.0) -> np.ndarray:
    """Log-domain branching one-pole smoother of gain reduction (dB).

    s[n] = a_A s[n-1] + (1-a_A) gr[n]   if gr[n] > s[n-1]
           a_R s[n-1] + (1-a_R) gr[n]   otherwise
    """
    aa = float(attack_coeff)
    ar = float(release_coeff)
    ca = 1.0 - aa
    cr = 1.0 - ar
    s = float(state)
    out = [0.0] * len(gr_db)
    i = 0
    for g in np.asarray(gr_db, dtype=np.float64).tolist():
        if g > s:
            s = aa * s + ca * g
        else:
            s = ar * s + cr * g
        out[i] = s
        i += 1
    return np.asarray(out, dtype=np.float64)


def compressor_gain_db(detector: np.ndarray, fs: float, p: Compressor) -> np.ndarray:
    """Total gain (dB, including makeup) for a detector signal sampled at *fs*.

    *detector* is the linear peak detector (max |x| over channels). The
    estimator calls this with a decimated detector to make parameter searches
    cheap; the renderer calls it at full rate.
    """
    level = to_db(detector)
    gr = gain_computer_db(level, p.threshold_db, p.ratio, p.knee_db)
    smoothed = smooth_branching(
        gr, time_constant_coeff(p.attack_ms, fs), time_constant_coeff(p.release_ms, fs)
    )
    return p.makeup_db - smoothed


def process_compressor(x: np.ndarray, fs: float, p: Compressor) -> np.ndarray:
    detector = np.max(np.abs(x), axis=0)
    gain = 10.0 ** (compressor_gain_db(detector, fs, p) / 20.0)
    return x * gain[None, :]


def deesser_crossover_sos(freq_hz: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linkwitz-Riley 4th-order crossover: each band is two identical RBJ
    Butterworth (Q = 1/sqrt 2) sections. LR4 low and high bands are in phase,
    so low + g*high is a clean dynamic split (g = 1 gives an all-pass)."""
    lp = rbj_sos("lowpass", freq_hz, 0.0, DEESSER_Q, fs)
    hp = rbj_sos("highpass", freq_hz, 0.0, DEESSER_Q, fs)
    return np.vstack([lp, lp]), np.vstack([hp, hp])


def process_deesser(x: np.ndarray, fs: float, p: DeEsser) -> np.ndarray:
    """Split-band de-esser.

    low, high = LR4 crossover of x at frequency_hz
    gr        = smooth(min(max(level_db(max|high|) - threshold, 0), reduction_db))
    y         = low + 10^(-gr/20) * high
    """
    from scipy.signal import sosfilt

    if p.reduction_db <= 0.0:
        return x
    sos_lp, sos_hp = deesser_crossover_sos(p.frequency_hz, fs)
    low = sosfilt(sos_lp, x, axis=-1)
    high = sosfilt(sos_hp, x, axis=-1)
    level = to_db(np.max(np.abs(high), axis=0))
    gr = np.clip(level - p.threshold_db, 0.0, p.reduction_db)
    gr = smooth_branching(
        gr,
        time_constant_coeff(DEESSER_ATTACK_MS, fs),
        time_constant_coeff(DEESSER_RELEASE_MS, fs),
    )
    g = 10.0 ** (-gr / 20.0)
    return low + g[None, :] * high
