"""Linear / static / time-based effects of the reference chain.

Every function takes and returns a float64 array of shape (2, n) (stereo).
The exact algorithms are specified in shared/CHAIN.md so the JUCE plugin can
reproduce them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import hadamard
from scipy.signal import lfilter, sosfilt

from ..preset import EQ, Delay, Doubler, EQBand, Reverb, Saturation
from .biquad import clamp_freq, rbj_sos

# ----------------------------------------------------------------------- gain


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def process_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return x * db_to_lin(gain_db)


# ------------------------------------------------------------------------- EQ


def eq_sos(bands: list[EQBand], fs: float) -> np.ndarray:
    rows = [
        rbj_sos(b.type, b.freq_hz, b.gain_db, b.q, fs)
        for b in bands
        if b.enabled and b.gain_db != 0.0
    ]
    if not rows:
        return np.zeros((0, 6))
    return np.vstack(rows)


def process_eq(x: np.ndarray, fs: float, p: EQ) -> np.ndarray:
    sos = eq_sos(p.bands, fs)
    if len(sos) == 0:
        return x
    return sosfilt(sos, x, axis=-1)


# ----------------------------------------------------------------- saturation


def process_saturation(x: np.ndarray, p: Saturation) -> np.ndarray:
    """wet = tanh(g x) / g with g = 10^(drive/20): unity small-signal gain,
    soft clipping at +-1/g. Output = (1-m) dry + m wet."""
    m = p.mix_pct / 100.0
    if m <= 0.0:
        return x
    g = db_to_lin(p.drive_db)
    wet = np.tanh(g * x) / g
    return (1.0 - m) * x + m * wet


# ---------------------------------------------------------------------- delay


def one_pole_lowpass_coeff(cutoff_hz: float, fs: float) -> float:
    """y += a (x - y),  a = 1 - exp(-2 pi fc / fs)."""
    fc = clamp_freq(cutoff_hz, fs)
    return 1.0 - math.exp(-2.0 * math.pi * fc / fs)


def ms_to_samples(ms: float, fs: float) -> int:
    """round(ms * fs / 1000) with ms first rounded to 0.001 ms (shared/CHAIN.md), so the
    plug-in, which stores parameters as float, lands on the same sample."""
    return int(round(round(float(ms), 3) * fs / 1000.0))


def process_delay(x: np.ndarray, fs: float, p: Delay) -> np.ndarray:
    """Feedback delay with a one-pole high-cut inside the loop.

    r[n]   = buf[n - D]                  (D = round(time_ms * fs / 1000))
    w[n]   = LP(r)[n]                    (one-pole, highcut_hz)
    buf[n] = x[n] + fb * w[n]
    y[n]   = (1 - m) x[n] + m w[n]
    """
    m = p.mix_pct / 100.0
    if m <= 0.0:
        return x
    n = x.shape[1]
    d = max(1, ms_to_samples(p.time_ms, fs))
    fb = p.feedback_pct / 100.0
    a = one_pole_lowpass_coeff(p.highcut_hz, fs)
    b_lp, a_lp = [a], [1.0, -(1.0 - a)]

    buf = np.zeros((x.shape[0], d + n))
    wet = np.zeros_like(x)
    zi = np.zeros((x.shape[0], 1))
    # Blocks no longer than D only read samples written in earlier blocks.
    for n0 in range(0, n, d):
        n1 = min(n0 + d, n)
        read = buf[:, n0:n1]  # == buf[:, d + t - d] for t in [n0, n1)
        w, zi = lfilter(b_lp, a_lp, read, axis=1, zi=zi)
        wet[:, n0:n1] = w
        buf[:, d + n0 : d + n1] = x[:, n0:n1] + fb * w
    return (1.0 - m) * x + m * wet


# --------------------------------------------------------------------- reverb

FDN_DELAYS_MS = (29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.7, 73.3)
_FDN_N = len(FDN_DELAYS_MS)
FDN_MATRIX = hadamard(_FDN_N).astype(np.float64) / math.sqrt(_FDN_N)
FDN_IN = np.ones(_FDN_N) / math.sqrt(_FDN_N)
FDN_OUT_L = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64) / math.sqrt(_FDN_N)
FDN_OUT_R = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64) / math.sqrt(_FDN_N)


def fdn_delays(fs: float) -> np.ndarray:
    return np.array([max(1, int(round(ms * fs / 1000.0))) for ms in FDN_DELAYS_MS])


def process_reverb(x: np.ndarray, fs: float, p: Reverb) -> np.ndarray:
    """8-line feedback delay network (Hadamard feedback matrix).

    Per line i: gain g_i = 10^(-3 d_i / (fs * decay_s)) (RT60 = decay_s at DC)
    and a one-pole low-pass at damping_hz. Input is the pre-delayed mid
    signal; L/R wet use orthogonal output sign patterns. The wet signal is
    scaled by sqrt(1 - mean(g)^2) so its energy is roughly independent of
    decay. y = (1 - m) dry + m wet.
    """
    m = p.mix_pct / 100.0
    if m <= 0.0:
        return x
    n = x.shape[1]
    d = fdn_delays(fs)
    g = 10.0 ** (-3.0 * d / (fs * max(p.decay_s, 1e-3)))
    norm = math.sqrt(max(1.0 - float(np.mean(g)) ** 2, 0.0))
    a = one_pole_lowpass_coeff(p.damping_hz, fs)
    b_lp, a_lp = [a], [1.0, -(1.0 - a)]

    mono = 0.5 * (x[0] + x[1])
    pre = ms_to_samples(p.predelay_ms, fs)
    inp = np.zeros(n)
    if pre < n:
        inp[pre:] = mono[: n - pre]

    max_d = int(d.max())
    buf = np.zeros((_FDN_N, max_d + n))
    taps = np.zeros((_FDN_N, n))
    zi = np.zeros((_FDN_N, 1))
    rows = np.arange(_FDN_N)[:, None]
    block = int(d.min())
    for n0 in range(0, n, block):
        n1 = min(n0 + block, n)
        t = np.arange(n0, n1)[None, :]
        read = buf[rows, max_d + t - d[:, None]]
        f, zi = lfilter(b_lp, a_lp, read, axis=1, zi=zi)
        f *= g[:, None]
        taps[:, n0:n1] = f
        buf[:, max_d + n0 : max_d + n1] = FDN_MATRIX @ f + FDN_IN[:, None] * inp[None, n0:n1]

    wet = np.vstack([FDN_OUT_L @ taps, FDN_OUT_R @ taps]) * norm
    return (1.0 - m) * x + m * wet


# --------------------------------------------------------------- doubler/width

# (base delay ms, modulation depth ms, LFO rate Hz) for the L and R voices.
DOUBLER_VOICES = ((14.0, 1.6, 0.31), (21.0, 2.1, 0.23))


def process_doubler(x: np.ndarray, fs: float, p: Doubler) -> np.ndarray:
    """Two modulated-delay voices (one per channel) + mid/side width.

    voice_c[n] = x_c(n - D_c(n)),  D_c(n) = (base + depth sin(2 pi rate n / fs)) ms,
    read with linear interpolation (zero before the start).
    y_c = x_c + m voice_c; then S *= width/100 with M = (L+R)/2, S = (L-R)/2.
    """
    m = p.mix_pct / 100.0
    w = p.width_pct / 100.0
    y = x
    n = x.shape[1]
    if m > 0.0:
        t = np.arange(n, dtype=np.float64)
        y = x.copy()
        for c, (base, depth, rate) in enumerate(DOUBLER_VOICES):
            delay = (base + depth * np.sin(2.0 * np.pi * rate * t / fs)) * fs / 1000.0
            voice = np.interp(t - delay, t, x[c], left=0.0, right=0.0)
            y[c] = x[c] + m * voice
    if w != 1.0:
        mid = 0.5 * (y[0] + y[1])
        side = 0.5 * (y[0] - y[1]) * w
        y = np.vstack([mid + side, mid - side])
    return y
