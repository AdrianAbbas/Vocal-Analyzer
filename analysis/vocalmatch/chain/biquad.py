"""RBJ "Audio EQ Cookbook" biquads.

These exact formulas are what the C++ plugin must use too (see shared/CHAIN.md),
so that the Python and C++ chains agree sample-for-sample (up to float precision).
Coefficients are returned normalised (a0 == 1) as one SOS row:
``[b0, b1, b2, 1, a1, a2]``.
"""

from __future__ import annotations

import numpy as np

KINDS = ("peak", "low_shelf", "high_shelf", "highpass", "lowpass")

MAX_FREQ_FRACTION = 0.49  # biquad centre frequencies are clamped below this * fs


def clamp_freq(freq_hz: float, fs: float) -> float:
    return float(min(max(freq_hz, 1.0), MAX_FREQ_FRACTION * fs))


def rbj_sos(kind: str, freq_hz: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """Return a normalised SOS row for one RBJ biquad."""
    f0 = clamp_freq(freq_hz, fs)
    q = max(float(q), 1e-3)
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    sw = np.sin(w0)
    alpha = sw / (2.0 * q)

    if kind == "peak":
        b0 = 1.0 + alpha * a_lin
        b1 = -2.0 * cw
        b2 = 1.0 - alpha * a_lin
        a0 = 1.0 + alpha / a_lin
        a1 = -2.0 * cw
        a2 = 1.0 - alpha / a_lin
    elif kind == "low_shelf":
        sq = 2.0 * np.sqrt(a_lin) * alpha
        b0 = a_lin * ((a_lin + 1) - (a_lin - 1) * cw + sq)
        b1 = 2.0 * a_lin * ((a_lin - 1) - (a_lin + 1) * cw)
        b2 = a_lin * ((a_lin + 1) - (a_lin - 1) * cw - sq)
        a0 = (a_lin + 1) + (a_lin - 1) * cw + sq
        a1 = -2.0 * ((a_lin - 1) + (a_lin + 1) * cw)
        a2 = (a_lin + 1) + (a_lin - 1) * cw - sq
    elif kind == "high_shelf":
        sq = 2.0 * np.sqrt(a_lin) * alpha
        b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cw + sq)
        b1 = -2.0 * a_lin * ((a_lin - 1) + (a_lin + 1) * cw)
        b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cw - sq)
        a0 = (a_lin + 1) - (a_lin - 1) * cw + sq
        a1 = 2.0 * ((a_lin - 1) - (a_lin + 1) * cw)
        a2 = (a_lin + 1) - (a_lin - 1) * cw - sq
    elif kind == "highpass":
        b0 = (1.0 + cw) / 2.0
        b1 = -(1.0 + cw)
        b2 = (1.0 + cw) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
    elif kind == "lowpass":
        b0 = (1.0 - cw) / 2.0
        b1 = 1.0 - cw
        b2 = (1.0 - cw) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
    else:
        raise ValueError(f"unknown biquad kind {kind!r}; expected one of {KINDS}")

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def sos_power_response(sos: np.ndarray, freqs_hz: np.ndarray, fs: float) -> np.ndarray:
    """|H(f)|^2 of a cascade of SOS rows, evaluated at arbitrary frequencies."""
    sos = np.atleast_2d(sos)
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / fs
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    h = np.ones_like(z1)
    for b0, b1, b2, _, a1, a2 in sos:
        h *= (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
    return np.abs(h) ** 2


def sos_response_db(sos: np.ndarray, freqs_hz: np.ndarray, fs: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(sos_power_response(sos, freqs_hz, fs), 1e-30))
