"""Saturation estimator.

A memoryless harmonic signal stays harmonic through any curve, so distortion cannot be read
from the harmonics of a voice alone. What tanh saturation changes is how the waveform
depends on level: loud frames lose their sample peaks (lower crest factor) and gain upper
harmonics (more high-frequency energy relative to the body of the voice), quiet frames stay
as they were. The features are the waveform crest factor (10 ms frames, sample peak over
RMS) and the high-frequency tilt (4-12 kHz over 200-2000 Hz) in frame groups ranked by level.

Fit by simulation on the signal entering the saturation (the dry vocal through the
estimated EQ and compressor), followed by the effects already estimated after it (delay,
reverb, doubler: linear, so y = (1 - m) post(z) + m post(tanh(g z)/g) and one render per
drive serves every mix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chain import to_stereo
from ..chain.effects import db_to_lin
from ..features import EPS, power_db
from ..preset import Saturation
from .common import band_filter

FRAME_MS = 10.0
GROUPS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 0.60))  # level-rank fractions of active frames
ACTIVE_REL_DB = 30.0
DRIVES_DB = np.arange(0.0, 36.0 + 1e-9, 3.0)
MIXES = np.linspace(0.0, 1.0, 21)
MIN_CONFIDENCE = 0.35
LOW_BAND, HIGH_BAND = (200.0, 2000.0), (4000.0, 12000.0)


@dataclass
class SaturationEstimate:
    saturation: Saturation
    diagnostics: dict = field(default_factory=dict)


class _Frames:
    """Per-frame peak, energy and band energies of y = (1 - m) A + m B for many m."""

    def __init__(self, fs: float):
        self.n = int(round(FRAME_MS * fs / 1000))
        self.fs = fs

    def split(self, x: np.ndarray) -> np.ndarray:
        k = x.shape[-1] // self.n
        return x[..., : k * self.n].reshape(*x.shape[:-1], k, self.n)


def _features(peak: np.ndarray, energy: np.ndarray, e_low: np.ndarray, e_high: np.ndarray) -> np.ndarray:
    """Rows of frame statistics (..., frames) -> (..., features)."""
    lv = power_db(energy)
    order = np.argsort(-lv, axis=-1)
    ref = np.take_along_axis(lv, order[..., :1], axis=-1)
    n_active = np.sum(lv > ref - ACTIVE_REL_DB, axis=-1, keepdims=True)
    crest = 20.0 * np.log10(np.maximum(peak, EPS)) - 0.5 * lv
    tilt = power_db(e_high) - power_db(e_low)
    c_sorted = np.take_along_axis(crest, order, axis=-1)
    t_sorted = np.take_along_axis(tilt, order, axis=-1)
    idx = np.arange(lv.shape[-1])
    feats = []
    for lo, hi in GROUPS:
        sel = (idx >= np.floor(lo * n_active)) & (idx < np.maximum(np.ceil(hi * n_active), np.floor(lo * n_active) + 1))
        cnt = np.maximum(sel.sum(axis=-1), 1)
        feats.append(np.sum(np.where(sel, c_sorted, 0.0), axis=-1) / cnt)
        feats.append(np.sum(np.where(sel, t_sorted, 0.0), axis=-1) / cnt)
    f = np.stack(feats, axis=-1)
    # Tilt relative to the quietest group: an EQ error shifts every group's tilt alike.
    f[..., 1::2] -= f[..., -1:]
    return f[..., :-1]


def signal_features(x: np.ndarray, fs: float) -> np.ndarray:
    st = to_stereo(x)
    fr = _Frames(fs)
    det = np.max(np.abs(st), axis=0)
    mid = 0.5 * (st[0] + st[1])
    peak = fr.split(det).max(axis=-1)
    energy = (fr.split(st) ** 2).mean(axis=(0, -1))
    e_low = (fr.split(band_filter(mid, LOW_BAND, fs)) ** 2).mean(axis=-1)
    e_high = (fr.split(band_filter(mid, HIGH_BAND, fs)) ** 2).mean(axis=-1)
    return _features(peak, energy, e_low, e_high)


def estimate_saturation(reference: np.ndarray, pre: np.ndarray, fs: float, post=None) -> SaturationEstimate:
    """*pre*: the signal entering the saturation (dry vocal through the estimated EQ and
    compressor). *post*: optional linear function (2, n) -> (2, n), the effects estimated after
    the saturation."""
    target = signal_features(reference, fs)
    z = to_stereo(pre)
    post = post or (lambda v: v)
    fr = _Frames(fs)
    a = post(z)
    a_mid = 0.5 * (a[0] + a[1])
    a_frames = fr.split(a)
    a_low = fr.split(band_filter(a_mid, LOW_BAND, fs))
    a_high = fr.split(band_filter(a_mid, HIGH_BAND, fs))
    m = MIXES[:, None, None]

    def score(drive_db: float, mixes: np.ndarray) -> np.ndarray:
        g = db_to_lin(drive_db)
        b = post(np.tanh(g * z) / g)
        b_mid = 0.5 * (b[0] + b[1])
        mm = mixes[:, None, None]
        out = []
        # The mixes are scored in chunks to keep memory small.
        for c0 in range(0, len(mixes), 7):
            mc = mm[c0 : c0 + 7]
            y = (1.0 - mc[:, :, None]) * a_frames[None] + mc[:, :, None] * fr.split(b)[None]  # (m, 2, frames, n)
            peak = np.abs(y).max(axis=(1, 3))
            energy = (y**2).mean(axis=(1, 3))
            yl = (1.0 - mc) * a_low[None] + mc * fr.split(band_filter(b_mid, LOW_BAND, fs))[None]
            yh = (1.0 - mc) * a_high[None] + mc * fr.split(band_filter(b_mid, HIGH_BAND, fs))[None]
            f = _features(peak, energy, (yl**2).mean(axis=-1), (yh**2).mean(axis=-1))
            out.append(np.sqrt(np.mean((f - target[None, :]) ** 2, axis=-1)))
        return np.concatenate(out)

    del m
    none_err = float(np.sqrt(np.mean((signal_features(a, fs) - target) ** 2)))
    grid = np.array([score(d, MIXES) for d in DRIVES_DB])  # (drives, mixes)
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    # Fine pass around the best grid point.
    best = (float(grid[i, j]), float(DRIVES_DB[i]), float(MIXES[j]))
    fine_mixes = np.clip(MIXES[j] + np.linspace(-0.05, 0.05, 11), 0.0, 1.0)
    for d in np.clip(DRIVES_DB[i] + np.arange(-2.5, 2.51, 0.5), 0.0, 36.0):
        e = score(float(d), fine_mixes)
        k = int(np.argmin(e))
        if e[k] < best[0]:
            best = (float(e[k]), float(d), float(fine_mixes[k]))
    err, drive, mix = best
    mix_pct = mix * 100.0
    explained = 1.0 - err / max(none_err, 1e-9)
    confidence = float(np.clip(explained, 0.0, 1.0) * np.exp(-err / 1.0))
    # Detection threshold: mild saturation moves these features less than small errors of the
    # compressor estimate do (attack time changes the crest factor too), so a weak fit is
    # treated as no saturation (milestone 4's joint refinement revisits this).
    enabled = mix_pct >= 1.0 and confidence >= MIN_CONFIDENCE
    if not enabled:
        confidence = float(np.exp(-none_err / 1.0))
    sat = Saturation(enabled=enabled, confidence=round(confidence, 3), drive_db=round(drive, 2) if enabled else 0.0,
                     mix_pct=round(mix_pct, 1) if enabled else 0.0)
    return SaturationEstimate(sat, {"detected": enabled, "fit_error": round(err, 3), "fit_error_without": round(none_err, 3)})
