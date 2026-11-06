"""Reverb estimator.

The reverb shows where the vocal falls silent: the direct sound stops, the tail decays. The
features are the *fastest level drops*: for every loud 10 ms frame t of the mid signal (four
bands and broadband) the drop L(t + tau) - L(t) for tau = 30 ms ... 1 s, and their lowest
quantiles. Those come from phrase ends and note releases; without reverb the level falls
straight into the noise, with reverb it can only fall as fast as the tail decays (its slope
gives the decay time, the faster fall in the high bands the damping), it first holds until
predelay + the first delay line have passed (the predelay), and the depth of the first drop
gives the mix. No phrase detection is needed, so the reference and the simulation are scored
exactly alike.

All four settings are fitted at once by simulation: the chain's own reverb on (a proxy of)
the signal entering it, the dry vocal through the delay already estimated. The reverb is
linear in the mix and a predelay only shifts the wet signal, so one FDN render per
(decay, damping) serves every (predelay, mix) through frame energy sums.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..chain import to_stereo
from ..chain.effects import process_delay, process_reverb
from ..features import power_db
from ..preset import Delay, Reverb
from .common import BANDS_HZ, band_filter, envelope_db, mid_side

HOP_MS = 10.0
TAUS_MS = np.array([30, 60, 100, 160, 250, 400, 650, 1000], float)
QUANTILES = np.array([0.01, 0.02, 0.04, 0.08, 0.15, 0.30])
LOUD_REL_DB = 25.0  # drops are measured from frames within this of the signal's 95th percentile
PREDELAYS_MS = np.array([0, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100, 125, 150, 200, 250], float)
COARSE_MIXES = np.linspace(0.0, 1.0, 51)
FLOOR_DB = -100.0
BOUNDS = {"decay_s": (0.1, 10.0), "damping_hz": (500.0, 20000.0), "predelay_ms": (0.0, 250.0), "mix_pct": (0.0, 100.0)}


@dataclass
class ReverbEstimate:
    reverb: Reverb
    diagnostics: dict = field(default_factory=dict)


def _frame_sums(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    """Per-frame mean of x (frames centred at k*hop, width win), like common.envelope_db."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n_frames = len(x) // hop
    centres = np.arange(n_frames) * hop
    lo = np.clip(centres - win // 2, 0, len(x))
    hi = np.clip(centres + win // 2, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def drop_features(levels_db: np.ndarray) -> np.ndarray:
    """levels_db: (..., n_frames) -> (..., len(TAUS_MS) * len(QUANTILES)) lowest quantiles of
    L(t + tau) - L(t) over loud frames t (rows are scored independently)."""
    lv = np.maximum(levels_db, FLOOR_DB)
    thr = np.percentile(lv, 95, axis=-1, keepdims=True) - LOUD_REL_DB
    out = []
    for tau in TAUS_MS:
        k = int(round(tau / HOP_MS))
        d = lv[..., k:] - lv[..., :-k]
        d = np.where(lv[..., :-k] > thr, d, np.inf)
        d = np.sort(d, axis=-1)
        count = np.sum(np.isfinite(d), axis=-1, keepdims=True)
        idx = np.clip(np.floor(QUANTILES * (count - 1)).astype(int), 0, d.shape[-1] - 1)
        out.append(np.take_along_axis(d, idx, axis=-1))
    return np.concatenate(out, axis=-1)


def _signal_bands(mid: np.ndarray, fs: float) -> list[np.ndarray]:
    return [mid] + [band_filter(mid, b, fs) for b in BANDS_HZ]


def signal_features(mid: np.ndarray, fs: float) -> np.ndarray:
    return np.concatenate([drop_features(envelope_db(b, fs, HOP_MS)) for b in _signal_bands(mid, fs)])


class _Simulator:
    """y = (1 - m) x + m shift(wet, p): drop features for many (p, m) from one wet render."""

    def __init__(self, proxy_mid: np.ndarray, fs: float):
        self.fs = fs
        self.x = proxy_mid
        self.hop = int(round(HOP_MS * fs / 1000))
        self.win = 2 * self.hop
        self.x_bands = _signal_bands(proxy_mid, fs)
        self.exx = [_frame_sums(b * b, self.hop, self.win) for b in self.x_bands]
        self.wet_cache: dict = {}

    def wet_bands(self, decay_s: float, damping_hz: float) -> list[np.ndarray]:
        key = (round(decay_s, 4), round(damping_hz, 1))
        if key not in self.wet_cache:
            p = Reverb(enabled=True, decay_s=decay_s, predelay_ms=0.0, mix_pct=100.0, damping_hz=damping_hz)
            wet = process_reverb(np.vstack([self.x, self.x]), self.fs, p)
            self.wet_cache = {key: _signal_bands(0.5 * (wet[0] + wet[1]), self.fs)}  # keep one
        return self.wet_cache[key]

    def features(self, decay_s: float, damping_hz: float, predelay_ms: float, mixes: np.ndarray) -> np.ndarray:
        wets = self.wet_bands(decay_s, damping_hz)
        shift = int(round(predelay_ms * self.fs / 1000))
        a, m = (1.0 - mixes)[:, None], mixes[:, None]
        feats = []
        for xb, exx, wb in zip(self.x_bands, self.exx, wets):
            w = np.concatenate([np.zeros(shift), wb[: len(wb) - shift]]) if shift else wb
            eww = _frame_sums(w * w, self.hop, self.win)
            exw = _frame_sums(xb * w, self.hop, self.win)
            energy = a * a * exx[None, :] + 2 * a * m * exw[None, :] + m * m * eww[None, :]
            feats.append(drop_features(power_db(np.maximum(energy, 0.0))))
        return np.concatenate(feats, axis=1)  # (mixes, features)

    def best(self, decay_s, damping_hz, target, predelays=PREDELAYS_MS, mixes=COARSE_MIXES):
        best = (np.inf, 0.0, 0.0)
        for p_ms in predelays:
            f = self.features(decay_s, damping_hz, p_ms, mixes)
            err = np.sqrt(np.mean((f - target[None, :]) ** 2, axis=1))
            j = int(np.argmin(err))
            if err[j] < best[0]:
                best = (float(err[j]), float(p_ms), float(mixes[j] * 100.0))
        return best


def estimate_reverb(reference: np.ndarray, dry: np.ndarray, fs: float, delay: Delay | None = None,
                    max_evals: int = 24, pre: np.ndarray | None = None) -> ReverbEstimate:
    """*pre*: the signal entering the reverb (the dry vocal through the estimates of every
    effect before it); without it, the dry vocal through *delay*."""
    ref_mid, _ = mid_side(reference)
    target = signal_features(ref_mid, fs)

    # Proxy of the signal entering the reverb: the dry vocal through the estimated delay.
    proxy = to_stereo(dry if pre is None else pre)
    if pre is None and delay is not None and delay.enabled:
        proxy = process_delay(proxy, fs, delay)
    proxy_mid = 0.5 * (proxy[0] + proxy[1])
    sim = _Simulator(proxy_mid, fs)
    none_err = float(np.sqrt(np.mean((signal_features(proxy_mid, fs) - target) ** 2)))

    # Coarse grid over decay x damping (predelay coarse), then Nelder-Mead.
    coarse_p = PREDELAYS_MS
    grid = []
    for decay in (0.3, 0.7, 1.3, 2.2, 3.5, 6.0):
        for damping in (2500.0, 7000.0, 16000.0):
            grid.append((sim.best(decay, damping, target, coarse_p)[0], decay, damping))
    grid.sort()
    _, d0, h0 = grid[0]
    z0 = np.array([np.log2(d0), np.log2(h0)])

    def objective(z):
        decay = float(np.clip(2.0 ** z[0], *BOUNDS["decay_s"]))
        damping = float(np.clip(2.0 ** z[1], *BOUNDS["damping_hz"]))
        return sim.best(decay, damping, target, coarse_p)[0]

    res = minimize(objective, z0, method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([z0, z0 + np.diag([0.5, 0.7])]), "maxfev": max_evals,
                            "xatol": 0.02, "fatol": 1e-4})
    decay = float(np.clip(2.0 ** res.x[0], *BOUNDS["decay_s"]))
    damping = float(np.clip(2.0 ** res.x[1], *BOUNDS["damping_hz"]))
    _, predelay, mix = sim.best(decay, damping, target)  # fine predelay grid
    fine = np.clip(np.linspace(mix - 2.0, mix + 2.0, 41), 0.0, 100.0) / 100.0
    err, predelay, mix = sim.best(decay, damping, target, [predelay], fine)

    explained = 1.0 - err / max(none_err, 1e-9)
    confidence = float(np.clip(explained, 0.0, 1.0) * np.exp(-err / 3.0))
    reverb = Reverb(enabled=mix >= 1.0, confidence=round(confidence, 3), decay_s=round(decay, 3),
                    predelay_ms=round(predelay, 1), mix_pct=round(mix, 2), damping_hz=round(damping, 1))
    return ReverbEstimate(reverb, {
        "detected": reverb.enabled,
        "profile_error_db": round(err, 3),
        "profile_error_without_reverb_db": round(none_err, 3),
        "evaluations": int(res.nfev + len(grid)),
    })
