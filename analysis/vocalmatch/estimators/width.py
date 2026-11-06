"""Doubler / width estimator.

The doubler adds a modulated, delayed copy of the vocal to each channel (14 and 21 ms), then
the width scales the side signal S = (L - R)/2. Features, per band and computed on each
signal itself: the side-to-mid energy ratio over loud frames (where the doubler's voices
sound) and over tail frames (15-45 dB below the loud level, where only the reverb's side
remains), and the left-right correlation over loud frames. The reverb's own side content is
known from the reverb estimate, so loud-frame side energy beyond it is the doubler, and the
tail side energy relative to the reverb's sets the width.

Fit by simulation on the signal entering the doubler (the dry vocal through the estimated
delay and reverb). y = z + m v is linear in the doubler mix and the width only scales S, so
every (mix, width) on a grid is scored from frame sums of products of z and v.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chain import to_stereo
from ..chain.effects import process_delay, process_doubler, process_reverb
from ..features import EPS, power_db
from ..preset import Delay, Doubler, Reverb
from .common import BANDS_HZ, band_filter

HOP_MS = 10.0
MIXES = np.linspace(0.0, 1.0, 51)
WIDTHS = np.linspace(0.0, 2.0, 51)
LOUD_REL_DB = 15.0
TAIL_REL_DB = (15.0, 45.0)
FLOOR_DB = -60.0  # side-to-mid ratios are clipped here (a mono signal has none)
MIN_SIDE_DB = -50.0  # below this the reference counts as mono: nothing to estimate


@dataclass
class WidthEstimate:
    doubler: Doubler
    diagnostics: dict = field(default_factory=dict)


def _frame_mean(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return x[: n * hop].reshape(n, hop).mean(axis=1)


def _bands(x: np.ndarray, fs: float) -> list[np.ndarray]:
    return [x] + [band_filter(x, b, fs) for b in BANDS_HZ]


def _masks(mid_energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lv = power_db(mid_energy)
    p95 = np.percentile(lv, 95, axis=-1, keepdims=True)
    loud = lv > p95 - LOUD_REL_DB
    tail = (lv < p95 - TAIL_REL_DB[0]) & (lv > p95 - TAIL_REL_DB[1])
    return loud, tail


def _features(em: np.ndarray, es: np.ndarray, ems: np.ndarray, masks) -> np.ndarray:
    """em, es, ems: (..., frames) frame energies of M, S and M*S for one band."""
    loud, tail = masks

    def ratio(mask):
        s = np.sum(np.where(mask, es, 0.0), axis=-1)
        m = np.sum(np.where(mask, em, 0.0), axis=-1)
        return np.maximum(power_db(s / np.maximum(m, EPS)), FLOOR_DB)

    # corr(L, R) with L = M + S, R = M - S: (E[M^2] - E[S^2]) / sqrt(E[L^2] E[R^2])
    m = np.sum(np.where(loud, em, 0.0), axis=-1)
    s = np.sum(np.where(loud, es, 0.0), axis=-1)
    x = np.sum(np.where(loud, ems, 0.0), axis=-1)
    el, er = m + s + 2 * x, m + s - 2 * x
    corr = (m - s) / np.sqrt(np.maximum(el * er, EPS))
    return np.stack([ratio(loud), ratio(tail), 20.0 * corr], axis=-1)  # corr scaled to ~dB weight


def signal_features(x: np.ndarray, fs: float) -> np.ndarray:
    st = to_stereo(x)
    hop = int(round(HOP_MS * fs / 1000))
    mid, side = 0.5 * (st[0] + st[1]), 0.5 * (st[0] - st[1])
    mb, sb = _bands(mid, fs), _bands(side, fs)
    masks = _masks(_frame_mean(mid * mid, hop))
    feats = [_features(_frame_mean(m * m, hop), _frame_mean(s * s, hop), _frame_mean(m * s, hop), masks)
             for m, s in zip(mb, sb)]
    return np.concatenate(feats)


def estimate_width(reference: np.ndarray, dry: np.ndarray, fs: float, delay: Delay | None = None,
                   reverb: Reverb | None = None, pre: np.ndarray | None = None) -> WidthEstimate:
    """*pre*: the signal entering the doubler (the dry vocal through the estimates of every
    effect before it); without it, the dry vocal through *delay* and *reverb*."""
    target = signal_features(reference, fs)
    ref_st = to_stereo(reference)
    side_db = float(power_db(np.mean((0.5 * (ref_st[0] - ref_st[1])) ** 2) / (np.mean((0.5 * (ref_st[0] + ref_st[1])) ** 2) + EPS)))
    if side_db < MIN_SIDE_DB:
        return WidthEstimate(Doubler(enabled=False), {"side_to_mid_db": round(side_db, 1), "detected": False,
                                                      "reason": "the reference is (nearly) mono"})

    z = to_stereo(dry if pre is None else pre)
    if pre is None and delay is not None and delay.enabled:
        z = process_delay(z, fs, delay)
    if pre is None and reverb is not None and reverb.enabled:
        z = process_reverb(z, fs, reverb)
    v = process_doubler(z, fs, Doubler(enabled=True, mix_pct=100.0, width_pct=100.0)) - z

    hop = int(round(HOP_MS * fs / 1000))
    zm, zs = 0.5 * (z[0] + z[1]), 0.5 * (z[0] - z[1])
    vm, vs = 0.5 * (v[0] + v[1]), 0.5 * (v[0] - v[1])
    sums = []
    for zmb, zsb, vmb, vsb in zip(_bands(zm, fs), _bands(zs, fs), _bands(vm, fs), _bands(vs, fs)):
        sums.append({k: _frame_mean(a * b, hop) for k, (a, b) in {
            "zmzm": (zmb, zmb), "zmvm": (zmb, vmb), "vmvm": (vmb, vmb), "zszs": (zsb, zsb), "zsvs": (zsb, vsb),
            "vsvs": (vsb, vsb), "zmzs": (zmb, zsb), "zmvs": (zmb, vsb), "vmzs": (vmb, zsb), "vmvs": (vmb, vsb)}.items()})

    def score(mixes: np.ndarray, widths: np.ndarray) -> np.ndarray:
        m_, w_ = mixes[:, None, None], widths[None, :, None]
        sims, masks = [], None
        for t in sums:  # M = zm + m vm, S = w (zs + m vs)
            em = t["zmzm"] + 2 * m_ * t["zmvm"] + m_ * m_ * t["vmvm"]
            es = w_ * w_ * (t["zszs"] + 2 * m_ * t["zsvs"] + m_ * m_ * t["vsvs"])
            ems = w_ * (t["zmzs"] + m_ * (t["zmvs"] + t["vmzs"]) + m_ * m_ * t["vmvs"])
            em = np.broadcast_to(em, (len(mixes), len(widths), em.shape[-1]))
            if masks is None:  # the masks follow the broadband mid level of each candidate
                masks = _masks(em)
            sims.append(_features(em, es, ems, masks))
        sim = np.concatenate(sims, axis=-1)
        return np.sqrt(np.mean((sim - target[None, None, :]) ** 2, axis=-1))

    err = score(MIXES, WIDTHS)
    i, j = np.unravel_index(int(np.argmin(err)), err.shape)
    fine_m = np.clip(MIXES[i] + np.linspace(-0.02, 0.02, 21), 0.0, 1.0)
    fine_w = np.clip(WIDTHS[j] + np.linspace(-0.04, 0.04, 21), 0.0, 2.0)
    err_f = score(fine_m, fine_w)
    k, l = np.unravel_index(int(np.argmin(err_f)), err_f.shape)
    mix, width = float(fine_m[k] * 100.0), float(fine_w[l] * 100.0)
    e_best = float(err_f[k, l])
    e_neutral = float(err[0, int(np.argmin(np.abs(WIDTHS - 1.0)))])
    explained = 1.0 - e_best / max(e_neutral, 1e-9)
    enabled = mix >= 1.0 or abs(width - 100.0) >= 2.0
    confidence = float(np.clip(explained, 0.0, 1.0) * np.exp(-e_best / 3.0)) if enabled else float(np.exp(-e_best / 3.0))
    return WidthEstimate(
        Doubler(enabled=enabled, confidence=round(confidence, 3), mix_pct=round(mix, 1), width_pct=round(width, 1)),
        {"detected": enabled, "side_to_mid_db": round(side_db, 1), "fit_error": round(e_best, 3),
         "fit_error_neutral": round(e_neutral, 3)},
    )
