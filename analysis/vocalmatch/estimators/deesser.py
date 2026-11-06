"""De-esser estimator.

The chain's de-esser turns down everything above its crossover while the high band is
louder than the threshold, by at most the reduction. It leaves most frames alone and caps
the loudest high-band frames (the sibilants). The features are the upper quantiles, over
active 10 ms frames, of the high-band levels (4-7, 7-11 and 11-16 kHz, mid signal) and of
their tilt against the body of the voice (300-3000 Hz), each relative to its median: the
de-esser compresses the top of these distributions, an EQ only shifts them. The threshold is
absolute (dBFS of the high band entering the de-esser), pinned by where the compression
starts within the distribution of the simulated input.

Fit by simulation on the signal entering the de-esser (the dry vocal through the estimated
EQ, compressor and saturation): a grid over crossover x threshold x reduction with a
decimated detector, then Nelder-Mead at full rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import sosfilt

from ..chain import to_stereo
from ..chain.dynamics import (
    DEESSER_ATTACK_MS,
    DEESSER_RELEASE_MS,
    deesser_crossover_sos,
    smooth_branching,
    time_constant_coeff,
    to_db,
)
from ..features import power_db
from ..preset import DeEsser
from .common import band_filter

FRAME_MS = 10.0
BODY_BAND = (300.0, 3000.0)
HIGH_BANDS = ((4000.0, 7000.0), (7000.0, 11000.0), (11000.0, 16000.0))
QUANTILES = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.995])
ACTIVE_REL_DB = 35.0
FREQS_HZ = (3500.0, 4500.0, 5500.0, 6500.0, 8000.0, 10000.0)
THRESHOLDS_DB = np.arange(-54.0, -2.0, 4.0)
REDUCTIONS_DB = np.array([3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 24.0])
MIN_IMPROVEMENT_DB = 0.25
MIN_IMPROVEMENT_REL = 0.25
# Detection threshold: below it the fit is more likely explaining small errors of the EQ and
# compressor estimates than a de-esser (milestone 4's joint refinement revisits this).
MIN_CONFIDENCE = 0.35
BOUNDS = {"frequency_hz": (2000.0, 16000.0), "threshold_db": (-60.0, 0.0), "reduction_db": (0.0, 24.0)}


@dataclass
class DeEsserEstimate:
    deesser: DeEsser
    diagnostics: dict = field(default_factory=dict)


def _frame_energy(x: np.ndarray, n: int) -> np.ndarray:
    k = x.shape[-1] // n
    return (x[..., : k * n].reshape(*x.shape[:-1], k, n) ** 2).mean(axis=-1)


def _features(e_body: np.ndarray, e_highs: list[np.ndarray], e_all: np.ndarray) -> np.ndarray:
    lv = power_db(e_all)
    active = lv > np.percentile(lv, 95) - ACTIVE_REL_DB
    body = power_db(e_body[active])
    out = []
    for e in e_highs:
        h = power_db(e[active])
        # Shapes only (each quantile minus the median): an EQ error in the high band shifts
        # these distributions but does not change their shape, so it cannot pass for de-essing.
        qh = np.quantile(h, QUANTILES)
        qt = np.quantile(h - body, QUANTILES)
        out.append(qh[1:] - qh[0])
        out.append(qt[1:] - qt[0])
    return np.concatenate(out)


def signal_features(x: np.ndarray, fs: float) -> np.ndarray:
    st = to_stereo(x)
    mid = 0.5 * (st[0] + st[1])
    n = int(round(FRAME_MS * fs / 1000))
    return _features(_frame_energy(band_filter(mid, BODY_BAND, fs), n),
                     [_frame_energy(band_filter(mid, b, fs), n) for b in HIGH_BANDS], _frame_energy(mid, n))


def post_kernels(post, fs: float, seconds: float = 4.0) -> list[np.ndarray]:
    """Frame-energy envelopes (body band, high bands, broadband) of the mid impulse response
    of the linear function *post* (2, n) -> (2, n)."""
    n = int(round(FRAME_MS * fs / 1000))
    imp = np.zeros((2, int(seconds * fs)))
    imp[:, n // 2] = 1.0
    h = post(imp)
    hm = 0.5 * (h[0] + h[1])
    bands = [band_filter(hm, BODY_BAND, fs)] + [band_filter(hm, b, fs) for b in HIGH_BANDS] + [hm]
    kernels = []
    for b in bands:
        e = _frame_energy(b, n) * n  # energy per frame of the response to a unit impulse
        last = np.nonzero(e > e.max() * 1e-7)[0]
        kernels.append(e[: (last[-1] + 1) if len(last) else 1])
    return kernels


def _apply_kernel(e: np.ndarray, k: np.ndarray | None) -> np.ndarray:
    if k is None:
        return e
    from scipy.signal import fftconvolve

    return np.maximum(fftconvolve(e, k)[: len(e)], 0.0)


class _Simulator:
    def __init__(self, z: np.ndarray, fs: float, kernels: list[np.ndarray] | None = None):
        self.z = to_stereo(z)
        self.fs = fs
        self.n = int(round(FRAME_MS * fs / 1000))
        self.split: dict = {}
        self.k = kernels or [None] * (2 + len(HIGH_BANDS))

    def bands_for(self, freq: float):
        key = round(freq, 1)
        if key not in self.split:
            sos_lp, sos_hp = deesser_crossover_sos(freq, self.fs)
            low = sosfilt(sos_lp, self.z, axis=-1)
            high = sosfilt(sos_hp, self.z, axis=-1)
            lm, hm = 0.5 * (low[0] + low[1]), 0.5 * (high[0] + high[1])
            det = np.max(np.abs(high), axis=0)
            fb = [band_filter(v, BODY_BAND, self.fs) for v in (lm, hm)]
            fh = [[band_filter(v, b, self.fs) for v in (lm, hm)] for b in HIGH_BANDS]
            self.split = {key: (det, lm, hm, fb, fh)}  # keep one crossover
        return self.split[key]

    def min_threshold(self, freq: float) -> float:
        """The lowest threshold allowed at this crossover: the median 10 ms high-band peak level
        over active frames. Lower, the de-esser would act most of the time: a static high cut,
        which is the EQ's job (and would let the two trade off)."""
        det = self.bands_for(freq)[0]
        k = len(det) // self.n
        peaks = to_db(det[: k * self.n].reshape(k, self.n).max(axis=1))
        lm = self.bands_for(freq)[1]
        lv = power_db(_frame_energy(lm, self.n))[:k]
        active = lv > np.percentile(lv, 95) - ACTIVE_REL_DB
        return float(np.median(peaks[active])) if active.any() else BOUNDS["threshold_db"][0]

    def features(self, freq: float, threshold: float, reduction: float, decimation: int = 1) -> np.ndarray:
        det, lm, hm, fb, fh = self.bands_for(freq)
        fs = self.fs / decimation
        if decimation > 1:
            nb = len(det) // decimation
            d = det[: nb * decimation].reshape(nb, decimation).max(axis=1)
        else:
            d = det
        gr = np.clip(to_db(d) - threshold, 0.0, reduction)
        gr = smooth_branching(gr, time_constant_coeff(DEESSER_ATTACK_MS, fs), time_constant_coeff(DEESSER_RELEASE_MS, fs))
        g = 10.0 ** (-gr / 20.0)
        if decimation > 1:
            g = np.repeat(g, decimation)
            g = np.concatenate([g, np.full(len(det) - len(g), g[-1] if len(g) else 1.0)])
        n = self.n

        def y(pair):  # band of the de-essed signal ~= band(low) + g * band(high) (g is slow)
            return pair[0] + g * pair[1]

        k = self.k
        return _features(_apply_kernel(_frame_energy(y(fb), n), k[0]),
                         [_apply_kernel(_frame_energy(y(p), n), kk) for p, kk in zip(fh, k[1:-1])],
                         _apply_kernel(_frame_energy(lm + g * hm, n), k[-1]))


def estimate_deesser(reference: np.ndarray, pre: np.ndarray, fs: float, decimation: int = 8, polish_evals: int = 40,
                     post=None, gate: bool = True) -> DeEsserEstimate:
    """*pre*: the signal entering the de-esser (the dry vocal through the estimated EQ,
    compressor and saturation). *post*: optional linear function (2, n) -> (2, n), the
    effects estimated after it."""
    target = signal_features(reference, fs)
    sim = _Simulator(pre, fs, post_kernels(post, fs) if post is not None else None)
    none_err = float(np.sqrt(np.mean((sim.features(FREQS_HZ[0], 0.0, 0.0, 1) - target) ** 2)))

    def err(f, t, r, dec):
        return float(np.sqrt(np.mean((sim.features(f, t, r, dec) - target) ** 2)))

    best = (none_err, FREQS_HZ[0], 0.0, 0.0)
    for f in FREQS_HZ:
        t_min = sim.min_threshold(f)
        for t in THRESHOLDS_DB[THRESHOLDS_DB >= t_min]:
            for r in REDUCTIONS_DB:
                e = err(f, t, r, decimation)
                if e < best[0]:
                    best = (e, f, t, r)
    if best[3] == 0.0:  # nothing beats "no de-esser"
        return DeEsserEstimate(DeEsser(enabled=False, confidence=round(float(np.exp(-none_err)), 3)),
                               {"detected": False, "fit_error": round(none_err, 3)})

    def objective(zz):
        f = float(np.clip(2.0 ** zz[0], *BOUNDS["frequency_hz"]))
        t = float(np.clip(zz[1], max(BOUNDS["threshold_db"][0], sim.min_threshold(f)), BOUNDS["threshold_db"][1]))
        r = float(np.clip(zz[2], *BOUNDS["reduction_db"]))
        return err(f, t, r, 1)

    z0 = np.array([np.log2(best[1]), best[2], best[3]])
    res = minimize(objective, z0, method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([z0, z0 + np.diag([0.25, 3.0, 2.0])]), "maxfev": polish_evals,
                            "xatol": 0.02, "fatol": 1e-4})
    e_full0 = objective(z0)
    zz = res.x if res.fun <= e_full0 else z0
    e_fin = min(res.fun, e_full0)
    f = float(np.clip(2.0 ** zz[0], *BOUNDS["frequency_hz"]))
    t = float(np.clip(zz[1], max(BOUNDS["threshold_db"][0], sim.min_threshold(f)), BOUNDS["threshold_db"][1]))
    r = float(np.clip(zz[2], *BOUNDS["reduction_db"]))
    # The post-effect kernel is an approximation (~0.5 dB residual), so a de-esser must earn
    # its place: a clear improvement over none, absolute and relative.
    explained = 1.0 - e_fin / max(none_err, 1e-9)
    confidence = float(np.clip(explained, 0.0, 1.0) * np.exp(-e_fin / 1.0))
    enabled = bool(r >= 0.5 and (not gate or (none_err - e_fin >= MIN_IMPROVEMENT_DB
                                              and e_fin <= (1.0 - MIN_IMPROVEMENT_REL) * none_err
                                              and confidence >= MIN_CONFIDENCE)))
    d = DeEsser(enabled=enabled, confidence=round(confidence, 3), frequency_hz=round(f, 1), threshold_db=round(t, 2),
                reduction_db=round(r, 2) if enabled else 0.0)
    if not enabled:
        d = DeEsser(enabled=False, confidence=round(float(np.exp(-none_err)), 3))
    return DeEsserEstimate(d, {"detected": enabled, "fit_error": round(float(e_fin), 3), "fit_error_without": round(none_err, 3),
                               "evaluations": int(res.nfev) + len(FREQS_HZ) * len(THRESHOLDS_DB) * len(REDUCTIONS_DB)})
