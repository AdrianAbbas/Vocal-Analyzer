"""Delay estimator.

1. Time: the delay's wet signal is a (low-passed) copy of the vocal D samples later, so the
   reference's autocorrelation has a sharp peak at lag D that the dry vocal's does not.
   Search 40 ms - 2 s, per band, after removing the slowly varying part of each
   autocorrelation (see find_delay_time). With a song tempo, a time within 2 % of a note
   value is snapped to it.
2. Feedback, mix and high-cut: simulate the chain's delay on the dry vocal and match the
   band-limited autocorrelation at D, 2D and 3D (four bands). The echo train's decay across
   k gives the feedback, its decay across frequency the high-cut, its height the mix. The
   output is linear in the mix, so for each (feedback, high-cut) the mix is solved in closed
   form over a fine grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..chain.effects import FDN_DELAYS_MS, one_pole_lowpass_coeff, process_delay
from ..features import to_mono
from ..preset import Delay
from .common import BANDS_HZ, autocorr_at, autocorr_fft, band_filter

MIN_LAG_MS = 40.0
MAX_LAG_MS = 2000.0
MIN_Z = 6.0  # an echo stands this many robust standard deviations above the noise, in two bands or more
NOTE_SNAP_TOL = 0.02

NOTE_VALUES = {  # in quarter notes
    "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125,
    **{f"1/{d}D": 1.5 * q for d, q in (("2", 2.0), ("4", 1.0), ("8", 0.5), ("16", 0.25))},
    **{f"1/{d}T": q * 2.0 / 3.0 for d, q in (("2", 2.0), ("4", 1.0), ("8", 0.5), ("16", 0.25))},
}

BOUNDS = {"feedback_pct": (0.0, 95.0), "mix_pct": (0.0, 100.0), "highcut_hz": (500.0, 20000.0)}


@dataclass
class DelayEstimate:
    delay: Delay
    diagnostics: dict = field(default_factory=dict)


def _features(bands: list[np.ndarray], d: int) -> np.ndarray:
    return np.concatenate([autocorr_at(b, [d, 2 * d, 3 * d]) for b in bands])


def _lag_highpass(r: np.ndarray, half: int) -> np.ndarray:
    """Remove the slowly varying part of an autocorrelation (moving average over +-half lags):
    an echo is a sharp peak, a voice's own long-lag correlation is not."""
    from scipy.ndimage import uniform_filter1d

    return r - uniform_filter1d(r, size=2 * half + 1, mode="nearest")


def find_delay_time(reference: np.ndarray, dry: np.ndarray, fs: float, exclude_fdn: bool = True) -> tuple[int | None, dict]:
    """The echo lag in samples, or None.

    Per band (BANDS_HZ), the reference's and the dry vocal's autocorrelations are high-passed
    in the lag domain, and z(lag) = (ref - max(dry, 0)) / robust spread. An echo is a sharp
    peak the dry vocal does not have, in at least two bands. A sung note correlates with
    itself at multiples of its pitch period; that shows in both signals and is removed by
    the comparison (when the dry vocal is the same performance) and by the consistency test
    (when it is not: consistent_over_time). Candidates that carry the reverb's signature (other peaks at the same
    predelay plus the other FDN line lengths: its early reflections) are rejected."""
    from scipy.ndimage import maximum_filter1d

    ref = to_mono(reference)
    dry_m = to_mono(dry)
    max_lag = int(MAX_LAG_MS * fs / 1000)
    lo = int(MIN_LAG_MS * fs / 1000)
    half = max(2, int(0.002 * fs))
    zs, signed = [], np.zeros(max_lag + 1)
    for band in BANDS_HZ:
        hr = _lag_highpass(autocorr_fft(band_filter(ref, band, fs), max_lag), half)
        hd = _lag_highpass(autocorr_fft(band_filter(dry_m, band, fs), max_lag), half)
        ex = hr - np.maximum(hd, 0.0)
        spread = float(np.median(np.abs(ex[lo:] - np.median(ex[lo:])))) * 1.4826 + 1e-9
        signed += (hr - hd) / spread
        # +-2 samples: the delay's low-pass shifts the echo by its group delay, more in low bands
        zs.append(maximum_filter1d(ex / spread, size=5))
    z = np.array(zs)
    z[:, :lo] = 0.0
    strong = np.sum(z >= MIN_Z, axis=0)
    score = np.sum(np.clip(z, 0.0, 50.0), axis=0)
    cand = np.where(strong >= 2, score, 0.0)
    # The chain's reverb, seen in the mid signal: early reflections at predelay + lines 0, 3, 4
    # and 7 (the lines whose left and right output signs agree), signs + - + -, of similar size.
    lines = np.array([int(round(FDN_DELAYS_MS[i] * fs / 1000)) for i in (0, 3, 4, 7)])
    max_pre = int(0.25 * fs)

    def strength(lag: int) -> float:
        w = signed[max(0, lag - 2) : lag + 3]
        return float(np.max(np.abs(w))) if len(w) else 0.0

    def reverb_signature(d: int) -> bool:
        a_d = strength(d)
        for li in lines:
            p = d - li  # this candidate as predelay + line li
            if p < 0 or p > max_pre:
                continue
            hits = sum(strength(p + lj) >= max(MIN_Z, 0.25 * a_d) for lj in lines if lj != li and p + lj <= max_lag)
            if hits >= 2:
                return True
        return False

    band_sigs = [band_filter(ref, band, fs) for band in BANDS_HZ]
    inconsistent = []
    rejected = []
    order = np.argsort(-cand)
    for d in order[:200]:
        d = int(d)
        if cand[d] <= 0:
            break
        if any(abs(d - r) <= 5 for r in rejected):
            continue
        if any(abs(d - r) <= 5 for r in inconsistent):
            continue
        if exclude_fdn and reverb_signature(d):
            rejected.append(d)
            continue
        if len(inconsistent) >= MAX_INCONSISTENT:
            break
        w = signed[d - 3 : d + 4]
        d_exact = d - 3 + int(np.argmax(w))
        if not consistent_over_time(band_sigs, d_exact, fs):
            inconsistent.append(d)
            continue
        # the echo's own lag: the strongest sample of the (unfiltered) excess near the peak
        w = signed[d - 3 : d + 4]
        d = d - 3 + int(np.argmax(w))
        return d, {"peak_lag_ms": round(d * 1000 / fs, 2), "peak_score": round(float(cand[d]), 1),
                   "bands_above": int(strong[d]), "rejected_as_reverb_ms": [round(r * 1000 / fs, 1) for r in rejected],
                   "rejected_as_inconsistent_ms": [round(r * 1000 / fs, 1) for r in inconsistent]}
    return None, {"peak_lag_ms": None, "rejected_as_reverb_ms": [round(r * 1000 / fs, 1) for r in rejected],
                  "rejected_as_inconsistent_ms": [round(r * 1000 / fs, 1) for r in inconsistent]}


CHUNK_S = 2.0
CHUNK_MIN_Z = 1.5
CHUNK_MIN_FRACTION = 0.8
CHUNK_MIN_MEDIAN = 2.0
MAX_INCONSISTENT = 30


def chunk_scores(band_sigs: list[np.ndarray], lag: int, fs: float) -> list[float]:
    """Per chunk of the reference (CHUNK_S, or 3 lags if longer; loud chunks only), the median
    over bands of the lag-high-passed autocorrelation at *lag*, in units of its spread."""
    from scipy.signal import fftconvolve

    half = max(2, int(0.002 * fs))
    w = 4 * half
    n = int(max(CHUNK_S, 3 * lag / fs) * fs)
    mono = band_sigs[0]
    starts = list(range(lag + w, len(mono) - n + 1, n))
    if not starts:
        return []
    energy = np.array([sum(float(np.dot(b[a : a + n], b[a : a + n])) for b in band_sigs) for a in starts])
    loud = energy > energy.max() * 10 ** (-20 / 10)
    out = []
    for a, ok in zip(starts, loud):
        if not ok:
            continue
        zs = []
        for y in band_sigs:
            seg = y[a : a + n]
            ctx = y[a - lag - w : a + n - lag + w]
            c = fftconvolve(ctx, seg[::-1], mode="valid")[::-1] / (float(np.dot(seg, seg)) + 1e-12)
            hp = _lag_highpass(c, half)  # index k <-> lag (lag - w + k)
            zs.append(float(hp[w - 2 : w + 3].max() / (np.std(hp) + 1e-12)))
        out.append(float(np.median(zs)))
    return out


def consistent_over_time(band_sigs: list[np.ndarray], lag: int, fs: float) -> bool:
    """An echo copies all the material, so its peak is there in every part of the song; a
    peak from the voice's own pitch periodicity (notes whose period divides the lag) is not."""
    z = chunk_scores(band_sigs, lag, fs)
    if len(z) < 2:
        return True  # too short to tell
    z = np.array(z)
    return bool(np.mean(z >= CHUNK_MIN_Z) >= CHUNK_MIN_FRACTION and np.median(z) >= CHUNK_MIN_MEDIAN)


def snap_to_tempo(time_ms: float, tempo_bpm: float | None) -> tuple[float, str | None]:
    if not tempo_bpm:
        return time_ms, None
    quarter = 60000.0 / tempo_bpm
    best = min(NOTE_VALUES.items(), key=lambda kv: abs(kv[1] * quarter / time_ms - 1.0))
    t = best[1] * quarter
    if abs(t / time_ms - 1.0) <= NOTE_SNAP_TOL and 1.0 <= t <= 2000.0:
        return t, best[0]
    return time_ms, None


def estimate_delay(reference: np.ndarray, dry: np.ndarray, fs: float, tempo_bpm: float | None = None,
                   max_evals: int = 80, pre: np.ndarray | None = None) -> DelayEstimate:
    """*pre*: the signal entering the delay (the dry vocal through the estimates of every
    effect before it) for the feedback / mix / high-cut fit; the dry vocal by default. The
    time is found against the dry vocal's own autocorrelation."""
    d, diag = find_delay_time(reference, dry, fs)
    if d is None:
        return DelayEstimate(Delay(enabled=False, confidence=0.0), {**diag, "detected": False})

    ref_bands = [band_filter(to_mono(reference), b, fs) for b in BANDS_HZ]
    target = _features(ref_bands, d)  # features at the peak lag and its multiples
    x = to_mono(dry if pre is None else pre)
    x_bands = [band_filter(x, b, fs) for b in BANDS_HZ]
    lags = [0, d, 2 * d, 3 * d]
    mixes = np.linspace(0.0, 1.0, 101)

    def fit(delay_samples: int):
        time_ms = delay_samples * 1000.0 / fs
        cache: dict = {}

        def best_mix(z) -> tuple[float, float]:
            fb = float(np.clip(z[0], *BOUNDS["feedback_pct"]))
            hc = float(np.clip(2.0 ** z[1], *BOUNDS["highcut_hz"]))
            key = (round(fb, 3), round(hc, 1))
            if key in cache:
                return cache[key]
            # The delay is linear and time-invariant: filtering then delaying == delaying then filtering.
            p = Delay(enabled=True, time_ms=time_ms, feedback_pct=fb, mix_pct=100.0, highcut_hz=hc)
            feats = np.zeros((len(mixes), 3 * len(x_bands)))
            for i, xb in enumerate(x_bands):
                wb = process_delay(np.vstack([xb, xb]), fs, p)[0]

                def c(u, v, lag):
                    return float(np.dot(u[lag:], v[: len(v) - lag])) if lag else float(np.dot(u, v))

                # r_y(L) for y = (1-m) x + m w, from the correlation terms at each lag.
                xx = np.array([c(xb, xb, L) for L in lags])
                ww = np.array([c(wb, wb, L) for L in lags])
                xw = np.array([c(xb, wb, L) + c(wb, xb, L) for L in lags])
                a, m = (1.0 - mixes)[:, None], mixes[:, None]
                r = a * a * xx + a * m * xw + m * m * ww
                feats[:, 3 * i : 3 * i + 3] = r[:, 1:] / np.maximum(r[:, :1], 1e-30)
            err = np.sqrt(np.mean((feats - target[None, :]) ** 2, axis=1))
            j = int(np.argmin(err))
            cache[key] = (float(err[j]), float(mixes[j] * 100.0))
            return cache[key]

        # Coarse grid, then Nelder-Mead on (feedback, log2 high-cut).
        grid = [(best_mix([fb, np.log2(hc)])[0], fb, hc) for fb in (0.0, 25.0, 50.0) for hc in (2000.0, 5000.0, 12000.0)]
        _, fb0, hc0 = min(grid)
        z0 = np.array([fb0, np.log2(hc0)])
        res = minimize(lambda z: best_mix(z)[0], z0, method="Nelder-Mead",
                       options={"initial_simplex": np.vstack([z0, z0 + np.diag([10.0, 0.7])]), "maxfev": max_evals,
                                "xatol": 0.05, "fatol": 1e-5})
        err, mix = best_mix(res.x)
        fb = float(np.clip(res.x[0], *BOUNDS["feedback_pct"]))
        hc = float(np.clip(2.0 ** res.x[1], *BOUNDS["highcut_hz"]))
        return err, fb, mix, hc, int(res.nfev)

    # The in-loop low-pass delays the echo by its group delay (1-a)/a samples, so the
    # autocorrelation peak sits slightly after D: fit, correct D, fit again.
    err, fb, mix, hc, evals = fit(d)
    a_lp = one_pole_lowpass_coeff(hc, fs)
    shift = int(round((1.0 - a_lp) / a_lp))
    delay_samples = d
    if shift >= 1:
        refit = fit(d - shift)
        evals += refit[-1]
        if refit[0] <= err:
            err, fb, mix, hc, _ = refit
            delay_samples = d - shift
    time_ms = delay_samples * 1000.0 / fs

    time_out, note = snap_to_tempo(time_ms, tempo_bpm)
    confidence = float(np.clip(1.0 - np.exp(-diag["peak_score"] / 40.0), 0.0, 1.0) * np.exp(-err / 0.02))
    delay = Delay(enabled=mix > 0.5, confidence=round(confidence, 3), time_ms=round(time_out, 3),
                  feedback_pct=round(fb, 2), mix_pct=round(mix, 2), highcut_hz=round(hc, 1), sync_note=note)
    return DelayEstimate(delay, {**diag, "detected": True, "fit_error": round(err, 5), "evaluations": evals})
