"""Refinement optimizer (milestone 4).

Starting from the estimates, adjust the chain's settings so that the dry vocal *rendered
through the whole chain* matches the reference as a whole. This fixes what the estimators,
each fitting one effect with the others held at their current estimates, cannot: the
interactions (a compressor's attack against a saturation, an EQ against a de-esser, ...).

The distance is alignment-free (the reference and the dry vocal are different
performances), the sum of:

* multi-resolution spectral statistics: for STFTs of 256, 1024 and 4096 samples, the mean
  and the spread over active frames of the 1/3-octave log power (mid signal), in dB;
* loudness statistics: the compressor estimator's envelope statistics distance (dB);
* every estimator's own features (reverb level drops, side/mid, de-esser and saturation
  distributions), each as an RMS difference in dB.

Search: block-coordinate Nelder-Mead, one effect's settings at a time (its group of 2-6
numbers, in scaled coordinates), cycling through the enabled effects until the evaluation
budget is spent. Before that, saturation and de-esser that the estimators left off (their
detection gate is conservative) are estimated without the gate and switched on if the whole
chain then matches better by 3 % (and 0.1). The result is kept only if it lowers the distance.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .chain import render, to_stereo
from .estimators import deesser as _dee
from .estimators import reverb as _rev
from .estimators import saturation as _sat
from .estimators import width as _wid
from .estimators.compressor import envelope_stats, stats_distance
from .features import EPS, fractional_octave_matrix, power_db, third_octave_centres
from .preset import Preset

FFT_SIZES = (256, 1024, 4096)
BAND_RANGE_HZ = (100.0, 16000.0)
ACTIVE_REL_DB = 35.0
WEIGHTS = {"spectrum": 1.0, "loudness": 1.0, "reverb": 0.5, "width": 0.5, "deesser": 0.5, "saturation": 1.0}


# --------------------------------------------------------------------- features
def spectral_stats(x: np.ndarray, fs: float) -> list[np.ndarray]:
    st = to_stereo(x)
    mid = 0.5 * (st[0] + st[1])
    out = []
    for n in FFT_SIZES:
        hop = n // 4
        k = (len(mid) - n) // hop + 1
        if k < 4:
            out.append(np.zeros(2))
            continue
        idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
        frames = mid[idx] * np.hanning(n)[None, :]
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        centres = third_octave_centres(*BAND_RANGE_HZ)
        bands = power_db(p @ fractional_octave_matrix(freqs, centres, 3.0).T + EPS)
        total = power_db(p.sum(axis=1))
        active = total > np.percentile(total, 95) - ACTIVE_REL_DB
        b = bands[active] if active.any() else bands
        out.append(np.concatenate([b.mean(axis=0), b.std(axis=0)]))
    return out


@dataclass
class Target:
    """The reference's features, computed once."""

    fs: float
    spectrum: list[np.ndarray]
    loudness: dict
    reverb: np.ndarray
    width: np.ndarray | None
    deesser: np.ndarray
    saturation: np.ndarray

    @classmethod
    def of(cls, reference: np.ndarray, fs: float) -> "Target":
        st = to_stereo(reference)
        side = 0.5 * (st[0] - st[1])
        stereo = float(np.mean(side**2)) > 1e-5 * float(np.mean(st**2) + EPS)
        return cls(fs, spectral_stats(st, fs), envelope_stats(st, fs), _rev.signal_features(0.5 * (st[0] + st[1]), fs),
                   _wid.signal_features(st, fs) if stereo else None, _dee.signal_features(st, fs), _sat.signal_features(st, fs))


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else 0.0


def distance_terms(y: np.ndarray, target: Target) -> dict[str, float]:
    fs = target.fs
    st = to_stereo(y)
    spec = spectral_stats(st, fs)
    terms = {
        "spectrum": float(np.mean([_rms(a, b) for a, b in zip(spec, target.spectrum)])),
        "loudness": stats_distance(envelope_stats(st, fs), target.loudness),
        "reverb": _rms(_rev.signal_features(0.5 * (st[0] + st[1]), fs), target.reverb),
        "deesser": _rms(_dee.signal_features(st, fs), target.deesser),
        "saturation": _rms(_sat.signal_features(st, fs), target.saturation),
    }
    if target.width is not None:
        terms["width"] = _rms(_wid.signal_features(st, fs), target.width)
    return terms


def distance(y: np.ndarray, target: Target) -> float:
    return float(sum(WEIGHTS[k] * v for k, v in distance_terms(y, target).items()))


# ------------------------------------------------------------- parameter groups
# (effect, attribute, transform, step, lower, upper): "log" works on log2 of the value.
GROUPS: dict[str, list[tuple[str, str, float, float, float]]] = {
    "compressor": [("threshold_db", "lin", 2.0, -60.0, 0.0), ("ratio", "log", 0.25, 1.0, 20.0),
                   ("attack_ms", "log", 0.6, 0.1, 200.0), ("release_ms", "log", 0.6, 5.0, 2000.0),
                   ("makeup_db", "lin", 1.0, -24.0, 36.0)],
    "saturation": [("drive_db", "lin", 3.0, 0.0, 36.0), ("mix_pct", "lin", 8.0, 0.0, 100.0)],
    "deesser": [("threshold_db", "lin", 3.0, -60.0, 0.0), ("reduction_db", "lin", 2.0, 0.0, 24.0)],
    "delay": [("feedback_pct", "lin", 5.0, 0.0, 95.0), ("mix_pct", "lin", 2.0, 0.0, 100.0)],
    "reverb": [("decay_s", "log", 0.3, 0.1, 10.0), ("mix_pct", "lin", 3.0, 0.0, 100.0),
               ("damping_hz", "log", 0.5, 500.0, 20000.0), ("predelay_ms", "lin", 10.0, 0.0, 250.0)],
    "doubler": [("mix_pct", "lin", 5.0, 0.0, 100.0), ("width_pct", "lin", 10.0, 0.0, 200.0)],
}
EQ_GAIN_STEP_DB = 1.0
ORDER = ("compressor", "eq", "saturation", "deesser", "reverb", "delay", "doubler")


def _get(p: Preset, effect: str) -> np.ndarray:
    e = p.effects
    if effect == "eq":
        return np.array([b.gain_db / EQ_GAIN_STEP_DB for b in e.eq.bands if b.enabled])
    fx = getattr(e, effect)
    out = []
    for attr, kind, step, lo, hi in GROUPS[effect]:
        v = getattr(fx, attr)
        out.append((math.log2(max(v, 1e-9)) if kind == "log" else v) / step)
    return np.array(out)


def _set(p: Preset, effect: str, z: np.ndarray) -> None:
    e = p.effects
    if effect == "eq":
        for b, g in zip([b for b in e.eq.bands if b.enabled], z):
            b.gain_db = float(np.clip(g * EQ_GAIN_STEP_DB, -12.0, 12.0))
        return
    fx = getattr(e, effect)
    for (attr, kind, step, lo, hi), v in zip(GROUPS[effect], z):
        v = v * step
        setattr(fx, attr, float(np.clip(2.0**v if kind == "log" else v, lo, hi)))


@dataclass
class RefineResult:
    preset: Preset
    distance_before: float
    distance_after: float
    evaluations: int
    terms_before: dict = field(default_factory=dict)
    terms_after: dict = field(default_factory=dict)
    seconds: float = 0.0
    added: list = field(default_factory=list)

    @property
    def diagnostics(self) -> dict:
        return {"distance_before": round(self.distance_before, 3), "distance_after": round(self.distance_after, 3),
                "switched_on": self.added,
                "evaluations": self.evaluations, "seconds": round(self.seconds, 1),
                "terms_before": {k: round(v, 3) for k, v in self.terms_before.items()},
                "terms_after": {k: round(v, 3) for k, v in self.terms_after.items()}}


def refine(preset: Preset, reference: np.ndarray, dry: np.ndarray, fs: float, max_evals: int = 60,
           target: Target | None = None, on_eval=None) -> RefineResult:
    t0 = time.perf_counter()
    target = target or Target.of(reference, fs)
    x = to_stereo(dry)
    evals = 0

    def score(p: Preset) -> float:
        nonlocal evals
        evals += 1
        if on_eval is not None:
            on_eval(min(evals, max_evals))
        return distance(render(p, x, fs).T, target)

    best = copy.deepcopy(preset)
    d0 = score(best)
    d_best = d0
    # Model selection: the estimators switch saturation and de-esser on only for a clear fit
    # (a weak one is as likely to explain estimate errors). Here the whole chain decides: each
    # one that is off is estimated without that gate and kept if the joint distance drops.
    added = []
    for name in ("saturation", "deesser"):
        if getattr(best.effects, name).enabled or evals >= max_evals:
            continue
        candidate = copy.deepcopy(best)
        fx = _ungated_estimate(name, candidate, reference, x, fs)
        if fx is None or not fx.enabled:
            continue
        setattr(candidate.effects, name, fx)
        d = score(candidate)
        if d < d_best * (1.0 - MIN_GAIN_TO_ADD) and d < d_best - MIN_ABS_GAIN_TO_ADD:
            best, d_best = candidate, d
            added.append(name)
    groups = [g for g in ORDER if getattr(best.effects, g).enabled and (g != "eq" or any(b.enabled for b in best.effects.eq.bands))]
    while groups and evals < max_evals:
        improved = False
        for g in groups:
            budget = min(max_evals - evals, 4 * (len(_get(best, g)) + 1))
            if budget < len(_get(best, g)) + 2:
                break
            z0 = _get(best, g)
            trial = copy.deepcopy(best)

            def f(z, g=g, trial=trial):
                _set(trial, g, z)
                return score(trial)

            res = minimize(f, z0, method="Nelder-Mead",
                           options={"initial_simplex": np.vstack([z0, z0 + np.eye(len(z0))]), "maxfev": budget,
                                    "xatol": 0.05, "fatol": 1e-3})
            if res.fun < d_best - 1e-4:
                _set(best, g, res.x)
                d_best = float(res.fun)
                improved = True
        if not improved:
            break
    terms_after = distance_terms(render(best, x, fs).T, target)
    terms_before = distance_terms(render(preset, x, fs).T, target)
    if d_best >= d0:
        best, d_best, terms_after = copy.deepcopy(preset), d0, terms_before
    out = RefineResult(best, d0, d_best, evals, terms_before, terms_after, time.perf_counter() - t0)
    out.added = added if d_best < d0 else []
    return out


MIN_GAIN_TO_ADD = 0.03  # an effect switched on by the refinement must lower the distance by 3 %
MIN_ABS_GAIN_TO_ADD = 0.1  # ... and by 0.1 (a near-perfect match has nothing left to explain)


def _ungated_estimate(name: str, p: Preset, reference: np.ndarray, x: np.ndarray, fs: float):
    from .pipeline import _Chain

    chain = _Chain(p.effects, fs)
    post = chain.time_effects()
    if name == "saturation":
        return _sat.estimate_saturation(reference, chain.upto(x, "saturation"), fs, post, min_confidence=0.0).saturation
    return _dee.estimate_deesser(reference, chain.upto(x, "deesser"), fs, post=post, gate=False).deesser
