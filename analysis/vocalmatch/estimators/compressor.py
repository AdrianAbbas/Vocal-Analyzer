"""Compressor estimator.

Works on *unpaired* signals: the reference vocal and the user's dry vocal are
different performances, so nothing here relies on sample alignment. Only the
statistics of each signal's level envelope are compared.

1. Envelope statistics (quantiles over active frames) of each signal:
   10 ms peak level, 100 ms RMS level, crest factor (10 ms peak - 10 ms RMS),
   local contrast (10 ms RMS - 100 ms RMS) and 20 ms level changes. The
   last three are level-independent, which is what pins down attack/release.
2. Static initial guess: quantile-quantile map of the dry vs reference peak
   level distributions. A compressor is (nearly) monotone in level, so the
   q-q curve approximates its static curve; fit threshold/ratio/make-up to it.
3. Refinement: simulate the real chain compressor on the dry signal
   (detector decimated for speed) and minimise the distance between the
   simulated and reference statistics with Nelder-Mead over threshold,
   ratio, attack and release, starting from the best point of a coarse
   attack x release grid. Make-up is solved in closed form at every step
   (it only shifts the level quantiles). A short full-rate polish follows.

The threshold found this way is relative to the dry signal as it enters the
compressor, i.e. it transfers directly to the user's own vocal level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize

from ..chain import to_stereo
from ..chain.dynamics import compressor_gain_db, gain_computer_db
from ..features import power_db
from ..preset import Compressor

QUANTILES = np.linspace(0.10, 0.99, 24)
QQ_QUANTILES = np.linspace(0.20, 0.995, 40)
LEVEL_FEATURES = ("peak", "rms_long")
SHAPE_FEATURES = ("crest", "contrast", "delta")
FEATURE_WEIGHTS = {"peak": 1.0, "rms_long": 1.0, "crest": 2.0, "contrast": 2.0, "delta": 1.0}

MIN_CONFIDENCE = 0.05

BOUNDS = {
    "threshold_db": (-60.0, 0.0),
    "ratio": (1.0, 20.0),
    "attack_ms": (0.1, 200.0),
    "release_ms": (5.0, 2000.0),
    "makeup_db": (-24.0, 36.0),
}


@dataclass
class CompressorEstimate:
    compressor: Compressor
    distance: float
    distance_uncompressed: float
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------------ features
def envelope_stats(x: np.ndarray, fs: float, gain_db: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Quantiles of envelope features. *x* is (2, n) or mono; an optional
    per-sample *gain_db* is applied first (used by the simulator)."""
    mono = np.max(np.abs(to_stereo(x)), axis=0) if np.ndim(x) == 2 else np.abs(np.asarray(x, float))
    if gain_db is not None:
        mono = mono * 10.0 ** (gain_db / 20.0)
    block = max(1, int(round(0.005 * fs)))
    nb = len(mono) // block
    blocks = mono[: nb * block].reshape(nb, block)
    e = np.mean(blocks * blocks, axis=1)
    pk = np.max(blocks, axis=1)
    e_short = 0.5 * (e[1:] + e[:-1])
    p_short = np.maximum(pk[1:], pk[:-1])
    kernel = np.ones(20) / 20.0
    e_long = np.convolve(e, kernel, mode="same")[1:]

    l_short = power_db(e_short)
    l_peak = 20.0 * np.log10(np.maximum(p_short, 1e-6))
    l_long = power_db(e_long)
    active = (l_long > np.percentile(l_long, 95) - 40.0) & (l_long > -80.0)
    lag = 4
    delta = l_short[lag:] - l_short[:-lag]
    active_d = active[lag:] & active[:-lag]

    def q(v, m):
        v = v[m]
        return np.quantile(v, QUANTILES) if len(v) else np.zeros(len(QUANTILES))

    return {
        "peak": q(l_peak, active),
        "rms_long": q(l_long, active),
        "crest": q(l_peak - l_short, active),
        "contrast": q(l_short - l_long, active),
        "delta": q(delta, active_d),
        "_peak_qq": np.quantile(l_peak[active], QQ_QUANTILES) if active.any() else np.zeros(len(QQ_QUANTILES)),
    }


def stats_distance(a: dict, b: dict, makeup_db: float = 0.0) -> float:
    """Weighted RMS distance (dB) between two stats dicts; *makeup_db* is
    added to the level features of *a*."""
    total = 0.0
    wsum = 0.0
    for k, w in FEATURE_WEIGHTS.items():
        shift = makeup_db if k in LEVEL_FEATURES else 0.0
        total += w * float(np.mean((a[k] + shift - b[k]) ** 2))
        wsum += w
    return float(np.sqrt(total / wsum))


def optimal_makeup(sim: dict, ref: dict) -> float:
    return float(np.mean(np.concatenate([ref[k] - sim[k] for k in LEVEL_FEATURES])))


# ------------------------------------------------------------- static guess
def static_fit(dry_stats: dict, ref_stats: dict, knee_db: float) -> tuple[float, float, float]:
    lin = dry_stats["_peak_qq"]
    lout = ref_stats["_peak_qq"]

    def resid(th):
        t, log_r, m = th
        return lin + m - gain_computer_db(lin, t, 2.0**log_r, knee_db) - lout

    x0 = [float(np.quantile(lin, 0.6)), 1.0, float(np.median(lout - lin))]
    lo = [BOUNDS["threshold_db"][0], 0.0, BOUNDS["makeup_db"][0]]
    hi = [BOUNDS["threshold_db"][1], np.log2(BOUNDS["ratio"][1]), BOUNDS["makeup_db"][1]]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    sol = least_squares(resid, x0, bounds=(lo, hi))
    t, log_r, m = sol.x
    return float(t), float(2.0**log_r), float(m)


# ------------------------------------------------------------------ search
class _Simulator:
    def __init__(self, dry: np.ndarray, fs: float, decimation: int, knee_db: float):
        self.x = to_stereo(dry)
        self.fs = fs
        self.dec = decimation
        self.knee = knee_db
        det = np.max(np.abs(self.x), axis=0)
        nb = len(det) // decimation
        self.n = nb * decimation
        self.det = det[: self.n].reshape(nb, decimation).max(axis=1)
        self.x = self.x[:, : self.n]
        self.cache: dict[tuple, tuple[dict, float]] = {}

    def params(self, t, r, a, rel, m=0.0) -> Compressor:
        return Compressor(enabled=True, threshold_db=t, ratio=r, attack_ms=a, release_ms=rel, knee_db=self.knee, makeup_db=m)

    def stats(self, t, r, a, rel) -> dict:
        key = (round(t, 4), round(r, 4), round(a, 4), round(rel, 4))
        if key not in self.cache:
            g = compressor_gain_db(self.det, self.fs / self.dec, self.params(t, r, a, rel))
            self.cache[key] = envelope_stats(self.x, self.fs, np.repeat(g, self.dec))
        return self.cache[key]

    def score(self, t, r, a, rel, ref) -> tuple[float, float]:
        s = self.stats(t, r, a, rel)
        m = optimal_makeup(s, ref)
        return stats_distance(s, ref, m), m


class _PostSimulator(_Simulator):
    """With effects after the compressor (milestone 3): the reference's statistics include
    the reverb's and delay's tails, so the simulation must too. To keep each evaluation
    cheap, the whole path runs at fs / decimation: the gain comes from the block-max
    detector (as in _Simulator), is applied to the signal resampled to fs / decimation, then
    the post effects run at that rate and the statistics are taken there (the reference's
    as well, see estimate_compressor). *post(v, fs)* must be linear in v, so make-up still
    only shifts the level statistics; a saturation after the compressor (level dependent)
    is applied with the make-up *makeup_hint* folded in."""

    def __init__(self, dry, fs, decimation, knee_db, post, saturation=None, makeup_hint=0.0):
        super().__init__(dry, fs, decimation, knee_db)
        from scipy.signal import resample_poly

        nb = len(self.det)
        self.xd = resample_poly(self.x, 1, decimation, axis=1)[:, :nb]
        self.fs_d = fs / decimation
        self.post = post
        self.saturation = saturation
        self.makeup_hint = makeup_hint

    def stats(self, t, r, a, rel) -> dict:
        key = (round(t, 4), round(r, 4), round(a, 4), round(rel, 4))
        if key not in self.cache:
            g = compressor_gain_db(self.det, self.fs_d, self.params(t, r, a, rel))
            y = self.xd * 10.0 ** (g / 20.0)
            if self.saturation is not None and self.saturation.enabled:
                from ..chain.effects import process_saturation

                k = 10.0 ** (self.makeup_hint / 20.0)
                y = process_saturation(y * k, self.saturation) / k
            self.cache[key] = envelope_stats(self.post(y, self.fs_d), self.fs_d)
        return self.cache[key]


def _to_params(z):
    t = float(np.clip(z[0], *BOUNDS["threshold_db"]))
    r = float(np.clip(2.0 ** z[1], *BOUNDS["ratio"]))
    a = float(np.clip(2.0 ** z[2], *BOUNDS["attack_ms"]))
    rel = float(np.clip(2.0 ** z[3], *BOUNDS["release_ms"]))
    return t, r, a, rel


def estimate_compressor(
    reference: np.ndarray,
    dry: np.ndarray,
    fs: float,
    knee_db: float = 6.0,
    decimation: int = 4,
    max_evals: int = 300,
    max_starts: int = 3,
    good_enough_db: float = 0.06,
    polish_evals: int = 60,
    post=None,
    saturation=None,
    makeup_hint: float = 0.0,
) -> CompressorEstimate:
    """*post(v, fs)*: optional linear function, the (estimated) effects after the compressor
    (de-esser excluded: it barely moves the envelope); *saturation*: an estimated saturation
    between them. With *post* the search runs at fs / decimation throughout."""
    if post is not None:
        from scipy.signal import resample_poly

        sim = _PostSimulator(dry, fs, decimation, knee_db, post, saturation, makeup_hint)
        ref = envelope_stats(resample_poly(to_stereo(reference), 1, decimation, axis=1), sim.fs_d)
        dry_stats = sim.stats(0.0, 1.0, 10.0, 100.0)  # ratio 1: no compression
        polish_evals = 0
    else:
        ref = envelope_stats(to_stereo(reference), fs)
        sim = _Simulator(dry, fs, decimation, knee_db)
        dry_stats = envelope_stats(sim.x, fs)

    m0 = optimal_makeup(dry_stats, ref)
    d_none = stats_distance(dry_stats, ref, m0)

    t0, r0, _ = static_fit(dry_stats, ref, knee_db)

    grid = []
    for a in (1.0, 3.0, 10.0, 30.0):
        for rel in (30.0, 100.0, 300.0, 1000.0):
            grid.append((sim.score(t0, r0, a, rel, ref)[0], a, rel))
    grid.sort()

    def objective(z):
        return sim.score(*_to_params(z), ref)[0]

    res = None
    n_evals = 0
    # Nelder-Mead from the best grid point; if the match is still poor, try
    # the next grid points too (the attack/release landscape has local minima).
    for _, a0, rel0 in grid[:max_starts]:
        z0 = np.array([t0, np.log2(r0), np.log2(a0), np.log2(rel0)])
        steps = np.diag([3.0, 0.5, 1.0, 1.0])
        r_ = minimize(
            objective,
            z0,
            method="Nelder-Mead",
            options={"initial_simplex": np.vstack([z0, z0 + steps]), "maxfev": max_evals, "xatol": 0.02, "fatol": 1e-4},
        )
        n_evals += r_.nfev
        if res is None or r_.fun < res.fun:
            res = r_
        if res.fun < good_enough_db:
            break
    z_best = res.x
    if polish_evals > 0 and decimation > 1:
        # Short full-rate polish: removes the small bias of the decimated detector.
        full = _Simulator(dry, fs, 1, knee_db)
        steps = np.diag([1.0, 0.15, 0.3, 0.3])
        r_ = minimize(
            lambda z: full.score(*_to_params(z), ref)[0],
            z_best,
            method="Nelder-Mead",
            options={"initial_simplex": np.vstack([z_best, z_best + steps]), "maxfev": polish_evals, "xatol": 0.01, "fatol": 1e-5},
        )
        n_evals += r_.nfev
        z_best, sim = r_.x, full
    t, r, a, rel = _to_params(z_best)
    dist, m = sim.score(t, r, a, rel, ref)
    m = float(np.clip(m, *BOUNDS["makeup_db"]))

    comp = Compressor(
        enabled=True,
        threshold_db=round(t, 2),
        ratio=round(r, 3),
        attack_ms=round(a, 3),
        release_ms=round(rel, 2),
        knee_db=knee_db,
        makeup_db=round(m, 2),
    )
    # Confidence: how much of the statistical gap the compressor explains,
    # discounted by how well the final fit matches.
    explained = 1.0 - dist / max(d_none, 1e-9)
    comp.confidence = round(float(np.clip(explained, 0.0, 1.0) * np.exp(-dist / 2.0)), 3)
    if comp.confidence < MIN_CONFIDENCE:
        # The fit explains (almost) nothing: the two performances simply differ in dynamics.
        # No compression, only the level match (make-up of the uncompressed statistics).
        comp.ratio, comp.makeup_db = 1.0, round(float(np.clip(m0, *BOUNDS["makeup_db"])), 2)
    if comp.ratio < 1.05:
        comp.enabled = comp.confidence >= MIN_CONFIDENCE or abs(comp.makeup_db) >= 0.1
        comp.ratio = 1.0
    return CompressorEstimate(
        compressor=comp,
        distance=dist,
        distance_uncompressed=d_none,
        diagnostics={
            "static_fit": {"threshold_db": round(t0, 2), "ratio": round(r0, 3)},
            "stats_distance_db": round(dist, 3),
            "stats_distance_uncompressed_db": round(d_none, 3),
            "evaluations": int(n_evals),
        },
    )
