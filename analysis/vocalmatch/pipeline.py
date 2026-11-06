"""End-to-end analysis: reference vocal + dry vocal -> Preset.

Every effect of the chain is estimated (milestone 3), each by its own module. They run as a
coordinate descent over the chain, in two passes: each estimator simulates its effect on the
dry vocal through the *current* estimates of every effect before it in the chain, and (where
it can afford to) scores the result through the current estimates of the effects after it.
The first pass starts from a neutral chain; the second repeats every estimate with the
others known.

Within a pass the order follows what each effect leaves visible in the reference:

1. Time-based effects: the delay from the echo in the autocorrelation, the reverb from the
   level drops after phrases, the doubler and width from the side signal.
2. EQ, against the dry vocal through the time-based effects. Those are linear and
   time-invariant (the doubler nearly so), so they commute with the EQ; the spectral change
   of the saturation and de-esser (from their current estimates) is handed over as a
   correction, so it is not fitted as EQ.
3. Compressor on the EQ'd dry vocal, its statistics taken after the effects that follow it.
4. Saturation, then de-esser. The saturation estimate is recorded but left off: only the
   refinement's whole-chain model selection switches it on (see refine.py).

Then (milestone 4, optional) the refinement optimizer adjusts everything jointly against the
whole-chain render. *reference* is an isolated vocal stem: separation.py extracts it from a
full song first.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .chain import to_stereo
from .chain.dynamics import process_compressor, process_deesser
from .chain.effects import process_delay, process_doubler, process_eq, process_reverb, process_saturation
from .estimators.compressor import estimate_compressor
from .estimators.deesser import estimate_deesser
from .estimators.delay import estimate_delay
from .estimators.eq import estimate_eq
from .estimators.reverb import estimate_reverb
from .estimators.saturation import estimate_saturation
from .estimators.width import estimate_width
from .features import ltas
from .preset import Effects, Preset, Reference

ESTIMATED = ("eq", "compressor", "saturation", "deesser", "delay", "reverb", "doubler")


class _Chain:
    """The current estimates as partial chains (dry -> entering effect k, effect k -> end)."""

    def __init__(self, effects: Effects, fs: float):
        self.e = effects
        self.fs = fs

    def upto(self, x: np.ndarray, stop: str) -> np.ndarray:
        """x through every estimated effect before *stop* (chain order)."""
        e, fs = self.e, self.fs
        steps = [
            ("eq", lambda v: process_eq(v, fs, e.eq)),
            ("compressor", lambda v: process_compressor(v, fs, e.compressor)),
            ("saturation", lambda v: process_saturation(v, e.saturation)),
            ("deesser", lambda v: process_deesser(v, fs, e.deesser)),
            ("delay", lambda v: process_delay(v, fs, e.delay)),
            ("reverb", lambda v: process_reverb(v, fs, e.reverb)),
            ("doubler", lambda v: process_doubler(v, fs, e.doubler)),
        ]
        for name, fn in steps:
            if name == stop:
                break
            if getattr(e, name).enabled:
                x = fn(x)
        return x

    def time_effects(self, after: str | None = None):
        """The linear time-based effects (delay -> reverb -> doubler) as v, fs -> v, from *after*."""
        e = self.e
        names = ["delay", "reverb", "doubler"]
        if after in names:
            names = names[names.index(after) + 1 :]
        fns = {"delay": process_delay, "reverb": process_reverb, "doubler": process_doubler}

        def post(v, fs=None):
            fs = self.fs if fs is None else fs
            for n in names:
                p = getattr(e, n)
                if p.enabled:
                    v = fns[n](v, fs, p)
            return v

        return post


def analyze(
    reference: np.ndarray,
    dry: np.ndarray,
    fs: int,
    reference_name: str = "",
    tempo_bpm: float | None = None,
    passes: int = 2,
    refine_evals: int = 0,
    max_seconds: float | None = 45.0,
) -> Preset:
    """*refine_evals* > 0 runs the refinement optimizer (refine.py) after the estimators,
    with that many full-chain evaluations (about 1 s each for 12 s of audio). Signals longer
    than *max_seconds* are analysed on excerpts of that total length (analysis_excerpt)."""
    ref = to_stereo(reference)
    x = to_stereo(dry)
    full_s = ref.shape[1] / fs
    if max_seconds:
        ref = analysis_excerpt(ref, fs, max_seconds)
        x = analysis_excerpt(x, fs, max_seconds)
    preset = Preset(
        reference=Reference(
            filename=reference_name,
            sample_rate_hz=int(fs),
            tempo_bpm=tempo_bpm,
            duration_s=round(full_s, 3),
        )
    )
    e = preset.effects
    # Gain stages are not estimated: level is matched by the compressor make-up.
    e.input_gain.enabled = True
    e.output_gain.enabled = True
    chain = _Chain(e, fs)
    timings: dict[str, float] = {}
    diag: dict[str, dict] = {}

    def timed(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - t0, 2)
        diag[name] = out.diagnostics
        return out

    for k in range(passes):
        # 1. Time-based effects. (The delay's time and echo shape barely depend on the rest;
        #    the first estimate is kept.)
        if k == 0:
            e.delay = timed("delay", estimate_delay, ref, x, fs, tempo_bpm).delay
        e.reverb = timed("reverb", estimate_reverb, ref, x, fs, pre=chain.upto(x, "reverb")).reverb
        e.doubler = timed("doubler", estimate_width, ref, x, fs, pre=chain.upto(x, "doubler")).doubler
        post = chain.time_effects()

        # 2. EQ, with the saturation / de-esser spectral change as a correction.
        correction = None
        if e.saturation.enabled or e.deesser.enabled:
            z = chain.upto(x, "saturation")
            z_out = _nonlinear(z, e, fs)
            _, p_in = ltas(post(z), fs)
            _, p_out = ltas(post(z_out), fs)
            correction = p_out / np.maximum(p_in, 1e-30)
        e.eq = timed("eq", estimate_eq, ref, post(x), fs, dry_correction=correction).eq

        # 3. Compressor, scored after the effects that follow it.
        has_post = any(getattr(e, n).enabled for n in ("delay", "reverb", "doubler")) or e.saturation.enabled
        comp_kwargs = {}
        if has_post:
            comp_kwargs = {"post": post, "saturation": copy.deepcopy(e.saturation) if e.saturation.enabled else None,
                           "makeup_hint": e.compressor.makeup_db}
        e.compressor = timed("compressor", estimate_compressor, ref, chain.upto(x, "compressor"), fs, **comp_kwargs).compressor

        # 4. Saturation, then de-esser.
        # Saturation is estimated (see diagnostics) but left off: mild saturation and small
        # errors of the compressor estimate look alike to any single-effect fit, so whether
        # it is there is decided against the whole chain by the refinement (model selection).
        sat = timed("saturation", estimate_saturation, ref, chain.upto(x, "saturation"), fs, post).saturation
        diag["saturation"] = {**diag["saturation"], "estimate": {"drive_db": sat.drive_db, "mix_pct": sat.mix_pct,
                                                                  "confidence": sat.confidence, "enabled": sat.enabled}}
        e.saturation.enabled = False
        e.deesser = timed("deesser", estimate_deesser, ref, chain.upto(x, "deesser"), fs, post=post).deesser

    if refine_evals > 0:
        from .refine import refine

        res = refine(preset, ref, x, fs, max_evals=refine_evals)
        preset.effects = res.preset.effects
        diag["refinement"] = res.diagnostics
        timings["refinement"] = round(res.seconds, 2)
    _round_settings(preset)
    preset.diagnostics = {**diag, "passes": passes, "seconds": timings}
    return preset


def _round_settings(preset: Preset) -> None:
    """Readable numbers in preset.json (the plug-in reads them exactly as written)."""
    e = preset.effects
    for name in ("compressor", "saturation", "deesser", "delay", "reverb", "doubler"):
        fx = getattr(e, name)
        for k, v in vars(fx).items():
            if isinstance(v, float) and k != "confidence":
                setattr(fx, k, round(v, 3))
    for b in e.eq.bands:
        b.gain_db = round(b.gain_db, 2)


def _nonlinear(z: np.ndarray, e: Effects, fs: float) -> np.ndarray:
    if e.saturation.enabled:
        z = process_saturation(z, e.saturation)
    if e.deesser.enabled:
        z = process_deesser(z, fs, e.deesser)
    return z


EXCERPT_WINDOW_S = 9.0


def analysis_excerpt(x: np.ndarray, fs: float, max_seconds: float) -> np.ndarray:
    """At most *max_seconds* of a (2, n) signal for analysis: windows of about 9 s spread
    evenly over it, each cut at the quietest 10 ms frame near its ends (so the joins fall in
    pauses, not in the middle of a note or a reverb tail). The estimators use distributions
    of features, which this sample keeps; the analysis time stays bounded for whole songs."""
    n = x.shape[1]
    total = int(max_seconds * fs)
    if n <= total:
        return x
    win = int(EXCERPT_WINDOW_S * fs)
    k = max(1, total // win)
    hop = int(round(0.01 * fs))
    e = np.convolve(np.mean(x * x, axis=0), np.ones(hop) / hop, mode="same")
    search = int(1.5 * fs)

    def quiet(at):
        a, b = max(0, at - search), min(n, at + search)
        return a + int(np.argmin(e[a:b]))

    pieces = []
    for i in range(k):
        centre = int((i + 0.5) * n / k)
        a, b = quiet(max(0, centre - win // 2)), quiet(min(n, centre + win // 2))
        if b > a:
            pieces.append(x[:, a:b])
    return np.concatenate(pieces, axis=1) if pieces else x[:, :total]
