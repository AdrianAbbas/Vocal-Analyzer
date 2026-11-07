"""Synthetic round-trip cases: process a dry vocal with known settings, run
the analyzer on the result, and measure how well it recovers them.

Shared by test_synthetic_roundtrip.py (asserts tolerances) and report.py
(prints a results table).
"""

from __future__ import annotations

import functools
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from vocalmatch.chain import render, to_stereo
from vocalmatch.chain.effects import process_eq
from vocalmatch.estimators.compressor import envelope_stats, estimate_compressor, stats_distance
from vocalmatch.estimators.eq import eq_response_db, estimate_eq
from vocalmatch.features import band_average_db, power_db, third_octave_centres
from vocalmatch.pipeline import analyze
from vocalmatch.preset import EQ, Compressor, EQBand, Preset
from vocalmatch.synth import synthetic_vocal

FS = 44100
DRY_SEED = 1
DRY_SECONDS = 12.0

# ------------------------------------------------------------- tolerances
EQ_BAND_TOL_DB = 1.5  # per 1/3-octave band, after removing the broadband offset
EQ_EVAL_RANGE_HZ = (100.0, 12500.0)
COMP_THRESHOLD_TOL_DB = 2.0
COMP_RATIO_TOL_REL = 0.20
COMP_TIME_TOL_FACTOR = 2.0  # attack / release within x/÷ 2
COMP_MAKEUP_TOL_DB = 1.5
OUTPUT_STATS_TOL_DB = 0.25  # envelope-statistics distance, render(estimate) vs reference


@functools.lru_cache(maxsize=4)
def dry_vocal(seed: int = DRY_SEED, seconds: float = DRY_SECONDS) -> np.ndarray:
    return synthetic_vocal(seed=seed, duration_s=seconds, fs=FS)


def eq_preset(bands: list[EQBand]) -> EQ:
    return EQ(enabled=True, confidence=1.0, bands=bands)


def comp(t, r, a, rel, m, knee=6.0) -> Compressor:
    return Compressor(enabled=True, confidence=1.0, threshold_db=t, ratio=r, attack_ms=a, release_ms=rel, knee_db=knee, makeup_db=m)


def _random_eq(seed: int) -> list[EQBand]:
    rng = np.random.default_rng(seed)
    bands = []
    if rng.random() < 0.7:
        bands.append(EQBand("low_shelf", float(np.exp(rng.uniform(np.log(60), np.log(400)))), float(rng.uniform(-9, 6)), 0.7071))
    for _ in range(int(rng.integers(1, 4))):
        bands.append(
            EQBand("peak", float(np.exp(rng.uniform(np.log(150), np.log(10000)))), float(rng.uniform(-8, 8)),
                   float(np.exp(rng.uniform(np.log(0.5), np.log(3)))))
        )
    if rng.random() < 0.7:
        bands.append(EQBand("high_shelf", float(np.exp(rng.uniform(np.log(3000), np.log(12000)))), float(rng.uniform(-6, 8)), 0.7071))
    return bands


EQ_CASES: dict[str, list[EQBand]] = {
    "presence": [EQBand("low_shelf", 150, -4, 0.7071), EQBand("peak", 3000, 5, 1.2), EQBand("high_shelf", 10000, 3, 0.7071)],
    "scoop+air": [EQBand("peak", 400, -6, 1.5), EQBand("peak", 1200, 2, 2.0), EQBand("peak", 6500, -4, 4.0), EQBand("high_shelf", 12000, 6, 0.7071)],
    "single-peak": [EQBand("peak", 2000, 6, 2.0)],
    "telephone": [EQBand("low_shelf", 300, -10, 0.9), EQBand("peak", 1500, 8, 0.8), EQBand("high_shelf", 4000, -10, 0.9)],
    **{f"random-{s}": _random_eq(s) for s in (11, 12, 13, 14)},
}

COMP_CASES: dict[str, Compressor] = {
    "medium 4:1": comp(-24, 4.0, 5.0, 80.0, 6.0),
    "gentle 2.5:1 slow": comp(-18, 2.5, 15.0, 200.0, 3.0),
    "heavy 8:1 fast": comp(-30, 8.0, 2.0, 50.0, 10.0),
    "slow attack 3:1": comp(-20, 3.0, 30.0, 400.0, 4.0),
}

COMBINED_CASES: dict[str, tuple[list[EQBand], Compressor]] = {
    "pop vocal": (EQ_CASES["presence"], comp(-22, 4.0, 4.0, 90.0, 5.0)),
    "lofi": (EQ_CASES["telephone"], comp(-26, 6.0, 8.0, 150.0, 8.0)),
}


# ---------------------------------------------------------------- results
@dataclass
class EQResult:
    max_err_db: float  # worst 1/3-octave band, offset removed
    offset_db: float  # broadband level difference (not observable; info only)
    band_err_db: np.ndarray
    centres_hz: np.ndarray
    estimated: EQ

    @property
    def ok(self) -> bool:
        return self.max_err_db <= EQ_BAND_TOL_DB


@dataclass
class CompResult:
    true: Compressor
    est: Compressor
    level_shift_db: float = 0.0  # input-level shift caused by EQ estimate (combined cases)
    output_stats_db: float = 0.0
    errors: dict = field(default_factory=dict)

    def __post_init__(self):
        t, e, d = self.true, self.est, self.level_shift_db
        self.errors = {
            "threshold_db": (e.threshold_db - d) - t.threshold_db,
            "ratio_rel": e.ratio / t.ratio - 1.0,
            "attack_factor": e.attack_ms / t.attack_ms,
            "release_factor": e.release_ms / t.release_ms,
            "makeup_db": (e.makeup_db + d) - t.makeup_db,
        }

    @property
    def checks(self) -> dict[str, bool]:
        er = self.errors
        f = COMP_TIME_TOL_FACTOR
        return {
            "threshold": abs(er["threshold_db"]) <= COMP_THRESHOLD_TOL_DB,
            "ratio": abs(er["ratio_rel"]) <= COMP_RATIO_TOL_REL,
            "attack": 1 / f <= er["attack_factor"] <= f,
            "release": 1 / f <= er["release_factor"] <= f,
            "makeup": abs(er["makeup_db"]) <= COMP_MAKEUP_TOL_DB,
            "output_stats": self.output_stats_db <= OUTPUT_STATS_TOL_DB,
        }

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def compare_eq(true: EQ, est: EQ, eval_range_hz: tuple[float, float] = EQ_EVAL_RANGE_HZ) -> EQResult:
    """*eval_range_hz*: the bands compared (a voice has no energy below its lowest note, so
    the EQ there cannot be measured; see duet_cases.eq_range_for)."""
    centres = third_octave_centres(*eval_range_hz)
    t = band_average_db(lambda f: eq_response_db(true, f, FS), centres)
    e = band_average_db(lambda f: eq_response_db(est, f, FS), centres)
    diff = e - t
    offset = float(np.median(diff))
    err = diff - offset
    return EQResult(float(np.max(np.abs(err))), offset, err, centres, est)


def _rms_db(x: np.ndarray) -> float:
    return float(power_db(np.mean(x * x)))


def _output_stats_distance(estimated: Preset, reference: np.ndarray, dry: np.ndarray) -> float:
    out = render(estimated, dry, FS).T
    a, b = envelope_stats(out, FS), envelope_stats(to_stereo(reference), FS)
    return stats_distance(a, b)


# ------------------------------------------------------------------ runners
def run_eq_case(name: str) -> EQResult:
    dry = dry_vocal()
    p = Preset()
    p.effects.eq = eq_preset(EQ_CASES[name])
    ref = render(p, dry, FS)
    return compare_eq(p.effects.eq, estimate_eq(ref, dry, FS).eq)


def run_comp_case(name: str) -> CompResult:
    dry = dry_vocal()
    p = Preset()
    p.effects.compressor = COMP_CASES[name]
    ref = render(p, dry, FS)
    est = estimate_compressor(ref, dry, FS).compressor
    q = Preset()
    q.effects.compressor = est
    return CompResult(COMP_CASES[name], est, 0.0, _output_stats_distance(q, ref, dry))


def run_combined_case(name: str) -> tuple[EQResult, CompResult, Preset]:
    bands, c = COMBINED_CASES[name]
    dry = dry_vocal()
    truth = Preset()
    truth.effects.eq = eq_preset(bands)
    truth.effects.compressor = c
    ref = render(truth, dry, FS)
    est = analyze(ref, dry, FS, reference_name=f"synthetic:{name}")
    eq_res = compare_eq(truth.effects.eq, est.effects.eq)
    # Broadband EQ level is unobservable, so compare compressor threshold and
    # make-up relative to the level actually entering the compressor.
    x = to_stereo(dry)
    shift = _rms_db(process_eq(x, FS, est.effects.eq)) - _rms_db(process_eq(x, FS, truth.effects.eq))
    comp_res = CompResult(c, est.effects.compressor, shift, _output_stats_distance(est, ref, dry))
    return eq_res, comp_res, est


# ======================================================== milestone 3 effects
# Each case renders the dry vocal through a realistic EQ + compressor and the effect, then
# runs that effect's estimator on the true signal entering it, followed by the true effects
# after it (the pipeline's coordinate descent supplies estimates of those instead; the
# full-chain round trip covers that).
from vocalmatch.chain.dynamics import process_compressor  # noqa: E402
from vocalmatch.chain.effects import process_delay, process_doubler, process_reverb, process_saturation  # noqa: E402
from vocalmatch.estimators.deesser import estimate_deesser  # noqa: E402
from vocalmatch.estimators.delay import estimate_delay  # noqa: E402
from vocalmatch.estimators.reverb import estimate_reverb  # noqa: E402
from vocalmatch.estimators.saturation import estimate_saturation  # noqa: E402
from vocalmatch.estimators.width import estimate_width  # noqa: E402
from vocalmatch.preset import DeEsser, Delay, Doubler, Reverb, Saturation  # noqa: E402

BASE_COMP = comp(-22, 4.0, 4.0, 90.0, 5.0)

# tolerances
DELAY_TIME_TOL_MS = 1.0
DELAY_FB_TOL_PCT = 5.0
DELAY_MIX_TOL_PCT = 2.0
DELAY_HIGHCUT_TOL_FACTOR = 1.25
REVERB_DECAY_TOL_REL = 0.20  # SPEC example: reverb decay +-20 %
REVERB_PREDELAY_TOL_MS = 15.0
REVERB_MIX_TOL_PCT = 3.0
REVERB_DAMPING_TOL_FACTOR = 1.5
DOUBLER_MIX_TOL_PCT = 5.0
WIDTH_TOL_PCT = 10.0
SAT_DRIVE_TOL_DB = 3.0
SAT_MIX_TOL_PCT = 10.0
DEESS_FREQ_TOL_REL = 0.15
DEESS_THRESHOLD_TOL_DB = 3.0
DEESS_REDUCTION_TOL_DB = 2.0

M3_CASES: dict[str, dict] = {
    "slapback delay": {"delay": Delay(True, 1.0, 120.0, 0.0, 22.0, 4000.0)},
    "1/8 dotted delay": {"delay": Delay(True, 1.0, 468.75, 35.0, 12.0, 5000.0)},
    "plate reverb": {"reverb": Reverb(True, 1.0, 1.6, 20.0, 20.0, 7000.0)},
    "hall reverb": {"reverb": Reverb(True, 1.0, 3.2, 60.0, 15.0, 5000.0)},
    "room reverb": {"reverb": Reverb(True, 1.0, 0.6, 5.0, 30.0, 9000.0)},
    "delay into reverb": {"delay": Delay(True, 1.0, 375.0, 25.0, 12.0, 6000.0),
                          "reverb": Reverb(True, 1.0, 1.8, 30.0, 18.0, 7000.0)},
    "doubler": {"doubler": Doubler(True, 1.0, 35.0, 100.0)},
    "doubler + width + reverb": {"reverb": Reverb(True, 1.0, 1.5, 20.0, 15.0, 7000.0),
                                 "doubler": Doubler(True, 1.0, 20.0, 140.0)},
    "warm saturation": {"saturation": Saturation(True, 1.0, 9.0, 50.0)},
    "hard saturation": {"saturation": Saturation(True, 1.0, 21.0, 30.0)},
    "de-esser": {"deesser": DeEsser(True, 1.0, 6500.0, -32.0, 5.0)},
    "strong de-esser": {"deesser": DeEsser(True, 1.0, 5000.0, -40.0, 10.0)},
}


@dataclass
class M3Result:
    true: dict
    est: dict
    errors: dict
    checks: dict

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def _truth(name: str) -> Preset:
    p = Preset()
    p.effects.eq = eq_preset(EQ_CASES["presence"])
    p.effects.compressor = BASE_COMP
    for k, v in M3_CASES[name].items():
        setattr(p.effects, k, v)
    return p


def _upto(p: Preset, x: np.ndarray, stop: str) -> np.ndarray:
    from vocalmatch.pipeline import _Chain

    return _Chain(p.effects, FS).upto(x, stop)


def _post(p: Preset, after: str):
    from vocalmatch.pipeline import _Chain

    return _Chain(p.effects, FS).time_effects(after)


def run_m3_case(name: str) -> M3Result:
    dry = to_stereo(dry_vocal())
    truth = _truth(name)
    ref = render(truth, dry, FS)
    fx = M3_CASES[name]
    est: dict = {}
    errors: dict = {}
    checks: dict = {}
    if "delay" in fx:
        t, e = fx["delay"], estimate_delay(ref, dry, FS, pre=_upto(truth, dry, "delay")).delay
        est["delay"] = e
        errors["delay"] = {"time_ms": e.time_ms - t.time_ms, "feedback_pct": e.feedback_pct - t.feedback_pct,
                           "mix_pct": e.mix_pct - t.mix_pct, "highcut_factor": e.highcut_hz / t.highcut_hz}
        er = errors["delay"]
        checks.update({"delay detected": e.enabled, "delay time": abs(er["time_ms"]) <= DELAY_TIME_TOL_MS,
                       "delay feedback": abs(er["feedback_pct"]) <= DELAY_FB_TOL_PCT,
                       "delay mix": abs(er["mix_pct"]) <= DELAY_MIX_TOL_PCT,
                       "delay high-cut": 1 / DELAY_HIGHCUT_TOL_FACTOR <= er["highcut_factor"] <= DELAY_HIGHCUT_TOL_FACTOR})
    if "reverb" in fx:
        t, e = fx["reverb"], estimate_reverb(ref, dry, FS, pre=_upto(truth, dry, "reverb")).reverb
        est["reverb"] = e
        errors["reverb"] = {"decay_rel": e.decay_s / t.decay_s - 1.0, "predelay_ms": e.predelay_ms - t.predelay_ms,
                            "mix_pct": e.mix_pct - t.mix_pct, "damping_factor": e.damping_hz / t.damping_hz}
        er = errors["reverb"]
        checks.update({"reverb detected": e.enabled, "reverb decay": abs(er["decay_rel"]) <= REVERB_DECAY_TOL_REL,
                       "reverb predelay": abs(er["predelay_ms"]) <= REVERB_PREDELAY_TOL_MS,
                       "reverb mix": abs(er["mix_pct"]) <= REVERB_MIX_TOL_PCT,
                       "reverb damping": 1 / REVERB_DAMPING_TOL_FACTOR <= er["damping_factor"] <= REVERB_DAMPING_TOL_FACTOR})
    if "doubler" in fx:
        t, e = fx["doubler"], estimate_width(ref, dry, FS, pre=_upto(truth, dry, "doubler")).doubler
        est["doubler"] = e
        errors["doubler"] = {"mix_pct": e.mix_pct - t.mix_pct, "width_pct": e.width_pct - t.width_pct}
        checks.update({"doubler detected": e.enabled, "doubler mix": abs(errors["doubler"]["mix_pct"]) <= DOUBLER_MIX_TOL_PCT,
                       "width": abs(errors["doubler"]["width_pct"]) <= WIDTH_TOL_PCT})
    if "saturation" in fx:
        t = fx["saturation"]
        e = estimate_saturation(ref, _upto(truth, dry, "saturation"), FS, _post(truth, None)).saturation
        est["saturation"] = e
        errors["saturation"] = {"drive_db": e.drive_db - t.drive_db, "mix_pct": e.mix_pct - t.mix_pct}
        checks.update({"saturation detected": e.enabled,
                       "saturation drive": abs(errors["saturation"]["drive_db"]) <= SAT_DRIVE_TOL_DB,
                       "saturation mix": abs(errors["saturation"]["mix_pct"]) <= SAT_MIX_TOL_PCT})
    if "deesser" in fx:
        t = fx["deesser"]
        e = estimate_deesser(ref, _upto(truth, dry, "deesser"), FS, post=_post(truth, None)).deesser
        est["deesser"] = e
        errors["deesser"] = {"frequency_rel": e.frequency_hz / t.frequency_hz - 1.0,
                             "threshold_db": e.threshold_db - t.threshold_db, "reduction_db": e.reduction_db - t.reduction_db}
        er = errors["deesser"]
        checks.update({"de-esser detected": e.enabled, "de-esser frequency": abs(er["frequency_rel"]) <= DEESS_FREQ_TOL_REL,
                       "de-esser threshold": abs(er["threshold_db"]) <= DEESS_THRESHOLD_TOL_DB,
                       "de-esser reduction": abs(er["reduction_db"]) <= DEESS_REDUCTION_TOL_DB})
    return M3Result(fx, est, errors, checks)


# ======================================================== milestone 4: the whole chain
FULL_CHAIN_REFINE_EVALS = 120


def full_chain_truth() -> Preset:
    p = Preset.load(Path(__file__).resolve().parents[2] / "shared/examples/example_preset.json")
    p.effects.doubler = Doubler(True, 1.0, 20.0, 120.0)
    return p


def run_full_chain(refine_evals: int = FULL_CHAIN_REFINE_EVALS):
    from vocalmatch.refine import Target, distance

    dry = to_stereo(dry_vocal())
    truth = full_chain_truth()
    ref = render(truth, dry, FS)
    est = analyze(ref, dry, FS, reference_name="synthetic:full chain", refine_evals=refine_evals)
    return truth, est, compare_eq(truth.effects.eq, est.effects.eq), est.diagnostics.get("refinement", {})
