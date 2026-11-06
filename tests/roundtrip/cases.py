"""Synthetic round-trip cases: process a dry vocal with known settings, run
the analyzer on the result, and measure how well it recovers them.

Shared by test_synthetic_roundtrip.py (asserts tolerances) and report.py
(prints a results table).
"""

from __future__ import annotations

import functools
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


def compare_eq(true: EQ, est: EQ) -> EQResult:
    centres = third_octave_centres(*EQ_EVAL_RANGE_HZ)
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
