"""Quick end-to-end check of an installed analyzer: ``vocalmatch selftest``.

Renders a short synthetic vocal through a known EQ + compressor, analyses
it, and checks the estimates against the same tolerances as the round-trip
tests (EQ +-1.5 dB per 1/3-octave band; threshold +-2 dB, ratio +-20 %).
The installer runs it after installing, like an Audio Unit validator.
"""

from __future__ import annotations

import time

import numpy as np

from .chain import render, to_stereo
from .chain.effects import process_eq
from .estimators.eq import eq_response_db
from .features import band_average_db, power_db, third_octave_centres
from .pipeline import analyze
from .preset import EQ, Compressor, EQBand, Preset, validate_preset_dict
from .synth import synthetic_vocal

FS = 44100


def run(verbose: bool = True) -> bool:
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    dry = synthetic_vocal(seed=21, duration_s=8.0, fs=FS)
    truth = Preset()
    truth.effects.eq = EQ(enabled=True, bands=[EQBand("peak", 3000, 5.0, 1.2), EQBand("high_shelf", 10000, 3.0, 0.7071)])
    truth.effects.compressor = Compressor(
        enabled=True, threshold_db=-22, ratio=4.0, attack_ms=5, release_ms=100, knee_db=6, makeup_db=5
    )
    ref = render(truth, dry, FS)
    est = analyze(ref, dry, FS, reference_name="selftest")
    checks: list[tuple[str, bool, str]] = []

    try:
        validate_preset_dict(est.to_dict())
        checks.append(("preset matches the schema", True, ""))
    except Exception as exc:  # noqa: BLE001 - report any validation problem
        checks.append(("preset matches the schema", False, str(exc).splitlines()[0]))

    centres = third_octave_centres(100, 12500)
    t = band_average_db(lambda f: eq_response_db(truth.effects.eq, f, FS), centres)
    e = band_average_db(lambda f: eq_response_db(est.effects.eq, f, FS), centres)
    err = (e - t) - np.median(e - t)
    worst = float(np.max(np.abs(err)))
    checks.append(("EQ recovered", worst <= 1.5, f"worst 1/3-octave band {worst:.2f} dB (tolerance 1.5)"))

    x = to_stereo(dry)
    rms = lambda y: float(power_db(np.mean(y * y)))  # noqa: E731
    shift = rms(process_eq(x, FS, est.effects.eq)) - rms(process_eq(x, FS, truth.effects.eq))
    c, tc = est.effects.compressor, truth.effects.compressor
    thr_err = (c.threshold_db - shift) - tc.threshold_db
    ratio_err = c.ratio / tc.ratio - 1
    checks.append(("compressor threshold", abs(thr_err) <= 2.0, f"{c.threshold_db - shift:.1f} dB vs {tc.threshold_db:.0f} dB"))
    checks.append(("compressor ratio", abs(ratio_err) <= 0.2, f"{c.ratio:.2f}:1 vs {tc.ratio:.0f}:1"))

    ok = all(passed for _, passed, _ in checks)
    for name, passed, detail in checks:
        say(f"  {'PASS' if passed else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    say(f"Self-test {'passed' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
    return ok
