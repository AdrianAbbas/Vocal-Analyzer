"""Quick end-to-end check of an installed analyzer: ``vocalmatch selftest``.

Renders a short synthetic vocal through a known EQ + compressor, analyses
it, and checks the estimates against the same tolerances as the round-trip
tests (EQ +-1.5 dB per 1/3-octave band; threshold +-2 dB, ratio +-20 %).
The installer runs it after installing, like an Audio Unit validator.

run_separation() checks source separation the same way: a short synthetic song (the
synthetic vocal over drums, bass and chords) through the real Demucs worker. The vocal stem
must have lost the band's low end (kick and bass) and be closer to the true vocal than the
mix is.
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
    est = analyze(ref, dry, FS, reference_name="selftest", passes=1)
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


def run_separation(verbose: bool = True) -> bool:
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from .separation import SeparationFailed, SeparationUnavailable, load_stem, run_demucs
    from .synth import synthetic_song

    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    mix, vocal, _ = synthetic_song(seed=31, duration_s=12.0, fs=FS)
    with tempfile.TemporaryDirectory() as tmp:
        song = Path(tmp) / "selftest song.wav"
        sf.write(str(song), mix, FS, subtype="FLOAT")
        try:
            v_path, _ = run_demucs(song, Path(tmp) / "stems",
                                   progress=lambda f, st: say(f"  ... {st}") if f >= 1.0 or f == 0.0 else None)
        except (SeparationUnavailable, SeparationFailed) as exc:
            say(f"  FAIL  source separation: {exc}")
            say(f"Separation self-test FAILED ({time.time() - t0:.0f} s)")
            return False
        stem, _ = load_stem(v_path, target_fs=FS)
    stem = stem.mean(axis=1)[: len(vocal)]
    m = mix.mean(axis=1)[: len(stem)]
    v = vocal[: len(stem)]

    def low_db(x):
        from scipy.signal import welch

        f, p = welch(x, fs=FS, nperseg=8192)
        return 10 * np.log10(np.sum(p[(f >= 20) & (f < 100)]) + 1e-30)

    def sdr(est):  # the true vocal against the best-scaled estimate
        g = float(np.dot(est, v) / (np.dot(est, est) + 1e-30))
        return 10 * np.log10(np.sum(v * v) / (np.sum((v - g * est) ** 2) + 1e-30))

    low_drop = low_db(m) - low_db(stem)
    gain = sdr(stem) - sdr(m)
    checks = [
        ("kick and bass removed from the vocal", low_drop >= 10.0, f"{low_drop:.1f} dB less below 100 Hz (needs 10)"),
        ("vocal extracted", gain >= 1.0, f"{sdr(stem):.1f} dB SDR vs {sdr(m):.1f} dB for the mix (needs +1)"),
    ]
    ok = all(passed for _, passed, _ in checks)
    for name, passed, detail in checks:
        say(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    say(f"Separation self-test {'passed' if ok else 'FAILED'} ({time.time() - t0:.0f} s)")
    return ok
