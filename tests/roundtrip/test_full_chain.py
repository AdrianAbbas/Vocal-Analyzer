"""Milestone 4: the whole chain at once (all eight effects) through analyze() with the
refinement optimizer. Tolerances are looser than the single-effect round trips: the
estimators see every effect at the same time."""

import pytest

from cases import run_full_chain

pytestmark = pytest.mark.slow


def test_full_chain_roundtrip():
    truth, est, eq_res, refinement = run_full_chain()
    t, e = truth.effects, est.effects
    report = {k: getattr(e, k) for k in ("compressor", "saturation", "deesser", "delay", "reverb", "doubler")}
    checks = {
        "refinement lowered the distance by 30 %": refinement["distance_after"] <= 0.7 * refinement["distance_before"],
        "EQ within 2.5 dB per band": eq_res.max_err_db <= 2.5,
        "compressor threshold +-2 dB": abs(e.compressor.threshold_db - t.compressor.threshold_db) <= 2.0,
        "compressor make-up +-1.5 dB": abs(e.compressor.makeup_db - t.compressor.makeup_db) <= 1.5,
        "compressor ratio +-30 %": abs(e.compressor.ratio / t.compressor.ratio - 1) <= 0.30,
        "delay time +-1 ms": e.delay.enabled and abs(e.delay.time_ms - t.delay.time_ms) <= 1.0,
        "delay mix +-3": abs(e.delay.mix_pct - t.delay.mix_pct) <= 3.0,
        "reverb decay +-25 %": e.reverb.enabled and abs(e.reverb.decay_s / t.reverb.decay_s - 1) <= 0.25,
        "reverb mix +-4": abs(e.reverb.mix_pct - t.reverb.mix_pct) <= 4.0,
        "doubler mix +-5": e.doubler.enabled and abs(e.doubler.mix_pct - t.doubler.mix_pct) <= 5.0,
        "width +-15": abs(e.doubler.width_pct - t.doubler.width_pct) <= 15.0,
        "de-esser found": e.deesser.enabled,
    }
    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"failed {failed}; EQ {eq_res.max_err_db:.2f} dB; {report}; {refinement}"
