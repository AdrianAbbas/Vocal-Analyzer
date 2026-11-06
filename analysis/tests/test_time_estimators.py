"""Unit tests for the delay, reverb and doubler/width estimators (round trips with
tolerances live in /tests/roundtrip)."""

import numpy as np
import pytest

from vocalmatch.chain import render, to_stereo
from vocalmatch.chain.effects import process_reverb
from vocalmatch.estimators.common import autocorr_at, autocorr_fft, envelope_db
from vocalmatch.estimators.delay import find_delay_time, snap_to_tempo
from vocalmatch.estimators.reverb import drop_features, estimate_reverb
from vocalmatch.estimators.width import estimate_width, signal_features
from vocalmatch.preset import Delay, Doubler, Preset, Reverb
from vocalmatch.synth import synthetic_vocal

FS = 44100


@pytest.fixture(scope="module")
def dry():
    return synthetic_vocal(seed=9, duration_s=8.0, fs=FS)


def _render(dry, **effects):
    p = Preset()
    for k, v in effects.items():
        setattr(p.effects, k, v)
    return render(p, dry, FS)


def test_autocorr_helpers_agree():
    x = np.random.default_rng(0).standard_normal(5000)
    x[1000:] += 0.5 * x[:-1000]
    full = autocorr_fft(x, 1500)
    np.testing.assert_allclose(autocorr_at(x, [1000, 1200]), full[[1000, 1200]], atol=1e-9)
    assert full[1000] > 0.3 and abs(full[1200]) < 0.1


def test_delay_time_found_exactly(dry):
    ref = _render(dry, delay=Delay(True, 1.0, 330.0, 20.0, 15.0, 9000.0))
    d, diag = find_delay_time(ref, dry, FS)
    assert d is not None and abs(d - round(0.33 * FS)) <= 2, diag


def test_no_delay_not_detected(dry):
    assert find_delay_time(to_stereo(dry), dry, FS)[0] is None
    ref = _render(dry, reverb=Reverb(True, 1.0, 1.5, 20.0, 25.0, 6000.0))
    assert find_delay_time(ref, dry, FS)[0] is None  # the reverb alone is not an echo


def test_tempo_snap():
    assert snap_to_tempo(468.0, 96.0) == (pytest.approx(468.75), "1/8D")
    assert snap_to_tempo(300.0, 96.0) == (300.0, None)  # no note value within 2 %
    assert snap_to_tempo(250.0, None) == (250.0, None)


def test_drop_features_see_the_tail(dry):
    lv_dry = envelope_db(to_stereo(dry).mean(axis=0), FS)
    wet = process_reverb(to_stereo(dry), FS, Reverb(True, 1.0, 2.5, 0.0, 40.0, 8000.0))
    lv_wet = envelope_db(wet.mean(axis=0), FS)
    f_dry, f_wet = drop_features(lv_dry), drop_features(lv_wet)
    # The fastest drops are much shallower with a long tail.
    assert np.mean(f_wet) > np.mean(f_dry) + 10.0


def test_no_reverb_estimated_as_none(dry):
    est = estimate_reverb(to_stereo(dry), dry, FS, max_evals=8)
    assert not est.reverb.enabled and est.reverb.mix_pct < 1.0


def test_width_features_and_mono_reference(dry):
    est = estimate_width(to_stereo(dry), dry, FS)
    assert not est.doubler.enabled and "mono" in est.diagnostics["reason"]
    wide = _render(dry, doubler=Doubler(True, 1.0, 50.0, 150.0))
    narrow = _render(dry, doubler=Doubler(True, 1.0, 50.0, 50.0))
    # loud-frame side-to-mid ratio (broadband) grows with the width
    assert signal_features(wide, FS)[0] > signal_features(narrow, FS)[0] + 6.0
