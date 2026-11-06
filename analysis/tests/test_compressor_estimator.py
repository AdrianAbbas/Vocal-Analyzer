"""Unit tests for the compressor estimator (round trips live in /tests/roundtrip)."""

import numpy as np
import pytest

from vocalmatch.chain import render, to_stereo
from vocalmatch.chain.dynamics import compressor_gain_db, gain_computer_db
from vocalmatch.estimators.compressor import (
    LEVEL_FEATURES,
    SHAPE_FEATURES,
    envelope_stats,
    estimate_compressor,
    static_fit,
)
from vocalmatch.preset import Compressor, Preset
from vocalmatch.synth import synthetic_vocal

FS = 44100


@pytest.fixture(scope="module")
def dry():
    return synthetic_vocal(seed=8, duration_s=8.0, fs=FS)


def test_stats_level_features_shift_shape_features_do_not(dry):
    a = envelope_stats(to_stereo(dry), FS)
    b = envelope_stats(to_stereo(dry * 10 ** (-6 / 20)), FS)
    for k in LEVEL_FEATURES:
        np.testing.assert_allclose(b[k], a[k] - 6.0, atol=1e-6)
    for k in SHAPE_FEATURES:
        np.testing.assert_allclose(b[k], a[k], atol=1e-6)


def test_static_fit_recovers_curve_from_quantiles():
    lin = np.linspace(-40, -3, 40)
    dry_stats = {"_peak_qq": lin}
    ref_stats = {"_peak_qq": lin + 5 - gain_computer_db(lin, -22, 3.0, 6.0)}
    t, r, m = static_fit(dry_stats, ref_stats, 6.0)
    assert t == pytest.approx(-22, abs=0.05) and r == pytest.approx(3.0, rel=0.01) and m == pytest.approx(5, abs=0.05)


def test_no_compression_is_detected(dry):
    est = estimate_compressor(dry, dry, FS)
    c = est.compressor
    gr = -compressor_gain_db(np.abs(dry), FS, Compressor(**{**c.__dict__, "makeup_db": 0.0}))
    assert np.percentile(gr, 99) < 0.5  # essentially no gain reduction
    assert abs(c.makeup_db) < 0.5


def test_level_only_change_goes_to_makeup(dry):
    est = estimate_compressor(dry * 10 ** (4 / 20), dry, FS).compressor
    assert est.makeup_db == pytest.approx(4.0, abs=0.5)


def test_estimate_matches_reference_statistics(dry):
    p = Preset()
    p.effects.compressor = Compressor(enabled=True, threshold_db=-22, ratio=4, attack_ms=5, release_ms=100, knee_db=6, makeup_db=6)
    ref = render(p, dry, FS)
    est = estimate_compressor(ref, dry, FS)
    assert est.distance < 0.1 < est.distance_uncompressed
    assert est.compressor.confidence > 0.8
