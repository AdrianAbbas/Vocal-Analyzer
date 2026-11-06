"""Unit tests for the EQ estimator (the full round trip lives in /tests/roundtrip)."""

import numpy as np
import pytest

from vocalmatch.chain import render
from vocalmatch.estimators.eq import GAIN_LIMIT_DB, eq_response_db, estimate_eq
from vocalmatch.features import band_average_db, third_octave_centres
from vocalmatch.preset import EQ, EQBand, Preset
from vocalmatch.synth import synthetic_vocal

FS = 44100


@pytest.fixture(scope="module")
def dry():
    return synthetic_vocal(seed=7, duration_s=8.0, fs=FS)


def _apply(bands, x):
    p = Preset()
    p.effects.eq = EQ(enabled=True, bands=bands)
    return render(p, x, FS)


def test_identity_gives_flat_eq(dry):
    est = estimate_eq(dry, dry, FS)
    c = third_octave_centres(100, 12500)
    resp = band_average_db(lambda f: eq_response_db(est.eq, f, FS), c)
    assert np.max(np.abs(resp)) < 0.3
    assert est.residual_db < 0.05


def test_single_peak_parameters_recovered(dry):
    est = estimate_eq(_apply([EQBand("peak", 3000, 6.0, 2.0)], dry), dry, FS)
    on = [b for b in est.eq.bands if b.enabled]
    assert len(on) == 1 and on[0].type == "peak"
    assert abs(np.log2(on[0].freq_hz / 3000)) < 1 / 12
    assert on[0].gain_db == pytest.approx(6.0, abs=0.75)
    assert on[0].q == pytest.approx(2.0, rel=0.25)


def test_layout_and_gain_limit(dry):
    est = estimate_eq(_apply([EQBand("peak", 1000, 18.0, 1.0)], dry), dry, FS)  # beyond the ±12 dB range
    assert [b.type for b in est.eq.bands] == ["low_shelf", "peak", "peak", "peak", "peak", "high_shelf"]
    assert all(abs(b.gain_db) <= GAIN_LIMIT_DB for b in est.eq.bands)
    peaks = [b.freq_hz for b in est.eq.bands if b.type == "peak"]
    assert peaks == sorted(peaks)


def test_estimate_is_independent_of_reference_level(dry):
    ref = _apply([EQBand("high_shelf", 6000, 5.0, 0.7071)], dry)
    a = estimate_eq(ref, dry, FS).eq
    b = estimate_eq(ref * 0.05, dry, FS).eq
    c = third_octave_centres(100, 12500)
    ra = band_average_db(lambda f: eq_response_db(a, f, FS), c)
    rb = band_average_db(lambda f: eq_response_db(b, f, FS), c)
    assert np.max(np.abs(ra - rb)) < 0.1
