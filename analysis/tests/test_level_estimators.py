"""Unit tests for the saturation and de-esser estimators (round trips with tolerances live
in /tests/roundtrip)."""

import numpy as np
import pytest

from vocalmatch.chain import to_stereo
from vocalmatch.chain.dynamics import process_deesser
from vocalmatch.chain.effects import process_reverb, process_saturation
from vocalmatch.estimators import deesser as D
from vocalmatch.estimators import saturation as S
from vocalmatch.preset import DeEsser, Reverb, Saturation
from vocalmatch.synth import synthetic_vocal

FS = 44100


@pytest.fixture(scope="module")
def loud():
    """A synthetic vocal around -3 dBFS peak (saturation acts on loud signals)."""
    return to_stereo(synthetic_vocal(seed=10, duration_s=6.0, fs=FS))


def test_saturation_lowers_crest_of_loud_frames_only(loud):
    a = S.signal_features(loud, FS)
    b = S.signal_features(process_saturation(loud * 2.0, Saturation(True, 1.0, 12.0, 100.0)) / 2.0, FS)
    crest_loud, crest_quiet = 0, 2 * (len(S.GROUPS) - 1)  # first and last crest features
    assert b[crest_loud] < a[crest_loud] - 1.0
    assert abs(b[crest_quiet] - a[crest_quiet]) < abs(b[crest_loud] - a[crest_loud])


def test_saturation_features_ignore_eq_like_tilt(loud):
    from scipy.signal import butter, sosfilt

    tilt = sosfilt(butter(1, 6000, btype="high", fs=FS, output="sos"), loud, axis=-1)
    a = S.signal_features(loud, FS)
    b = S.signal_features(loud + 0.5 * tilt, FS)
    tilt_feats = slice(1, None, 2)
    assert np.max(np.abs(a[tilt_feats] - b[tilt_feats])) < 1.0


def test_saturation_none_and_found(loud):
    est = S.estimate_saturation(loud, loud, FS)
    assert not est.saturation.enabled
    ref = process_saturation(loud, Saturation(True, 1.0, 15.0, 60.0))
    est = S.estimate_saturation(ref, loud, FS)
    assert est.saturation.drive_db == pytest.approx(15.0, abs=1.0)
    assert est.saturation.mix_pct == pytest.approx(60.0, abs=5.0)


def test_deesser_features_are_eq_invariant(loud):
    a = D.signal_features(loud, FS)
    b = D.signal_features(loud * 2.0, FS)  # broadband gain: shapes unchanged
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_deesser_found_and_none(loud):
    truth = DeEsser(True, 1.0, 6000.0, -38.0, 8.0)
    ref = process_deesser(loud, FS, truth)
    est = D.estimate_deesser(ref, loud, FS, polish_evals=20).deesser
    assert est.enabled
    assert est.frequency_hz == pytest.approx(6000.0, rel=0.15)
    assert est.threshold_db == pytest.approx(-38.0, abs=3.0)
    assert est.reduction_db == pytest.approx(8.0, abs=2.0)
    assert not D.estimate_deesser(loud, loud, FS, polish_evals=10).deesser.enabled


def test_deesser_post_kernel_accounts_for_the_tail(loud):
    rv = Reverb(True, 1.0, 1.8, 20.0, 25.0, 7000.0)
    post = lambda v: process_reverb(v, FS, rv)  # noqa: E731
    ref = post(loud)  # reverb only, no de-esser
    est = D.estimate_deesser(ref, loud, FS, polish_evals=10, post=post).deesser
    assert not est.enabled
