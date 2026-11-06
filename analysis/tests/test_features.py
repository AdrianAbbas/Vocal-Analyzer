import numpy as np
import pytest

from vocalmatch.chain import to_stereo
from vocalmatch.chain.effects import process_eq
from vocalmatch.features import (
    active_mask,
    fractional_octave_matrix,
    frame_rms_db,
    ltas,
    third_octave_centres,
    voicing,
)
from vocalmatch.preset import EQ, EQBand
from vocalmatch.synth import synthetic_vocal

FS = 44100


@pytest.fixture(scope="module")
def vocal():
    return synthetic_vocal(seed=5, duration_s=6.0, fs=FS)


def test_third_octave_centres():
    c = third_octave_centres(100, 10000)
    assert c[0] == pytest.approx(99.2, abs=0.1) and c[-1] == pytest.approx(10079, abs=1)
    np.testing.assert_allclose(np.diff(np.log2(c)), 1 / 3)


def test_smoothing_rows_sum_to_one():
    f = np.fft.rfftfreq(4096, 1 / FS)
    m = fractional_octave_matrix(f, third_octave_centres(50, 16000))
    np.testing.assert_allclose(m.sum(axis=1), 1.0)


def test_active_mask_finds_phrases(vocal):
    lev = frame_rms_db(vocal, FS)
    mask = active_mask(lev)
    assert 0.4 < mask.mean() < 0.95  # gaps between phrases are excluded


def test_ltas_is_level_and_eq_offset_invariant(vocal):
    f, p1 = ltas(vocal, FS)
    _, p2 = ltas(vocal * 0.1, FS)
    np.testing.assert_allclose(p1, p2, rtol=1e-4)
    # a broadband gain hidden in an EQ (two shelves summing to a constant)
    # must not change the normalised LTAS either
    eq = EQ(enabled=True, bands=[EQBand("low_shelf", 1000, 6, 0.7071), EQBand("high_shelf", 1000, 6, 0.7071)])
    y = process_eq(to_stereo(vocal), FS, eq)[0]
    _, p3 = ltas(y, FS)
    sel = (f > 100) & (f < 8000)
    assert np.median(np.abs(10 * np.log10(p3[sel] / p1[sel]))) < 0.5


def test_voicing_separates_tone_and_noise():
    n = FS
    tone = np.sin(2 * np.pi * 180 * np.arange(n) / FS)
    hiss = np.random.default_rng(0).standard_normal(n)
    assert voicing(tone, FS, 2048, 1024).mean() > 0.9
    assert voicing(hiss, FS, 2048, 1024).mean() < 0.1
