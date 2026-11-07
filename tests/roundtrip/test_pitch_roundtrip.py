"""Milestone 7 round trip: a natural synthetic singer as the dry vocal, the same kind of
singer (another performance) through a pitch corrector of known amount and retune speed as
the reference."""

import pytest

from vocalmatch.estimators.pitch import estimate_pitch_correction
from vocalmatch.synth import PitchStyle, synthetic_vocal

FS = 44100
AMOUNT_TOL_PCT = 12.0

CASES = {  # correction, retune_ms -> expected retune class (None: not detected)
    "natural": (0.0, 20.0, None),
    "half, 40 ms": (0.5, 40.0, "slow"),
    "70 %, 30 ms": (0.7, 30.0, "medium"),
    "hard, 10 ms": (1.0, 10.0, "fast"),
    "hard, 60 ms": (1.0, 60.0, "medium"),
    "90 %, 20 ms": (0.9, 20.0, None),  # class not checked: between fast and medium
}


@pytest.mark.parametrize("name", list(CASES))
def test_pitch_correction_roundtrip(name):
    correction, retune_ms, retune_class = CASES[name]
    dry = synthetic_vocal(seed=2, duration_s=20.0, fs=FS, pitch=PitchStyle())
    ref = synthetic_vocal(seed=3, duration_s=20.0, fs=FS, pitch=PitchStyle(correction=correction, retune_ms=retune_ms))
    e = estimate_pitch_correction(ref, dry, FS)
    assert e.amount_pct == pytest.approx(100.0 * correction, abs=AMOUNT_TOL_PCT), e.to_dict()
    assert e.detected == (correction >= 0.3), e.to_dict()
    if retune_class is not None:
        assert e.retune == retune_class, e.to_dict()
