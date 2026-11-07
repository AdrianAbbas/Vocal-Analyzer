"""Pitch tracker and pitch-correction estimator (milestone 7)."""

import numpy as np
import pytest

from vocalmatch.estimators.pitch import estimate_pitch_correction, notes_of, pitch_stats, pitch_track
from vocalmatch.synth import PitchStyle, synthetic_vocal

FS = 44100


def _tone(midi_notes, dur=0.4, fs=FS, cents=0.0):
    out = []
    for m in midi_notes:
        f = 440.0 * 2 ** ((m + cents / 100.0 - 69) / 12)
        t = np.arange(int(dur * fs)) / fs
        x = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 8))
        out += [x * np.hanning(len(t)) ** 0.1, np.zeros(int(0.1 * fs))]
    return np.concatenate(out) * 0.2


def test_tracker_is_accurate_on_steady_tones():
    notes = [57, 60, 64, 67, 69, 72]
    tr = pitch_track(_tone(notes), FS)
    found = notes_of(tr)
    assert len(found) == len(notes)
    for n, m in zip(found, notes):
        assert n.centre == pytest.approx(m, abs=0.03)  # 3 cents


def test_tuning_offset_and_key():
    # A major scale, 30 cents sharp
    scale = [57, 59, 61, 62, 64, 66, 68, 69, 64, 61, 57, 69, 66, 62]
    st = pitch_stats(_tone(scale, cents=30.0), FS)
    assert st.tuning_offset_cents == pytest.approx(30.0, abs=3.0)
    assert st.key in ("A major", "F# minor")
    assert st.centre_dev_cents < 3.0


def test_natural_vs_corrected():
    dry = synthetic_vocal(seed=2, duration_s=12.0, fs=FS, pitch=PitchStyle())
    same = synthetic_vocal(seed=3, duration_s=12.0, fs=FS, pitch=PitchStyle())
    hard = synthetic_vocal(seed=3, duration_s=12.0, fs=FS, pitch=PitchStyle(correction=1.0, retune_ms=10.0))
    a = estimate_pitch_correction(same, dry, FS)
    b = estimate_pitch_correction(hard, dry, FS)
    assert not a.detected and a.amount_pct < 20
    assert b.detected and b.amount_pct > 80 and b.retune == "fast"
    assert set(b.to_dict()) == {"detected", "confidence", "amount_pct", "retune_ms", "retune", "key", "tuning_offset_cents"}


def test_too_little_singing_is_not_guessed():
    e = estimate_pitch_correction(np.zeros(FS), np.zeros(FS), FS)
    assert not e.detected and e.confidence == 0.0
