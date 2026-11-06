from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt, sosfreqz

from vocalmatch.chain import render, to_stereo
from vocalmatch.chain.biquad import rbj_sos, sos_response_db
from vocalmatch.chain.dynamics import gain_computer_db, process_compressor, process_deesser
from vocalmatch.chain.effects import (
    process_delay,
    process_doubler,
    process_eq,
    process_reverb,
    process_saturation,
)
from vocalmatch.preset import EQ, Compressor, DeEsser, Delay, Doubler, EQBand, Preset, Reverb, Saturation

FS = 48000


def noise(seconds=1.0, seed=0, fs=FS):
    return np.random.default_rng(seed).standard_normal(int(seconds * fs)) * 0.1


def sine(freq, seconds=1.0, amp=0.5, fs=FS):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * fs)) / fs)


# ------------------------------------------------------------------ biquads
@pytest.mark.parametrize("gain", [-9.0, 4.0, 12.0])
def test_peak_gain_at_centre(gain):
    sos = rbj_sos("peak", 1000, gain, 1.5, FS)
    assert sos_response_db(sos, np.array([1000.0]), FS)[0] == pytest.approx(gain, abs=1e-6)


def test_shelf_asymptotes():
    ls = rbj_sos("low_shelf", 200, 6.0, 0.7071, FS)
    hs = rbj_sos("high_shelf", 5000, -4.0, 0.7071, FS)
    r_ls = sos_response_db(ls, np.array([10.0, 200.0, 10000.0]), FS)
    r_hs = sos_response_db(hs, np.array([50.0, 5000.0, 20000.0]), FS)
    assert r_ls[0] == pytest.approx(6.0, abs=0.05) and abs(r_ls[2]) < 0.05 and r_ls[1] == pytest.approx(3.0, abs=0.05)
    assert abs(r_hs[0]) < 0.05 and r_hs[2] == pytest.approx(-4.0, abs=0.1) and r_hs[1] == pytest.approx(-2.0, abs=0.05)


def test_highpass_minus3db_at_cutoff():
    sos = rbj_sos("highpass", 4000, 0.0, 0.7071, FS)
    assert sos_response_db(sos, np.array([4000.0]), FS)[0] == pytest.approx(-3.01, abs=0.02)


def test_response_matches_scipy():
    sos = np.vstack([rbj_sos("peak", 700, 5, 2, FS), rbj_sos("high_shelf", 8000, -3, 0.7, FS)])
    w, h = sosfreqz(sos, worN=512, fs=FS)
    np.testing.assert_allclose(sos_response_db(sos, w, FS), 20 * np.log10(np.abs(h)), atol=1e-9)


# ------------------------------------------------------------- whole chain
def test_neutral_preset_is_identity():
    x = noise()
    y = render(Preset(), x, FS)
    assert y.shape == (len(x), 2)
    np.testing.assert_array_equal(y[:, 0], x)
    np.testing.assert_array_equal(y[:, 1], x)


def test_enabled_but_neutral_effects_are_identity():
    p = Preset()
    for name in ("input_gain", "eq", "compressor", "saturation", "deesser", "delay", "reverb", "doubler", "output_gain"):
        getattr(p.effects, name).enabled = True
    p.effects.compressor.threshold_db = 0.0  # ratio 1 -> no gain change
    x = noise()
    np.testing.assert_allclose(render(p, x, FS)[:, 0], x, atol=1e-12)


def test_to_stereo_shapes():
    x = noise(0.1)
    assert to_stereo(x).shape == (2, len(x))
    assert to_stereo(np.stack([x, -x], axis=1)).shape == (2, len(x))
    np.testing.assert_array_equal(to_stereo(np.stack([x, -x]))[1], -x)


def test_render_tail_length():
    p = Preset()
    p.effects.reverb = Reverb(enabled=True, mix_pct=30)
    assert render(p, noise(0.5), FS, tail_s=1.0).shape == (int(1.5 * FS), 2)


# ---------------------------------------------------------------------- EQ
def test_eq_matches_response_on_noise():
    bands = [EQBand("low_shelf", 150, -6, 0.7071), EQBand("peak", 2500, 7, 1.0)]
    x = to_stereo(noise(4.0))
    y = process_eq(x, FS, EQ(enabled=True, bands=bands))
    sos = np.vstack([rbj_sos(b.type, b.freq_hz, b.gain_db, b.q, FS) for b in bands])
    # Compare filtered-noise spectrum ratio with the analytic response
    seg = 8192
    X = np.mean(np.abs(np.fft.rfft(x[0, : seg * 20].reshape(20, seg) * np.hanning(seg), axis=1)) ** 2, axis=0)
    Y = np.mean(np.abs(np.fft.rfft(y[0, : seg * 20].reshape(20, seg) * np.hanning(seg), axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(seg, 1 / FS)
    sel = (f > 80) & (f < 16000)
    ratio = 10 * np.log10(Y[sel] / X[sel])
    expected = sos_response_db(sos, f[sel], FS)
    assert np.median(np.abs(ratio - expected)) < 0.5


# --------------------------------------------------------------- compressor
def test_gain_computer_curve():
    lv = np.array([-40.0, -23.0, -20.0, -17.0, 0.0])
    gr = gain_computer_db(lv, -20.0, 4.0, 6.0)
    assert gr[0] == 0.0 and gr[1] == 0.0  # below knee
    assert gr[2] == pytest.approx(0.75 * 3 ** 2 / 12)  # middle of knee
    assert gr[3] == pytest.approx(3 - 3 / 4)  # top of knee joins the ratio line
    assert gr[4] == pytest.approx(20 - 5)


def test_compressor_steady_state_level():
    p = Compressor(enabled=True, threshold_db=-20, ratio=4, attack_ms=1, release_ms=50, knee_db=0, makeup_db=3)
    # square wave so the peak detector sees a constant level of -8 dBFS
    x = 10 ** (-8 / 20) * np.sign(np.sin(2 * np.pi * 100 * np.arange(FS) / FS))
    y = process_compressor(to_stereo(x), FS, p)[0]
    out_db = 20 * np.log10(np.max(np.abs(y[FS // 2 :])))
    assert out_db == pytest.approx(-20 + 12 / 4 + 3, abs=0.05)


def test_compressor_attack_time_constant():
    attack = 10.0
    p = Compressor(enabled=True, threshold_db=-30, ratio=20, attack_ms=attack, release_ms=100, knee_db=0)
    x = np.full(FS // 2, 10 ** (-10 / 20))
    g = 20 * np.log10(process_compressor(to_stereo(x), FS, p)[0] / x)
    target = -(20 - 20 / 20)  # static reduction
    n63 = int(np.argmax(g <= 0.632 * target))
    assert n63 / FS * 1000 == pytest.approx(attack, rel=0.02)


# ----------------------------------------------------- saturation / de-esser
def test_saturation_small_signal_unity_and_bounded():
    p = Saturation(enabled=True, drive_db=12, mix_pct=100)
    small = sine(200, amp=1e-4)
    np.testing.assert_allclose(process_saturation(to_stereo(small), p)[0], small, rtol=1e-6)
    loud = process_saturation(to_stereo(sine(200, amp=1.0)), p)
    assert np.max(np.abs(loud)) <= 10 ** (-12 / 20) + 1e-12


def test_deesser_cuts_sibilance_not_lows():
    p = DeEsser(enabled=True, frequency_hz=6000, threshold_db=-40, reduction_db=10)
    low = sine(300, seconds=0.5, amp=0.3)
    hiss = sine(12000, seconds=0.5, amp=0.3)
    y_low = process_deesser(to_stereo(low), FS, p)[0]
    y_hiss = process_deesser(to_stereo(hiss), FS, p)[0]
    rms = lambda v: np.sqrt(np.mean(v[FS // 10 :] ** 2))  # noqa: E731
    assert 20 * np.log10(rms(y_low) / rms(low)) > -0.5
    assert 20 * np.log10(rms(y_hiss) / rms(hiss)) < -7.0


def test_deesser_below_threshold_is_allpass():
    p = DeEsser(enabled=True, frequency_hz=6000, threshold_db=0, reduction_db=10)
    x = np.zeros(16384)
    x[0] = 0.5  # -6 dBFS impulse stays below a 0 dB threshold
    y = process_deesser(to_stereo(x), FS, p)[0]
    np.testing.assert_allclose(np.abs(np.fft.rfft(y)), 0.5, rtol=1e-4)


# ------------------------------------------------------------ delay / reverb
def test_delay_impulse_response():
    p = Delay(enabled=True, time_ms=100, feedback_pct=50, mix_pct=100, highcut_hz=20000)
    x = np.zeros(FS)
    x[0] = 1.0
    y = process_delay(to_stereo(x), FS, p)[0]
    d = int(0.1 * FS)
    taps = [np.sum(y[k * d - 50 : k * d + 50]) for k in (1, 2, 3)]  # one-pole smears each echo
    np.testing.assert_allclose(taps, [1.0, 0.5, 0.25], rtol=0.02)
    assert np.all(np.abs(y[: d - 1]) < 1e-12)


def test_reverb_rt60():
    decay = 1.2
    p = Reverb(enabled=True, decay_s=decay, predelay_ms=20, mix_pct=100, damping_hz=20000)
    x = np.zeros(4 * FS)
    x[0] = 1.0
    y = process_reverb(to_stereo(x), FS, p)
    # decay_s is the low-frequency RT60 (in-loop damping shortens the highs)
    y = sosfilt(butter(4, 1000, fs=FS, output="sos"), y, axis=-1)
    e = y[0] ** 2 + y[1] ** 2
    schroeder = 10 * np.log10(np.cumsum(e[::-1])[::-1] / np.sum(e))
    t = np.arange(len(e)) / FS
    sel = (schroeder < -5) & (schroeder > -35)
    slope = np.polyfit(t[sel], schroeder[sel], 1)[0]
    assert -60 / slope == pytest.approx(decay, rel=0.1)


def test_reverb_predelay():
    p = Reverb(enabled=True, decay_s=1.0, predelay_ms=20, mix_pct=100)
    x = np.zeros(FS // 2)
    x[0] = 1.0
    y = process_reverb(to_stereo(x), FS, p)
    first = np.argmax(np.any(np.abs(y) > 1e-9, axis=0))
    assert first == int(0.020 * FS) + int(round(min_fdn_ms() * FS / 1000))


def min_fdn_ms():
    from vocalmatch.chain.effects import FDN_DELAYS_MS

    return min(FDN_DELAYS_MS)


def test_reverb_outputs_are_decorrelated():
    p = Reverb(enabled=True, decay_s=2.0, mix_pct=100)
    y = process_reverb(to_stereo(noise(2.0)), FS, p)
    c = np.corrcoef(y[0, FS:], y[1, FS:])[0, 1]
    assert abs(c) < 0.3


# --------------------------------------------------------------- doubler/width
def test_width_zero_is_mono_and_mix_adds_voices():
    x = to_stereo(noise(1.0))
    mono = process_doubler(x, FS, Doubler(enabled=True, mix_pct=50, width_pct=0))
    np.testing.assert_allclose(mono[0], mono[1])
    wide = process_doubler(x, FS, Doubler(enabled=True, mix_pct=50, width_pct=100))
    assert np.corrcoef(wide[0], wide[1])[0, 1] < 0.9


# ------------------------------------------------------------- match amount
def test_match_amount_scales_departure_from_neutral():
    from vocalmatch.chain import apply_match_amount

    p = Preset.load(Path(__file__).resolve().parents[2] / "shared/examples/example_preset.json")
    half = apply_match_amount(p, 0.5)
    assert half.effects.compressor.ratio == pytest.approx(1.0 + (p.effects.compressor.ratio - 1.0) / 2)
    assert half.effects.compressor.threshold_db == p.effects.compressor.threshold_db
    assert [b.gain_db for b in half.effects.eq.bands] == pytest.approx([b.gain_db / 2 for b in p.effects.eq.bands])
    assert half.effects.reverb.mix_pct == pytest.approx(p.effects.reverb.mix_pct / 2)
    assert p.effects.compressor.ratio == 4.0  # the input preset is not modified
    x = noise()
    np.testing.assert_allclose(render(p, x, FS, match_amount=0.0)[:, 0], x, atol=1e-12)
    np.testing.assert_array_equal(render(p, x, FS, match_amount=1.0), render(p, x, FS))
