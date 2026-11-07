"""Deterministic synthetic "vocal" for round-trip tests.

It is not meant to sound like a singer, only to share the properties the
estimators care about: phrases separated by gaps, harmonic notes with vibrato
and vowel formants, note-to-note dynamics of ~15 dB, breath noise, and
sibilant bursts, so the long-term spectrum covers ~80 Hz - 16 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

# Formant centre (Hz), bandwidth (Hz), relative level (dB): rough sung-vowel values.
VOWELS = {
    "a": ((800, 1150, 2900, 3900, 4950), (80, 90, 120, 130, 140), (0, -6, -32, -20, -50)),
    "e": ((400, 1600, 2700, 3300, 4950), (60, 80, 120, 150, 200), (0, -24, -30, -35, -60)),
    "i": ((270, 2140, 2950, 3900, 4950), (60, 90, 100, 120, 120), (0, -12, -26, -26, -44)),
    "o": ((450, 800, 2830, 3800, 4950), (70, 80, 100, 130, 135), (0, -11, -22, -22, -50)),
    "u": ((325, 700, 2530, 3500, 4950), (50, 60, 170, 180, 200), (0, -16, -35, -40, -60)),
}

# A minor-ish scale, MIDI notes 45 (110 Hz) .. 67 (392 Hz)
SCALE_MIDI = (45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67)


def _midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _spectral_envelope(freqs: np.ndarray, vowel: str) -> np.ndarray:
    centres, bws, levels = VOWELS[vowel]
    env = np.zeros_like(freqs)
    for fc, bw, lv in zip(centres, bws, levels):
        env += 10 ** (lv / 20) / (1.0 + ((freqs - fc) / (bw / 2.0)) ** 2)
    # broadband "brilliance" floor so harmonics extend to high frequencies
    env += 10 ** (-38 / 20) / (1.0 + freqs / 4000.0)
    return env


@dataclass(frozen=True)
class PitchStyle:
    """An intonation model for pitch-correction tests (milestone 7). A natural singer lands
    each note off pitch by a random detune (std *detune_cents*) and scoops into it from below
    (*scoop_cents*, settling with time constant *scoop_ms*); vibrato and drift add to that.
    *correction* (0..1) is a pitch corrector's amount: it scales the detune, vibrato and drift
    by (1 - correction) and pulls the scoop in with time constant retune_ms / 3."""

    detune_cents: float = 15.0
    scoop_cents: float = 80.0
    scoop_ms: float = 70.0
    correction: float = 0.0
    retune_ms: float = 20.0


@dataclass(frozen=True)
class Voice:
    """A singer's voice for multi-singer tests: *pitch_semitones* moves the whole range (a
    female voice sits about +7..+12 above the default), *formant_scale* scales the vowel
    formants (a shorter vocal tract: > 1), *vibrato_scale* the vibrato rate, *breath* the level
    of the breath noise. Voice() is the default synthetic singer."""

    pitch_semitones: float = 0.0
    formant_scale: float = 1.0
    vibrato_scale: float = 1.0
    breath: float = 1.0


def _note(rng, fs, dur, f0, vowel, level_db, pitch=None, prng=None, voice=None):
    n = int(dur * fs)
    t = np.arange(n) / fs
    # vibrato fades in after 150 ms
    vib_depth = 0.35 * np.clip((t - 0.15) / 0.2, 0, 1)
    vib_rate = rng.uniform(4.8, 6.2) * (voice.vibrato_scale if voice is not None else 1.0)
    vib = vib_depth * np.sin(2 * np.pi * vib_rate * t + rng.uniform(0, 2 * np.pi))
    drift = 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 2 * np.pi))
    offset = vib + drift  # semitones
    if pitch is not None:
        keep = 1.0 - pitch.correction
        detune = prng.normal(0.0, pitch.detune_cents) / 100.0
        scoop = prng.uniform(0.5, 1.5) * pitch.scoop_cents / 100.0
        tau = keep * pitch.scoop_ms + pitch.correction * pitch.retune_ms / 3.0
        offset = keep * (vib + drift + detune) - scoop * np.exp(-t / (max(tau, 0.1) / 1000.0))
    f_inst = f0 * 2.0 ** (offset / 12.0)
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    k_max = int(0.47 * fs / (f0 * 1.03))
    ks = np.arange(1, k_max + 1)
    amps = _spectral_envelope(ks * f0 / (voice.formant_scale if voice is not None else 1.0), vowel) / ks**0.3
    jitter = rng.uniform(0, 2 * np.pi, size=k_max)
    sig = np.zeros(n)
    for k, a, ph in zip(ks, amps, jitter):
        sig += a * np.sin(k * phase + ph)

    # breath / aspiration noise, shaped by the same vowel envelope (coarsely)
    noise = rng.standard_normal(n)
    noise = sosfilt(butter(2, [300, 9000], btype="band", fs=fs, output="sos"), noise)
    sig += 0.02 * (voice.breath if voice is not None else 1.0) * np.max(np.abs(sig)) * noise / (np.std(noise) + 1e-12)

    # amplitude envelope: attack, gentle swell/decay, release
    att = rng.uniform(0.02, 0.06)
    rel = rng.uniform(0.06, 0.14)
    env = np.ones(n)
    na, nr = int(att * fs), int(rel * fs)
    env[:na] = 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    env[n - nr :] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    swell = rng.uniform(-6, 6)  # dB across the note
    env *= 10 ** (swell * (t / dur - 0.5) / 20)
    sig *= env / (np.max(np.abs(sig)) + 1e-12)
    return sig * 10 ** (level_db / 20)


def _sibilant(rng, fs, dur, level_db):
    n = int(dur * fs)
    noise = rng.standard_normal(n)
    noise = sosfilt(butter(4, [4500, min(11000, 0.45 * fs)], btype="band", fs=fs, output="sos"), noise)
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    s = noise * env
    return s / (np.max(np.abs(s)) + 1e-12) * 10 ** (level_db / 20)


def synthetic_vocal(
    seed: int = 0, duration_s: float = 14.0, fs: int = 44100, peak_dbfs: float = -3.0,
    pitch: PitchStyle | None = None, voice: Voice | None = None,
) -> np.ndarray:
    """Return a mono float64 vocal-like signal of ~duration_s seconds. With *pitch*, the
    intonation follows that model (its random draws come from their own generator, so the
    same seed gives the same notes with or without it). *voice* changes the singer (see
    Voice); None is the default singer, bit-identical to earlier versions."""
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 10_000) if pitch is not None else None
    total = int(duration_s * fs)
    out = np.zeros(total + fs)
    pos = int(0.3 * fs)
    vowels = list(VOWELS)
    while pos < total - fs:
        n_notes = int(rng.integers(4, 8))
        phrase_level = rng.uniform(-6, 0)
        for _ in range(n_notes):
            dur = rng.uniform(0.18, 0.65)
            f0 = _midi_to_hz(SCALE_MIDI[int(rng.integers(len(SCALE_MIDI)))] + (voice.pitch_semitones if voice is not None else 0.0))
            level = phrase_level + rng.uniform(-14, 0)
            if rng.random() < 0.3:  # consonant "s" before the vowel
                s = _sibilant(rng, fs, rng.uniform(0.07, 0.14), level + rng.uniform(-14, -6))
                out[pos : pos + len(s)] += s
                pos += int(0.8 * len(s))
            note = _note(rng, fs, dur, f0, vowels[int(rng.integers(len(vowels)))], level, pitch, prng, voice)
            out[pos : pos + len(note)] += note
            pos += int(len(note) * rng.uniform(0.85, 1.0))  # slight legato overlap
            if pos >= total:
                break
        pos += int(rng.uniform(0.4, 1.0) * fs)  # gap between phrases
    out = out[:total]
    out += 10 ** (-80 / 20) * rng.standard_normal(total)  # noise floor
    return out / np.max(np.abs(out)) * 10 ** (peak_dbfs / 20)
