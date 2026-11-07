"""Pitch-correction estimation (milestone 7).

How hard is the reference vocal tuned? A pitch corrector pulls every note onto the scale:
what it takes away is the singer's own intonation, which shows in a pitch track of the
vocal (YIN, 5 ms hop):

* note-centre deviation: how far each note's steady pitch lands from the nearest semitone
  (after removing the recording's overall tuning offset): natural singing scatters by
  10-20 cents, a corrected vocal sits on the grid;
* within-note movement: vibrato and drift around the note centre;
* settling: after a note starts, how fast the pitch reaches the note: an exponential fitted
  to the median onset curve, reported as 3 time constants (95 % of the way). A natural scoop
  takes ~150-300 ms, a fast corrector a few tens of ms: the retune speed.

The *amount* compares the reference's deviation and movement with the dry vocal's (your
singing is the baseline for "uncorrected"), so it is an estimate of the correction the
reference has *relative to you*. The key (Krumhansl-Schmuckler on the note histogram) and the
tuning offset are reported too. This is an observation, not an effect of the chain: v1 of
the plug-in does no pitch correction (SPEC non-goals), it tells you what your tuner should do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from ..features import to_mono

FS_TRACK = 11025
HOP_S = 0.005
WIN_S = 0.040
F0_RANGE = (70.0, 900.0)
YIN_THRESHOLD = 0.15
NOTE_SPLIT_CENTS = 100.0
MIN_NOTE_S = 0.12
ONSET_SKIP_S = 0.10
SETTLE_WINDOW_S = 0.25
DETECT_AMOUNT = 0.30
TRACKER_NOISE_CENTS = 2.0  # the tracker's own scatter on a perfectly steady tone
ONSET_FIT_S = 0.15
RETUNE_FAST, RETUNE_MEDIUM = 0.35, 0.8  # reference settling / dry settling

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class PitchTrack:
    times: np.ndarray  # s
    midi: np.ndarray  # fractional MIDI note, NaN when unvoiced


@dataclass
class Note:
    start: float
    end: float
    midi: np.ndarray  # the track over the note
    times: np.ndarray

    @property
    def centre(self) -> float:
        steady = self.midi[self.times - self.start >= ONSET_SKIP_S]
        return float(np.median(steady if len(steady) else self.midi))


@dataclass
class PitchStats:
    tuning_offset_cents: float
    centre_dev_cents: float  # RMS distance of note centres from the grid
    movement_cents: float  # RMS movement around the note centre (steady part)
    settle_ms: float | None  # 3 x the time constant of the median onset approach (95 % of the way)
    key: str | None
    key_score: float
    notes: int


@dataclass
class PitchCorrectionEstimate:
    detected: bool
    confidence: float
    amount_pct: float
    retune_ms: float | None
    retune: str | None  # "fast" / "medium" / "slow", relative to the dry vocal's own settling
    key: str | None
    tuning_offset_cents: float
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"detected": self.detected, "confidence": round(self.confidence, 3), "amount_pct": round(self.amount_pct, 1),
                "retune_ms": None if self.retune_ms is None else round(self.retune_ms, 1), "retune": self.retune, "key": self.key,
                "tuning_offset_cents": round(self.tuning_offset_cents, 1)}


# ------------------------------------------------------------------------- tracker
def pitch_track(x: np.ndarray, fs: float) -> PitchTrack:
    """YIN (de Cheveigne & Kawahara 2002) on the mono signal at 11.025 kHz, vectorised."""
    mono = to_mono(x)
    y = resample_poly(mono, FS_TRACK, int(fs)) if int(fs) != FS_TRACK else mono
    hop = int(HOP_S * FS_TRACK)
    w = int(WIN_S * FS_TRACK)
    tau_max = int(FS_TRACK / F0_RANGE[0])
    tau_min = int(FS_TRACK / F0_RANGE[1])
    n_frames = (len(y) - w - tau_max) // hop
    if n_frames <= 0:
        return PitchTrack(np.zeros(0), np.zeros(0))
    starts = np.arange(n_frames) * hop
    idx = starts[:, None] + np.arange(w + tau_max)[None, :]
    seg = y[idx]
    level = np.sqrt(np.mean(seg[:, :w] ** 2, axis=1))
    nfft = 1 << int(np.ceil(np.log2(2 * (w + tau_max))))
    a = np.fft.rfft(seg[:, :w], nfft, axis=1)
    b = np.fft.rfft(seg, nfft, axis=1)
    r = np.fft.irfft(np.conj(a) * b, nfft, axis=1)[:, : tau_max + 1]  # sum_j x[j] x[j + tau], j < w
    c = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(seg**2, axis=1)], axis=1)
    e0 = c[:, w][:, None]
    taus = np.arange(tau_max + 1)
    e_tau = c[:, taus + w] - c[:, taus]
    d = e0 + e_tau - 2.0 * r
    d[:, 0] = 0.0
    cum = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tau_max + 1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] / np.maximum(cum, 1e-12)
    midi = np.full(n_frames, np.nan)
    active = level > np.percentile(level, 95) * 10 ** (-30 / 20)
    for i in np.nonzero(active)[0]:
        row = dn[i]
        below = np.nonzero(row[tau_min:] < YIN_THRESHOLD)[0]
        if len(below) == 0:
            continue
        t = tau_min + below[0]
        while t + 1 <= tau_max and row[t + 1] < row[t]:
            t += 1
        if 1 <= t < tau_max:  # parabolic interpolation
            den = row[t - 1] - 2 * row[t] + row[t + 1]
            t = t + 0.5 * (row[t - 1] - row[t + 1]) / den if den > 0 else t
        f0 = FS_TRACK / t
        if F0_RANGE[0] <= f0 <= F0_RANGE[1]:
            midi[i] = 69.0 + 12.0 * np.log2(f0 / 440.0)
    times = (starts + w / 2) / FS_TRACK
    return PitchTrack(times, midi)


def notes_of(track: PitchTrack) -> list[Note]:
    notes, start = [], None
    m = track.midi
    for i in range(len(m) + 1):
        voiced = i < len(m) and np.isfinite(m[i])
        jump = voiced and start is not None and abs(m[i] - m[i - 1]) * 100.0 > NOTE_SPLIT_CENTS
        if start is not None and (not voiced or jump):
            if track.times[i - 1] - track.times[start] >= MIN_NOTE_S:
                notes.append(Note(track.times[start], track.times[i - 1], m[start:i], track.times[start:i]))
            start = None
        if voiced and start is None:
            start = i
    return notes


# ------------------------------------------------------------------------ features
def _circular_offset(centres: np.ndarray) -> float:
    ang = 2 * np.pi * (centres % 1.0)
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * 100.0)


def _key(notes: list[Note], offset_cents: float) -> tuple[str | None, float]:
    hist = np.zeros(12)
    for n in notes:
        hist[int(round(n.centre - offset_cents / 100.0)) % 12] += n.end - n.start
    if hist.sum() <= 0:
        return None, 0.0
    best = (None, -2.0)
    for tonic in range(12):
        for name, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(hist, np.roll(prof, tonic))[0, 1])
            if r > best[1]:
                best = (f"{PITCH_CLASSES[tonic]} {name}", r)
    return best


def pitch_stats(x: np.ndarray, fs: float) -> PitchStats:
    notes = notes_of(pitch_track(x, fs))
    if len(notes) < 4:
        return PitchStats(0.0, float("nan"), float("nan"), None, None, 0.0, len(notes))
    centres = np.array([n.centre for n in notes])
    offset = _circular_offset(centres)
    shifted = centres - offset / 100.0
    centre_dev = float(np.sqrt(np.mean((100.0 * (shifted - np.round(shifted))) ** 2)))
    moves, onsets = [], []
    k_fit = int(round(ONSET_FIT_S / HOP_S))
    for n, c in zip(notes, centres):
        rel = n.times - n.start
        steady = rel >= ONSET_SKIP_S
        if steady.sum() >= 5:
            moves.append(100.0 * (n.midi[steady] - c))
        if n.end - n.start >= SETTLE_WINDOW_S and len(n.midi) > k_fit:
            onsets.append(100.0 * (n.midi[:k_fit] - c))

    def denoise(v: float) -> float:  # remove the tracker's own scatter (in quadrature)
        return float(np.sqrt(max(v * v - TRACKER_NOISE_CENTS**2, 0.0)))

    movement = denoise(float(np.sqrt(np.mean(np.concatenate(moves) ** 2)))) if moves else float("nan")
    centre_dev = denoise(centre_dev)
    settle = _settle_ms(np.median(np.array(onsets), axis=0)) if len(onsets) >= 3 else None
    key, score = _key(notes, offset)
    return PitchStats(offset, centre_dev, movement, settle, key, score, len(notes))


def _settle_ms(curve_cents: np.ndarray) -> float:
    """Fit |c(t)| = A exp(-t / tau) to the median onset curve where it stands above the
    tracker's noise; 3 tau (95 % of the way). The first frames of a note sit half a window
    after its start, so the fit, not the first crossing, measures fast correctors."""
    mag = np.abs(curve_cents)
    t = np.arange(len(mag)) * HOP_S * 1000.0
    sel = mag > 3.0 * TRACKER_NOISE_CENTS
    # only the decaying part: up to the first frame back in the noise
    if not sel[0]:
        return 0.0  # already on the note when the tracker first sees it
    end = int(np.argmin(sel)) if not sel.all() else len(sel)
    if end < 3:
        return 3.0 * HOP_S * 1000.0 * max(end, 1) / 3.0
    slope = np.polyfit(t[:end], np.log(mag[:end]), 1)[0]
    tau = -1.0 / slope if slope < 0 else float("inf")
    return float(min(3.0 * tau, 600.0))


# ------------------------------------------------------------------------ estimator
def estimate_pitch_correction(reference: np.ndarray, dry: np.ndarray, fs: float) -> PitchCorrectionEstimate:
    ref = pitch_stats(reference, fs)
    base = pitch_stats(dry, fs)
    diag = {"reference": ref.__dict__, "dry": base.__dict__}
    if ref.notes < 4 or base.notes < 4 or not np.isfinite(ref.movement_cents) or not np.isfinite(base.movement_cents):
        return PitchCorrectionEstimate(False, 0.0, 0.0, None, None, ref.key, ref.tuning_offset_cents, {**diag, "reason": "too few notes"})
    # How much of the dry vocal's intonation (note landing + movement) the reference keeps.
    keep_dev = ref.centre_dev_cents / max(base.centre_dev_cents, 1.0)
    keep_move = ref.movement_cents / max(base.movement_cents, 1.0)
    amount = float(np.clip(1.0 - (0.5 * keep_dev + 0.5 * keep_move), 0.0, 1.0))
    detected = amount >= DETECT_AMOUNT
    retune_ms, retune = None, None
    if detected and ref.settle_ms is not None:
        # The absolute time is approximate (the tracker's 40 ms window blurs onsets: up to 2x);
        # its ratio to the dry vocal's own settling is the more reliable class.
        retune_ms = float(np.clip(ref.settle_ms, 0.0, 600.0))
        ratio = retune_ms / base.settle_ms if base.settle_ms else 0.0
        retune = "fast" if ratio < RETUNE_FAST else ("medium" if ratio < RETUNE_MEDIUM else "slow")
    # Confidence: agreement of the two measures, discounted for few notes.
    agreement = 1.0 - min(1.0, abs(keep_dev - keep_move))
    confidence = float(np.clip(agreement * min(1.0, min(ref.notes, base.notes) / 20.0), 0.0, 1.0))
    return PitchCorrectionEstimate(detected, confidence, 100.0 * amount, retune_ms, retune, ref.key, ref.tuning_offset_cents,
                                   {**diag, "keep_centre_deviation": round(keep_dev, 3), "keep_movement": round(keep_move, 3)})
