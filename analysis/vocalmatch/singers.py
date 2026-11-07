"""Singer separation: who sings when, in a vocal stem with more than one singer.

On a duet the vocal stem holds every singer, so analysing all of it averages their
processing. This module cuts the stem into phrase chunks, describes each chunk by a singer
embedding, clusters the chunks into Singer A, B, C..., and finds the sections where
two singers sing at once (those are excluded: neither singer's processing can be measured
there).

Everything here is this project's own DSP code on numpy / scipy (BSD-3-Clause): no
pretrained model, nothing downloaded, nothing gated. See docs/LICENSES.md.

Chunks: the stem is cut into phrases at pauses (>= 250 ms), long phrases into ~2.5 s chunks.
The embedding of a chunk (over its active frames):
- 12 mel-cepstral coefficients and 12 cepstra of the peak envelope (independent of the
  harmonic spacing), as means: the vocal tract's formants and the voice's (and its
  processing's) spectral balance;
- the 10th, 50th and 90th percentile of log f0 (a normalised-autocorrelation pitch track with
  sub-harmonic checks): the singer's range;
- spectral flatness (breathiness) and the spread of the level (dynamics, compression).
Each feature is scaled by its noise *within* phrases (half against half of the same phrase:
one phrase is one singer), so what changes with every vowel or note counts little and what
stays put within a singer counts. The 3 main directions of that space are clustered (Ward).
A split into singers is accepted when the cluster centres are clearly apart (distance and
distance over spread) and the chunks take turns (a runs test on the time order: singers sing
for several phrases at a time; a split of one voice by vowel or register flips at random).
Chunks nearer than UNCERTAIN_DISTANCE_RATIO to another singer stay unassigned.

Overlap: per frame, the strongest pitch is found and its harmonics are removed from the
whitened spectrum; if a second harmonic series is still strong, two voices sound at once.
A second pitch equal to the lead's previous note is a tail (reverb, delay, legato), unless it
returns as the strongest pitch. Windows (0.75 s) where many voiced frames have two pitches are
overlap, and a phrase that is largely overlap is excluded whole (conservative).

Confidence (0..1) combines the silhouette, how many frames were assigned, and the balance of
cluster sizes. Below UNSURE_CONFIDENCE the result says so and suggests region selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.fft import dct
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .features import frames, power_db, to_mono

FRAME_S = 0.05  # the frame grid of the result (activity lane, labels)
WIN_S = 0.75
HOP_S = 0.25
MAX_SINGERS = 4
N_COMPONENTS = 3
MIN_CENTRE_DISTANCE = 9.0  # cluster centres at least this far apart (chunk-noise units) ...
MIN_CENTRE_RATIO = 3.8  # ... and this many within-cluster spreads
UNCERTAIN_DISTANCE_RATIO = 0.8  # a chunk nearer than this to another singer stays unassigned
CHUNK_S = 2.5
MIN_CHUNK_S = 0.6
MIN_CLUSTER_SHARE = 0.06  # a singer must have at least this share of the singing ...
MIN_CLUSTER_S = 4.0  # ... and this much (a third or fourth singer: twice that)
UNSURE_CONFIDENCE = 0.5
OVERLAP_FRAME_SHARE = 0.35  # windows where this share of voiced frames has two pitches
OVERLAP_PHRASE_SHARE = 0.25  # a phrase this much overlap is excluded whole
TAIL_MEMORY_S = 1.5
TWO_PITCH_RATIO = 0.5
LETTERS = "ABCDEFGH"


@dataclass
class SingerInfo:
    id: str
    seconds: float  # assigned singing (overlap excluded)
    preview: tuple[float, float]  # a short, representative excerpt
    share: float  # of all assigned singing
    median_f0_hz: float

    def to_dict(self) -> dict:
        return {"id": self.id, "seconds": round(self.seconds, 2), "share": round(self.share, 3),
                "preview": [round(self.preview[0], 2), round(self.preview[1], 2)],
                "median_f0_hz": round(self.median_f0_hz, 1)}


@dataclass
class SingerMap:
    """Who sings when, on a FRAME_S grid over the whole stem."""

    duration_s: float
    labels: np.ndarray  # per frame: singer index, -1 = nobody / unassigned, -2 = overlap
    activity: np.ndarray  # per frame: vocal level 0..1 (relative to the stem's loud level)
    singers: list[SingerInfo] = field(default_factory=list)
    confidence: float = 1.0
    overlap_s: float = 0.0
    unassigned_s: float = 0.0
    active_s: float = 0.0
    silhouette: float = 0.0
    separation: float = 0.0
    runs_z: float = 0.0
    message: str = ""

    @property
    def unsure(self) -> bool:
        return self.confidence < UNSURE_CONFIDENCE

    def index(self, singer_id: str) -> int:
        for i, s in enumerate(self.singers):
            if s.id == singer_id.strip().upper():
                return i
        raise ValueError(f"no singer {singer_id!r} (found: {', '.join(s.id for s in self.singers) or 'none'})")

    def segments(self) -> list[tuple[float, float, str]]:
        """Contiguous runs: (start_s, end_s, 'A' | 'B' | ... | 'overlap')."""
        out = []
        for a, b, v in _runs(self.labels):
            if v >= 0:
                out.append((a * FRAME_S, b * FRAME_S, self.singers[v].id))
            elif v == -2:
                out.append((a * FRAME_S, b * FRAME_S, "overlap"))
        return out

    def mask(self, singer_ids, fs: float, n: int, tail_s: float = 2.0) -> np.ndarray:
        """Sample mask of the chosen singers' sections (overlap and other singers excluded),
        each extended into the silence after it by up to *tail_s* (its reverb / delay tail)."""
        idx = {self.index(s) for s in singer_ids}
        lab = self.labels
        keep = np.isin(lab, list(idx))
        tail = int(round(tail_s / FRAME_S))
        k = 0
        out = keep.copy()
        while k < len(lab):
            if keep[k] and (k + 1 == len(lab) or not keep[k + 1]):
                j = k + 1
                while j < len(lab) and j - k <= tail and lab[j] == -1 and self.activity[j] < 0.02:
                    out[j] = True
                    j += 1
            k += 1
        return _frames_to_samples(out, fs, n)

    def to_dict(self) -> dict:
        return {
            "frame_s": FRAME_S,
            "duration_s": round(self.duration_s, 3),
            "confidence": round(self.confidence, 3),
            "unsure": self.unsure,
            "silhouette": round(self.silhouette, 3),
            "separation": round(self.separation, 3),
            "runs_z": round(self.runs_z, 2),
            "message": self.message,
            "active_s": round(self.active_s, 2),
            "overlap_s": round(self.overlap_s, 2),
            "unassigned_s": round(self.unassigned_s, 2),
            "singers": [s.to_dict() for s in self.singers],
            "segments": [[round(a, 2), round(b, 2), w] for a, b, w in self.segments()],
            "activity": [int(round(100 * v)) for v in self.activity],
        }


def _runs(v: np.ndarray):
    """(start, end, value) runs of a 1-d array."""
    if len(v) == 0:
        return []
    change = np.flatnonzero(np.diff(v)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(v)]
    return [(int(a), int(b), int(v[a])) for a, b in zip(starts, ends)]


def _frames_to_samples(mask: np.ndarray, fs: float, n: int) -> np.ndarray:
    hop = FRAME_S * fs
    idx = np.minimum((np.arange(n) / hop).astype(int), len(mask) - 1)
    return mask[idx] if len(mask) else np.zeros(n, bool)


# ------------------------------------------------------------------------- features
def _mel_matrix(freqs: np.ndarray, n_mels: int = 40, lo: float = 80.0, hi: float = 8000.0) -> np.ndarray:
    def mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    pts = 700.0 * (10 ** (np.linspace(mel(lo), mel(hi), n_mels + 2) / 2595.0) - 1.0)
    m = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - a) / (c - a)
        down = (b - freqs) / (b - c)
        m[i] = np.maximum(0.0, np.minimum(up, down))
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1e-12)


@dataclass
class _Frames:
    level_db: np.ndarray
    active: np.ndarray
    f0: np.ndarray  # Hz, 0 = unvoiced
    mfcc: np.ndarray  # (n, 12)
    envc: np.ndarray  # (n, 12) cepstra of the peak envelope (independent of the harmonic spacing)
    flatness: np.ndarray
    two_pitch: np.ndarray


def _pitch_track(fr: np.ndarray, fs: float, f0_min=70.0, f0_max=1000.0, threshold=0.5) -> np.ndarray:
    """Per-frame f0 from the normalised autocorrelation (first peak within 85 % of the best,
    which avoids sub-octave errors). 0 where unvoiced."""
    win = fr.shape[1]
    x = fr - fr.mean(axis=1, keepdims=True)
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(x * np.hanning(win), n_fft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, n_fft, axis=1)[:, :win]
    ac = ac / np.maximum(ac[:, :1], 1e-20)
    # unbias the window's taper (the autocorrelation of the Hann window itself)
    w_ac = np.fft.irfft(np.abs(np.fft.rfft(np.hanning(win), n_fft)) ** 2, n_fft)[:win]
    ac = ac / np.maximum(w_ac / w_ac[0], 0.05)
    lo, hi = max(2, int(fs / f0_max)), min(win - 2, int(fs / f0_min))
    seg = ac[:, lo:hi]
    best = seg.max(axis=1)
    out = np.zeros(len(fr))
    for i in np.flatnonzero(best > threshold):
        s = seg[i]
        ok = (s >= 0.85 * best[i]) & (s >= np.roll(s, 1)) & (s >= np.roll(s, -1))
        ok[0] = ok[-1] = False
        if not ok.any():
            continue
        k = int(np.flatnonzero(ok)[0])
        # parabolic interpolation of the peak
        a, b, c = s[k - 1], s[k], s[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        out[i] = fs / (lo + k + d)
    return out


def _second_pitch(mag: np.ndarray, freqs: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                  f0_min=70.0, f0_max=1000.0, max_hz=5000.0, peak=1.0, width=0.025, near=0.042) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For voiced frames: the strongest harmonic series left after removing the harmonics of
    f0, as (f0 octave-corrected, f0_2 Hz, salience ratio second / first). 0 where none."""
    n = len(f0)
    f2 = np.zeros(n)
    ratio = np.zeros(n)
    if not voiced.any():
        return f0, f2, ratio
    df = freqs[1] - freqs[0]
    top = int(max_hz / df)
    m = mag[:, :top]
    # whiten: relative to the local spectral envelope (a 400 Hz moving average of the
    # magnitude, which the harmonic peaks dominate; noise between harmonics stays below it)
    env = uniform_filter1d(m, size=max(3, int(400 / df)), axis=1)
    s = np.maximum(0.0, np.log(np.maximum(m, 1e-12) / np.maximum(env, 1e-12)))
    s = maximum_filter1d(s, size=3, axis=1)  # tolerate +-1 bin
    cands = f0_min * 2.0 ** (np.arange(0, np.log2(f0_max / f0_min) * 48) / 48.0)
    H = 12
    hk = np.arange(1, H + 1)
    w = 1.0 / np.sqrt(hk)
    bins = np.rint(np.outer(cands, hk) / df).astype(int)  # (cand, H)
    valid = bins < top
    bins = np.minimum(bins, top - 1)

    f0 = f0.copy()
    for i in np.flatnonzero(voiced):
        spec = s[i].copy()
        # sub-harmonic checks: when the harmonics of f0 / k that are not harmonics of f0 are
        # there (k = 2: 1, 3, 5; k = 3: 1, 2, 4, 5), the pitch is f0 / k (a strong formant can
        # make the 2nd or 3rd harmonic dominate); when the odd harmonics of f0 are all
        # missing, the pitch is 2 f0
        for _ in range(2):
            for k_sub, hs in ((2, (1, 3, 5)), (3, (1, 2, 4, 5))):
                sub = f0[i] / k_sub
                hb = np.rint(sub * np.array(hs) / df).astype(int)
                hb = hb[hb < top]
                if sub >= f0_min and len(hb) >= 2 and np.sum(spec[hb] > peak) >= 2:
                    f0[i] = sub
                    break
            else:
                break
        for _ in range(2):
            odd = np.rint(f0[i] * np.array([1, 3, 5]) / df).astype(int)
            odd = odd[odd < top]
            if len(odd) and np.all(spec[odd] < peak) and 2 * f0[i] <= f0_max:
                f0[i] *= 2
            else:
                break
        b1 = np.rint(f0[i] * hk / df).astype(int)
        b1 = b1[b1 < top]
        s1 = float(np.sum(w[: len(b1)] * spec[b1]))
        if s1 <= 0:
            continue
        # remove f0's harmonics (+-width, at least +-3 bins: vibrato smears them)
        for h, b in enumerate(b1, start=1):
            r = max(2, int(width * h * f0[i] / df))
            spec[max(0, b - r) : b + r + 1] = 0.0
        sal = np.sum(np.where(valid, spec[bins], 0.0) * w, axis=1)
        # the series must have at least 3 of its first 6 harmonics present
        present = (np.where(valid[:, :6], spec[bins[:, :6]], 0.0) > peak).sum(axis=1) >= 3
        ratio_to_f0 = cands / f0[i]
        octave = np.zeros(len(cands), bool)
        for r_ in (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0):
            octave |= np.abs(np.log2(ratio_to_f0 / r_)) < near
        sal = np.where(present & ~octave, sal, 0.0)
        k = int(np.argmax(sal))
        if sal[k] > 0:
            f2[i] = cands[k]
            ratio[i] = sal[k] / s1
    return f0, f2, ratio


def _same(a, b) -> np.ndarray:
    """Pitches within half a semitone (0 = no pitch never matches)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a > 0) & (b > 0) & (np.abs(np.log2(np.maximum(a, 1e-9) / np.maximum(b, 1e-9))) < 0.042)


def _two_voices(f0: np.ndarray, f2: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Frames where two voices sound at once.

    A frame qualifies when its second harmonic series is strong (ratio > TWO_PITCH_RATIO).
    (Note changes inside one analysis frame also show two pitches, for a frame or two: the
    window share OVERLAP_FRAME_SHARE absorbs those.) A second pitch equal to the
    lead's previous note (within TAIL_MEMORY_S) is that note's tail (reverb, delay, legato),
    unless it comes back as the strongest pitch soon after: in a duet the louder voice keeps
    changing, a tail never returns."""
    n = len(f0)
    cand = (ratio > TWO_PITCH_RATIO) & (f2 > 0) & (f0 > 0)
    two = np.zeros(n, bool)
    mem = int(round(TAIL_MEMORY_S / FRAME_S))
    ahead = int(round(0.5 / FRAME_S))
    for i in np.flatnonzero(cand):
        # the previous note: the last pitch before this one (going back from the current note)
        j = i
        while j > 0 and i - j < mem and (f0[j - 1] == 0 or _same(f0[j - 1], f0[i])):
            j -= 1
        k = j
        while k > 0 and i - k < mem and f0[k - 1] > 0 and not _same(f0[k - 1], f0[i]) and _same(f0[k - 1], f0[j - 1]):
            k -= 1
        prev = f0[k:j][f0[k:j] > 0] if j > k else np.zeros(0)
        tail = prev.size > 0 and bool(_same(np.median(prev), f2[i]))
        returns = bool(_same(f0[i + 1 : i + 1 + ahead], f2[i]).any())
        two[i] = (not tail) or returns
    return two


def _frame_features(x: np.ndarray, fs: float) -> tuple[_Frames, int]:
    hop = int(round(FRAME_S * fs))
    win = 1 << int(np.ceil(np.log2(0.08 * fs)))  # ~93 ms at 44.1 kHz
    pad = np.pad(x, (win // 2, win))
    fr = frames(pad, win, hop)
    n = int(np.ceil(len(x) / hop))
    fr = fr[:n]
    level = power_db(np.mean(fr * fr, axis=1))
    loud = np.percentile(level, 95) if len(level) else 0.0
    active = level > max(loud - 35.0, -70.0)
    mag = np.abs(np.fft.rfft(fr * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    pw = min(win, 1 << int(np.ceil(np.log2(0.04 * fs))))  # ~46 ms, centred on the frame
    f0 = _pitch_track(fr[:, win // 2 - pw // 2 : win // 2 + pw // 2], fs)
    f0[~active] = 0.0
    p = mag**2
    melspec = p @ _mel_matrix(freqs).T
    mfcc = dct(np.log(melspec + 1e-12), type=2, norm="ortho", axis=1)[:, 1:13]
    env = maximum_filter1d(mag, size=max(3, int(450 / (freqs[1] - freqs[0]))), axis=1)
    envc = dct(np.log((env**2) @ _mel_matrix(freqs, 24, 100.0, 10000.0).T + 1e-12), type=2, norm="ortho", axis=1)[:, 1:13]
    band = (freqs > 300) & (freqs < 8000)
    pb = p[:, band] + 1e-20
    flat = np.exp(np.mean(np.log(pb), axis=1)) / np.mean(pb, axis=1)
    voiced = f0 > 0
    f0, f2, ratio = _second_pitch(mag, freqs, f0, voiced)
    two = _two_voices(f0, f2, ratio)
    return _Frames(level, active, f0, mfcc, envc, flat, two), n


# ------------------------------------------------------------------------ clustering
def _silhouette(z: np.ndarray, lab: np.ndarray) -> float:
    ks = np.unique(lab)
    if len(ks) < 2:
        return 0.0
    d = np.sqrt(np.maximum(0.0, ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)))
    s = np.zeros(len(z))
    for i in range(len(z)):
        own = lab == lab[i]
        a = d[i, own].sum() / max(own.sum() - 1, 1)
        b = min(d[i, lab == k].mean() for k in ks if k != lab[i])
        s[i] = (b - a) / max(a, b, 1e-12)
    return float(s.mean())


def _phrases(active: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Runs of activity separated by at least *min_gap* inactive frames."""
    out, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
            last = i
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                out.append((start, last + 1))
                start = None
    if start is not None:
        out.append((start, last + 1))
    return out


def _embedding(fr: _Frames, sel: np.ndarray) -> np.ndarray | None:
    """The singer embedding of the frames *sel* (bool mask): see the module docstring."""
    act = sel & fr.active
    voiced = act & (fr.f0 > 0)
    if act.sum() < 4 or voiced.sum() < 3:
        return None
    lf0 = np.log2(fr.f0[voiced])
    return np.r_[fr.mfcc[act].mean(axis=0), fr.envc[act].mean(axis=0), np.percentile(lf0, [10, 50, 90]),
                 np.log(np.median(fr.flatness[act]) + 1e-9), np.std(fr.level_db[act])]


def detect_singers(vocals: np.ndarray, fs: float, max_singers: int = MAX_SINGERS) -> SingerMap:
    """Who sings when in a vocal stem ((n,), (n, ch) or (ch, n)). See the module docstring."""
    x = to_mono(vocals)
    duration = len(x) / fs
    fr, n = _frame_features(x, fs)
    loud = np.percentile(fr.level_db, 95) if n else 0.0
    activity = np.clip((fr.level_db - (loud - 50.0)) / 50.0, 0.0, 1.0)
    activity[~fr.active] = np.minimum(activity[~fr.active], 0.01)
    labels = np.full(n, -1, dtype=int)
    result = SingerMap(duration, labels, activity, active_s=float(fr.active.sum() * FRAME_S))

    # 1. Overlap: windows where many voiced frames carry two pitches.
    per, step = int(round(WIN_S / FRAME_S)), int(round(HOP_S / FRAME_S))
    ov = np.zeros(n, bool)
    for a in range(0, max(1, n - per + 1), step):
        b = min(n, a + per)
        v = fr.f0[a:b] > 0
        if v.sum() >= 3 and fr.two_pitch[a:b][v].mean() >= OVERLAP_FRAME_SHARE:
            ov[a:b] |= fr.active[a:b]
    # Conservative: a phrase that is largely overlap is overlap as a whole (between the
    # detected windows, both singers are usually still there, just harder to see).
    for a, b in _phrases(fr.active, int(round(0.25 / FRAME_S))):
        act = fr.active[a:b]
        if act.any() and ov[a:b][act].mean() >= OVERLAP_PHRASE_SHARE:
            ov[a:b] |= act
    solo = fr.active & ~ov

    # 2. Chunks: phrases (split at pauses of 250 ms or more), long ones cut into ~CHUNK_S pieces.
    chunks = []
    for a, b in _phrases(solo, int(round(0.25 / FRAME_S))):
        m = max(1, int(round((b - a) * FRAME_S / CHUNK_S)))
        edges = np.linspace(a, b, m + 1).round().astype(int)
        chunks += [(int(p), int(q)) for p, q in zip(edges[:-1], edges[1:]) if (q - p) * FRAME_S >= MIN_CHUNK_S]
    emb, halves, keep = [], [], []
    for a, b in chunks:
        sel = np.zeros(n, bool)
        sel[a:b] = True
        e = _embedding(fr, sel)
        h1 = np.zeros(n, bool)
        h1[a : (a + b) // 2] = True
        e1, e2 = _embedding(fr, h1 & sel), _embedding(fr, ~h1 & sel)
        if e is None:
            continue
        emb.append(e)
        keep.append((a, b))
        if e1 is not None and e2 is not None:
            halves.append(e1 - e2)
    if not emb:
        result.confidence = 0.0
        result.message = "No singing found in the vocal."
        return result
    E = np.array(emb)
    chunks = keep

    # 3. Features in units of the noise of one chunk, measured within phrases (one phrase =
    #    one singer: what changes with the vowel or the note counts little, what stays put
    #    within a singer counts), then the 3 main directions of variation (with a few dozen
    #    chunks, more dimensions would let any split look clean).
    if len(halves) >= 4:
        noise = np.sqrt(np.mean(np.array(halves) ** 2, axis=0) / 4.0) + 1e-9
    else:
        noise = E.std(axis=0) + 1e-9
    Z = (E - E.mean(axis=0)) / noise
    w = np.array([max(1, b - a) for a, b in chunks], float)
    k_best, lab_best, sep_best, z_best = 1, np.zeros(len(Z), int), 0.0, 0.0
    if len(Z) >= 6:
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        Z = Z @ vt[: min(N_COMPONENTS, len(vt))].T
        tree = linkage(Z, method="ward")
        total_s = float(w.sum() * FRAME_S)
        for k in range(2, max_singers + 1):
            lab = fcluster(tree, k, criterion="maxclust") - 1
            if len(np.unique(lab)) < k:
                break
            secs = np.array([w[lab == c].sum() * FRAME_S for c in range(k)])
            if secs.min() < max(MIN_CLUSTER_S, MIN_CLUSTER_SHARE * total_s) * (1.0 if k == 2 else 2.0):
                break
            sep = _separation(Z, lab)
            z = _runs_z(lab)
            if _accept(sep, z) and sep > (1.15 * sep_best if k > 2 else 0.0):
                k_best, lab_best, sep_best, z_best = k, lab, sep, z
            else:
                break
        if k_best == 1:
            # how close a second singer came (for the confidence)
            lab = fcluster(tree, 2, criterion="maxclust") - 1
            if len(np.unique(lab)) == 2:
                sep_best, z_best = _separation(Z, lab), _runs_z(lab)
    result.silhouette = _silhouette(Z, lab_best) if k_best > 1 else 0.0
    result.separation = sep_best
    result.runs_z = z_best

    # 4. Labels: singers in order of appearance; chunks too close to another singer stay unassigned.
    first = {}
    for (a, _), l in zip(chunks, lab_best):
        first.setdefault(int(l), a)
    order = sorted(first, key=first.get)
    remap = {old: new for new, old in enumerate(order)}
    lab_best = np.array([remap[int(l)] for l in lab_best])
    cent = np.array([Z[lab_best == c].mean(axis=0) for c in range(k_best)])
    for (a, b), z, l in zip(chunks, Z, lab_best):
        if k_best > 1:
            d = np.sqrt(((cent - z) ** 2).sum(axis=1))
            other = np.min(np.delete(d, l))
            if d[l] > UNCERTAIN_DISTANCE_RATIO * other:
                continue
        labels[a:b][solo[a:b]] = l
    labels[ov] = -2
    # close short gaps (< 0.3 s) inside one singer's phrase, so the phrase stays whole
    for a, b, v in _runs(labels):
        if v == -1 and 0 < a and b < n and labels[a - 1] == labels[b] >= 0 and (b - a) * FRAME_S < 0.3:
            labels[a:b] = labels[b]

    result.overlap_s = float(ov.sum() * FRAME_S)
    singing_s = float(solo.sum() * FRAME_S)
    assigned_s = float(((labels >= 0) & fr.active).sum() * FRAME_S)
    result.unassigned_s = max(0.0, singing_s - assigned_s)
    for i in range(k_best):
        sel = labels == i
        secs = float((sel & fr.active).sum() * FRAME_S)
        f0s = fr.f0[sel & (fr.f0 > 0)]
        runs = [(a, b) for a, b, v in _runs(labels) if v == i]
        a, b = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)
        mid = (a + b) / 2 * FRAME_S
        half = min(2.0, (b - a) * FRAME_S / 2)
        result.singers.append(SingerInfo(LETTERS[i], secs, (max(0.0, mid - half), mid + half),
                                         secs / max(assigned_s, 1e-9), float(np.median(f0s)) if f0s.size else 0.0))

    covered = assigned_s / max(singing_s, 1e-9)
    if k_best == 1:
        # one voice; less sure when a split came close to the thresholds
        # (a split that came close to the thresholds, by separation and by turn-taking)
        near = min(max(0.0, sep_best - 0.75) / 0.25, max(0.0, -z_best) / 1.0)
        result.confidence = float(np.clip((0.6 + 0.4 * covered) * (1.0 - 0.5 * near), 0.0, 1.0))
    else:
        shares = np.array([s.share for s in result.singers])
        balance = float(np.clip(shares.min() * k_best * 2.0, 0.0, 1.0))
        clarity = float(np.clip((sep_best - 1.0) / 0.25, 0.0, 1.0))
        result.confidence = (0.4 + 0.6 * clarity) * (0.6 + 0.4 * covered) * (0.7 + 0.3 * balance)
    result.message = describe(result)
    return result


def _separation(Z: np.ndarray, lab: np.ndarray) -> float:
    """How clearly the clusters *lab* are apart, 1 = just at the acceptance thresholds: the
    smallest distance between two cluster centres (in chunk-noise units, see step 3) over
    MIN_CENTRE_DISTANCE, and that distance relative to the spread inside the clusters over
    MIN_CENTRE_RATIO; the smaller of the two."""
    ks = np.unique(lab)
    cent = np.array([Z[lab == k].mean(axis=0) for k in ks])
    spread = np.sqrt(np.mean(np.concatenate([((Z[lab == k] - cent[i]) ** 2).ravel() for i, k in enumerate(ks)])))
    dmin = min(np.linalg.norm(cent[i] - cent[j]) for i in range(len(ks)) for j in range(i + 1, len(ks)))
    return float(min(dmin / MIN_CENTRE_DISTANCE, dmin / max(spread, 1e-9) / MIN_CENTRE_RATIO))


def _runs_z(lab: np.ndarray, n_perm: int = 400) -> float:
    """Wald-Wolfowitz runs test of the chunk labels in time order, as a z score against random
    orders of the same labels: singers take turns for several phrases at a time (few runs,
    z well below 0); a split of one voice by vowel or register flips at random (z near 0)."""
    if len(lab) < 4 or len(np.unique(lab)) < 2:
        return 0.0
    runs = 1 + int(np.sum(lab[1:] != lab[:-1]))
    rng = np.random.default_rng(0)
    perm = np.array([1 + np.sum(np.diff(rng.permutation(lab)) != 0) for _ in range(n_perm)])
    return float((runs - perm.mean()) / max(perm.std(), 1e-9))


def _accept(sep: float, z: float) -> bool:
    """A split into singers: clearly apart and taking turns, or very clearly taking turns."""
    return (sep >= 1.0 and z <= -1.0) or (sep >= 0.75 and z <= -2.5)


def describe(m: SingerMap) -> str:
    total = m.active_s
    parts = []
    if len(m.singers) == 1:
        parts.append("One singer found.")
    elif m.singers:
        parts.append(f"{len(m.singers)} singers found: " + ", ".join(f"{s.id} {s.seconds:.0f} s" for s in m.singers) + ".")
    if m.overlap_s > 0 and len(m.singers) == 1:
        parts.append(f"Two voices at once (doubles or harmonies) in {m.overlap_s:.1f} s "
                     f"({100 * m.overlap_s / max(total, 1e-9):.0f} %): left out when a singer is chosen.")
    elif m.overlap_s > 0:
        parts.append(f"Overlap excluded: {m.overlap_s:.1f} s ({100 * m.overlap_s / max(total, 1e-9):.0f} % of the singing).")
    if m.unsure:
        parts.append(f"Singer detection is unsure (confidence {100 * m.confidence:.0f} %): "
                     "mark the sections to analyse with region selection instead.")
    return " ".join(parts)
