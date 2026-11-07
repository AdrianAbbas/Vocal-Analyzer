"""Which parts of the reference vocal are analysed: regions marked by hand (--regions, the
plug-in's region selection) and/or the sections of chosen singers (--singer, singers.py).

Everything is a sample mask over the reference vocal stem, in the song's timeline (a Demucs
stem has the song's length): the separation's section selection (where the vocal is clearly
present), the marked regions, and the chosen singers' sections are combined with AND, and the
kept runs are joined with 5 ms fades.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

JOIN_FADE_S = 0.005


def parse_time(text: str) -> float:
    """'83.5', '1:23.5' or '0:01:23.5' -> seconds."""
    parts = text.strip().split(":")
    if not parts or any(p.strip() == "" for p in parts) or len(parts) > 3:
        raise ValueError(f"not a time: {text!r}")
    value = 0.0
    for p in parts:
        value = value * 60.0 + float(p)
    if value < 0:
        raise ValueError(f"negative time: {text!r}")
    return value


def parse_regions(spec: str) -> list[tuple[float, float]]:
    """'12.5-30, 1:05-1:20.5' -> [(12.5, 30.0), (65.0, 80.5)], sorted and merged where they
    overlap. Raises ValueError on a malformed or empty region."""
    out = []
    for item in re.split(r"[,;\s]+", spec.strip()):
        if not item:
            continue
        m = re.fullmatch(r"([0-9:.]+)-([0-9:.]+)", item)
        if not m:
            raise ValueError(f"not a region (start-end): {item!r}")
        a, b = parse_time(m.group(1)), parse_time(m.group(2))
        if b <= a:
            raise ValueError(f"region ends before it starts: {item!r}")
        out.append((a, b))
    if not out:
        raise ValueError("no regions given")
    return merge_regions(out)


def merge_regions(regions) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in regions):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def format_regions(regions) -> str:
    return ",".join(f"{a:.2f}-{b:.2f}" for a, b in regions)


def regions_mask(regions, fs: float, n: int) -> np.ndarray:
    mask = np.zeros(n, bool)
    for a, b in regions:
        mask[max(0, int(round(a * fs))) : min(n, int(round(b * fs)))] = True
    return mask


def join_masked(audio: np.ndarray, mask: np.ndarray, fs: float) -> np.ndarray:
    """The samples of *audio* ((n,) or (n, ch)) where *mask* is set, runs joined with short
    fades (no clicks at the joins)."""
    a = np.asarray(audio)
    if not mask.any():
        return a[:0]
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    fade = max(1, int(round(JOIN_FADE_S * fs)))
    pieces = []
    for s, e in zip(edges[::2], edges[1::2]):
        p = np.array(a[s:e], dtype=np.float64, copy=True)
        k = min(fade, len(p) // 2)
        if k > 0:
            ramp = np.linspace(0.0, 1.0, k, endpoint=False)
            if p.ndim == 2:
                ramp = ramp[:, None]
            if s > 0:
                p[:k] *= ramp
            if e < len(a):
                p[len(p) - k :] *= ramp[::-1]
        pieces.append(p)
    return np.concatenate(pieces)


@dataclass
class ReferenceSelection:
    """What was analysed, for the report and preset.reference.selection."""

    audio: np.ndarray  # (n, ch) joined
    analysed_s: float
    total_s: float
    regions: list = field(default_factory=list)
    singers: list = field(default_factory=list)
    overlap_excluded_s: float = 0.0
    singer_confidence: float | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"analysed_s": round(self.analysed_s, 2), "total_s": round(self.total_s, 2)}
        if self.regions:
            d["regions"] = [[round(a, 3), round(b, 3)] for a, b in self.regions]
        if self.singers:
            d["singers"] = list(self.singers)
            d["overlap_excluded_s"] = round(self.overlap_excluded_s, 2)
        if self.singer_confidence is not None:
            d["singer_confidence"] = round(float(self.singer_confidence), 3)
        return d


def select_reference(stem: np.ndarray, fs: float, base_mask: np.ndarray | None = None, regions=None,
                     singer_map=None, singers=None) -> ReferenceSelection:
    """The part of *stem* ((n, ch)) to analyse: *base_mask* (the separation's selection, or
    all), within *regions* (seconds, or None = everywhere), restricted to the sections of
    *singers* (ids in *singer_map*, or None = everyone; overlaps are excluded then)."""
    x = np.asarray(stem)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    total_s = n / fs
    mask = np.ones(n, bool) if base_mask is None else np.asarray(base_mask, bool)[:n].copy()
    if len(mask) < n:
        mask = np.r_[mask, np.zeros(n - len(mask), bool)]
    sel = ReferenceSelection(x[:0], 0.0, total_s)
    if regions:
        regions = merge_regions(regions)
        beyond = [r for r in regions if r[0] >= total_s]
        if beyond:
            sel.notes.append(f"{len(beyond)} region(s) start after the end of the song ({total_s:.1f} s)")
        mask &= regions_mask(regions, fs, n)
        sel.regions = regions
    if singers:
        sm = singer_map.mask(singers, fs, n)
        sel.singers = [s.strip().upper() for s in singers]
        # overlap that would have been analysed without the singer choice
        ov = _frames_to_sample_mask(singer_map.labels == -2, singer_map, fs, n)
        sel.overlap_excluded_s = float((mask & ov).sum() / fs)
        mask &= sm
        sel.singer_confidence = singer_map.confidence
    sel.analysed_s = float(mask.sum() / fs)
    sel.audio = join_masked(x, mask, fs)
    return sel


def _frames_to_sample_mask(frame_mask, singer_map, fs, n):
    from .singers import _frames_to_samples

    return _frames_to_samples(np.asarray(frame_mask), fs, n)
