"""Source separation (milestone 4): pull the lead vocal out of a full song with Demucs
(htdemucs), cache the stems on disk, and keep only the sections where the vocal is clearly
there.

Demucs is an optional dependency (PyTorch is large): ``pip install "vocalmatch[separation]"``
or ``bash install.sh --with-separation``. It runs as ``python -m demucs --two-stems vocals``
in a subprocess, so nothing here imports torch.

Cache: ``<song>.stems/<model>/vocals.wav`` and ``no_vocals.wav`` next to the song (git
ignores ``*.stems/``), with the song's SHA-256 so an edited song is separated again. When the
song's folder is not writable, ``$VOCALMATCH_CACHE`` or ``~/.cache/vocalmatch/stems/<sha>``.

Section selection: the vocal stem is cut into sections at its silences (>= 250 ms, 40 dB
below its loud level). A section is kept when it is mostly voiced and the vocal is not
buried under the accompaniment (median vocal-to-accompaniment ratio >= -15 dB, i.e. the
stem is not mostly bleed). Sections are joined at the silences, each with the silence that
follows it (up to 2 s), so reverb and delay tails stay intact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import frame_rms_db, to_mono, voicing

MODEL = "htdemucs"
FRAME_MS = 50.0
SILENCE_REL_DB = 40.0
MIN_SILENCE_MS = 250.0
MIN_SNR_DB = -15.0
MIN_VOICED = 0.3
MAX_TAIL_S = 2.0  # of the silence after a kept section (its reverb / delay tail)


class SeparationUnavailable(RuntimeError):
    pass


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir_for(song: Path, sha: str, model: str = MODEL) -> Path:
    local = song.with_name(song.name + ".stems") / model
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".write-test"
        probe.write_text("")
        probe.unlink()
        return local
    except OSError:
        base = Path(os.environ.get("VOCALMATCH_CACHE", Path.home() / ".cache" / "vocalmatch" / "stems"))
        d = base / sha / model
        d.mkdir(parents=True, exist_ok=True)
        return d


def run_demucs(song: Path, out_dir: Path, model: str = MODEL) -> tuple[Path, Path]:
    """Separate *song* into out_dir/vocals.wav and out_dir/no_vocals.wav."""
    if not demucs_available():
        raise SeparationUnavailable(
            "Demucs is not installed. Install it with:  pip install \"vocalmatch[separation]\"  "
            "(or bash install.sh --with-separation), or pass --separate no with an isolated vocal stem.")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model, "-o", tmp, str(song)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"demucs failed ({r.returncode}): {r.stderr.strip()[-800:]}")
        found = list(Path(tmp).rglob("vocals.wav"))
        if not found:
            raise RuntimeError(f"demucs wrote no vocals.wav (output: {r.stdout[-400:]})")
        src = found[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "vocals.wav", out_dir / "vocals.wav")
        shutil.copy(src / "no_vocals.wav", out_dir / "no_vocals.wav")
    return out_dir / "vocals.wav", out_dir / "no_vocals.wav"


def separate(song: str | os.PathLike, model: str = MODEL, separator=run_demucs) -> tuple[Path, Path, bool]:
    """(vocals.wav, no_vocals.wav, from_cache). *separator(song, out_dir, model)* is the
    separation step (Demucs; tests pass a stand-in)."""
    song = Path(song)
    sha = file_sha256(song)
    d = cache_dir_for(song, sha, model)
    stamp = d / "source.sha256"
    vocals, rest = d / "vocals.wav", d / "no_vocals.wav"
    if vocals.is_file() and rest.is_file() and stamp.is_file() and stamp.read_text().strip() == sha:
        return vocals, rest, True
    separator(song, d, model)
    stamp.write_text(sha + "\n")
    return vocals, rest, False


@dataclass
class Selection:
    audio: np.ndarray  # (n, ch) the kept sections, joined
    kept_s: float
    total_s: float
    sections: list = field(default_factory=list)  # (start_s, end_s, kept, snr_db, voiced)

    @property
    def diagnostics(self) -> dict:
        return {"kept_s": round(self.kept_s, 2), "total_s": round(self.total_s, 2),
                "sections": [[round(a, 2), round(b, 2), k, round(s, 1), round(v, 2)] for a, b, k, s, v in self.sections]}


def select_sections(vocals: np.ndarray, accompaniment: np.ndarray | None, fs: float) -> Selection:
    """Keep the sections where the vocal is clearly present (see the module docstring)."""
    v = np.asarray(vocals, dtype=np.float64)
    v2 = v if v.ndim == 2 else v[:, None]
    mono = to_mono(v2)
    hop = int(round(FRAME_MS * fs / 1000))
    lv = frame_rms_db(mono, fs, win_ms=FRAME_MS, hop_ms=FRAME_MS)
    la = frame_rms_db(to_mono(accompaniment), fs, win_ms=FRAME_MS, hop_ms=FRAME_MS) if accompaniment is not None else None
    voiced = voicing(mono, fs, win=hop, hop=hop)
    n = min(len(lv), len(voiced), len(la) if la is not None else len(lv))
    lv, voiced = lv[:n], voiced[:n]
    loud = np.percentile(lv, 95)
    silent = lv < loud - SILENCE_REL_DB
    min_sil = int(round(MIN_SILENCE_MS / FRAME_MS))

    # sections: runs of sound separated by silences of at least MIN_SILENCE_MS
    bounds, k, start = [], 0, None
    while k < n:
        if not silent[k]:
            if start is None:
                start = k
            k += 1
            continue
        j = k
        while j < n and silent[j]:
            j += 1
        if start is not None and (j - k >= min_sil or j == n):
            bounds.append((start, k, j))  # sound [start, k), then silence until j
            start = None
        k = j
    if start is not None:
        bounds.append((start, n, n))

    pieces, sections, kept = [], [], 0
    for s, e, gap_end in bounds:
        act = ~silent[s:e]
        snr = float(np.median((lv[s:e] - la[:n][s:e])[act])) if la is not None and act.any() else 0.0
        vf = float(np.mean(voiced[s:e][act])) if act.any() else 0.0
        keep = vf >= MIN_VOICED and snr >= MIN_SNR_DB
        sections.append((s * hop / fs, e * hop / fs, keep, snr, vf))
        if keep:
            a, b = s * hop, min(gap_end * hop, e * hop + int(MAX_TAIL_S * fs), len(v2))
            pieces.append(v2[a:b])
            kept += b - a
    if not pieces:  # nothing passes: analyse everything rather than nothing
        return Selection(v2, len(v2) / fs, len(v2) / fs, sections)
    return Selection(np.concatenate(pieces), kept / fs, len(v2) / fs, sections)
