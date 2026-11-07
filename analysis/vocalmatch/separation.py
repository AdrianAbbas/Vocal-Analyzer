"""Source separation (milestone 4): pull the lead vocal out of a full song with Demucs
(htdemucs), cache the stems on disk, and keep only the sections where the vocal is clearly
there.

Demucs + PyTorch are installed by default by install.sh / install-mac.sh (the
``vocalmatch[separation]`` extra). The model runs in its own process
(``python -m vocalmatch.demucs_worker``, see that module): the song is decoded once with
libsndfile (MP3 included, no ffmpeg needed) and separated in 20 s chunks, each with 2 s of
context on both sides, so memory stays flat on long songs and the plug-in gets real
progress. A crash or an out-of-memory kill of the worker is reported as an error, not as
an empty result, and the analyzer's own memory is free of PyTorch afterwards.

Cache: ``~/Library/Caches/VocalMatch/stems/<sha>/<model>-v2/`` on macOS,
``~/.cache/vocalmatch/stems/...`` elsewhere (``$VOCALMATCH_CACHE`` overrides), keyed by
the song's SHA-256, so a second Analyze of the same song skips the separation and an edited
song is separated again. The stems are 24-bit FLAC at -6 dB (``source.json`` holds the
gain); only the most recent CACHE_KEEP songs are kept.

Without separation, a reference that looks like a full mix (bass and drums under the vocal,
no silences between phrases: looks_like_full_mix) is refused with a clear message rather
than analysed as if it were a vocal.

Section selection: the vocal stem is cut into sections at its silences (>= 250 ms, 40 dB
below its loud level). A section is kept when it is mostly voiced and the vocal is not
buried under the accompaniment (median vocal-to-accompaniment ratio >= -15 dB, i.e. the
stem is not mostly bleed). Sections are joined at the silences, each with the silence that
follows it (up to 2 s), so reverb and delay tails stay intact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import frame_rms_db, to_mono, voicing

MODEL = "htdemucs"
CACHE_VERSION = 2  # chunked separation, FLAC stems
CACHE_KEEP = 8  # songs
STEM_GAIN = 0.5  # stems are stored at -6 dB (headroom in 24-bit FLAC)
FRAME_MS = 50.0
SILENCE_REL_DB = 40.0
MIN_SILENCE_MS = 250.0
MIN_SNR_DB = -15.0
MIN_VOICED = 0.3
MAX_TAIL_S = 2.0  # of the silence after a kept section (its reverb / delay tail)

NOT_INSTALLED = "Source separation not installed: run the installer again or use an isolated vocal"


class SeparationUnavailable(RuntimeError):
    pass


class SeparationFailed(RuntimeError):
    pass


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def real_home() -> Path:
    """The user's home folder, even when $HOME points into a host's sandbox container."""
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError, OSError):
        return Path.home()


def app_data_dir() -> Path:
    if sys.platform == "darwin":
        return real_home() / "Library" / "Application Support" / "VocalMatch"
    return Path(os.environ.get("XDG_DATA_HOME", real_home() / ".local" / "share")) / "vocalmatch"


def models_dir() -> Path:
    """Where the separation model's weights are kept (downloaded once, about 80 MB)."""
    return app_data_dir() / "models"


def cache_root() -> Path:
    if os.environ.get("VOCALMATCH_CACHE"):
        return Path(os.environ["VOCALMATCH_CACHE"])
    if sys.platform == "darwin":
        return real_home() / "Library" / "Caches" / "VocalMatch" / "stems"
    return Path(os.environ.get("XDG_CACHE_HOME", real_home() / ".cache")) / "vocalmatch" / "stems"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir_for(song: Path, sha: str, model: str = MODEL) -> Path:
    for base in (cache_root(), Path(tempfile.gettempdir()) / "vocalmatch-stems"):
        d = base / sha[:24] / f"{model}-v{CACHE_VERSION}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write-test"
            probe.write_text("")
            probe.unlink()
            return d
        except OSError:
            continue
    raise SeparationFailed(f"no writable folder for the separated stems (tried {cache_root()})")


def prune_cache(keep: int = CACHE_KEEP, root: Path | None = None) -> None:
    """Delete all but the *keep* most recently used songs' stems."""
    root = root or cache_root()
    try:
        songs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return
    for d in songs[keep:]:
        shutil.rmtree(d, ignore_errors=True)


def worker_env() -> dict:
    """Environment for the separation worker: bounded thread pools (OpenMP / MKL /
    Accelerate), the model cache in the app's own folder, no user site-packages."""
    env = dict(os.environ)
    threads = str(max(1, min(4, os.cpu_count() or 1)))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, threads)
    env.setdefault("TORCH_HOME", str(models_dir() / "torch"))
    env.setdefault("HF_HOME", str(models_dir() / "huggingface"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    for k in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    return env


def _signal_name(code: int) -> str:
    import signal

    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


def run_demucs(song: Path, out_dir: Path, model: str = MODEL, progress=None) -> tuple[Path, Path]:
    """Separate *song* into out_dir/vocals.flac and out_dir/no_vocals.flac (the worker
    process; *progress(fraction, stage)*)."""
    if not demucs_available():
        raise SeparationUnavailable(NOT_INSTALLED)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", "-m", "vocalmatch.demucs_worker", str(song), str(out_dir), "--model", model]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env(),
                            bufsize=1)
    errors: list[str] = []
    err_lines: list[str] = []

    def pump_stderr():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\n"))
            print(f"[separation] {line.rstrip()}", file=sys.stderr, flush=True)

    t = threading.Thread(target=pump_stderr, daemon=True)
    t.start()
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PROGRESS "):
                parts = line.split(" ", 2)
                if progress is not None:
                    try:
                        progress(float(parts[1]), parts[2] if len(parts) > 2 else "separating")
                    except ValueError:
                        pass
            elif line.startswith("ERROR "):
                errors.append(line[6:])
            else:
                print(f"[separation] {line}", file=sys.stderr, flush=True)
        code = proc.wait()
    finally:
        if proc.poll() is None:  # cancelled (SIGTERM / Ctrl-C): stop the worker too
            proc.kill()
            proc.wait()
    t.join(5)
    if code != 0:
        if errors:
            raise SeparationFailed(errors[-1])
        if code < 0:
            hint = (" (macOS stopped it, most likely because it ran out of memory: close other apps and try again)"
                    if -code == 9 else "")
            raise SeparationFailed(f"the source separation crashed ({_signal_name(code)}){hint}")
        tail = " | ".join(err_lines[-3:])
        raise SeparationFailed(f"the source separation stopped (exit {code}){': ' + tail if tail else ''}")
    vocals, rest = out_dir / "vocals.flac", out_dir / "no_vocals.flac"
    if not vocals.is_file() or not rest.is_file():
        raise SeparationFailed("the source separation wrote no stems")
    return vocals, rest


def separate(song: str | os.PathLike, model: str = MODEL, separator=None, progress=None) -> tuple[Path, Path, bool]:
    """(vocals, no_vocals, from_cache). *separator(song, out_dir, model, progress=...)* is
    the separation step (Demucs; tests pass a stand-in). The stems are stored at STEM_GAIN:
    load them with load_stem()."""
    song = Path(song)
    sha = file_sha256(song)
    d = cache_dir_for(song, sha, model)
    stamp = d / "source.json"
    try:
        meta = json.loads(stamp.read_text())
    except (OSError, ValueError):
        meta = {}
    if meta.get("sha256") == sha and meta.get("complete"):
        vocals, rest = Path(d / meta["vocals"]), Path(d / meta["no_vocals"])
        if vocals.is_file() and rest.is_file():
            os.utime(d.parent)  # most recently used
            return vocals, rest, True
    stamp.unlink(missing_ok=True)
    vocals, rest = (separator or run_demucs)(song, d, model, progress=progress)
    stamp.write_text(json.dumps({"sha256": sha, "song": str(song), "model": model, "version": CACHE_VERSION,
                                 "vocals": Path(vocals).name, "no_vocals": Path(rest).name, "gain": STEM_GAIN,
                                 "complete": True}) + "\n")
    os.utime(d.parent)
    prune_cache()
    return Path(vocals), Path(rest), False


def load_stem(path: str | os.PathLike, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    """A cached stem at the song's level (undoes STEM_GAIN)."""
    from .audio_io import load_audio

    x, fs = load_audio(str(path), target_fs=target_fs)
    gain = STEM_GAIN
    try:
        gain = float(json.loads((Path(path).parent / "source.json").read_text()).get("gain", STEM_GAIN))
    except (OSError, ValueError):
        pass
    return x / gain, fs


def full_mix_features(x: np.ndarray, fs: float) -> dict:
    """What tells a full mix from an isolated vocal: the share of energy below 80 Hz (kick
    and bass; a vocal has almost none) and the share of 50 ms frames that are silent (40 dB
    below the loud level: the gaps between sung phrases; a mix has none)."""
    from scipy.signal import welch

    mono = to_mono(x)
    f, p = welch(mono, fs=fs, nperseg=8192)
    total = float(np.sum(p[f >= 20])) + 1e-30
    low = float(np.sum(p[(f >= 20) & (f < 80)])) / total
    lv = frame_rms_db(mono, fs, win_ms=FRAME_MS, hop_ms=FRAME_MS)
    loud = np.percentile(lv, 95)
    silent = float(np.mean(lv < loud - SILENCE_REL_DB))
    return {"low_80hz": round(low, 4), "silent": round(silent, 4)}


def looks_like_full_mix(x: np.ndarray, fs: float) -> tuple[bool, dict]:
    feats = full_mix_features(x, fs)
    return (feats["low_80hz"] > 0.02 and feats["silent"] < 0.10), feats


@dataclass
class Selection:
    audio: np.ndarray  # (n, ch) the kept sections, joined
    kept_s: float
    total_s: float
    sections: list = field(default_factory=list)  # (start_s, end_s, kept, snr_db, voiced)
    mask: np.ndarray | None = None  # (n,) per sample of the stem: kept

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
    mask = np.zeros(len(v2), bool)
    for s, e, gap_end in bounds:
        act = ~silent[s:e]
        snr = float(np.median((lv[s:e] - la[:n][s:e])[act])) if la is not None and act.any() else 0.0
        vf = float(np.mean(voiced[s:e][act])) if act.any() else 0.0
        keep = vf >= MIN_VOICED and snr >= MIN_SNR_DB
        sections.append((s * hop / fs, e * hop / fs, keep, snr, vf))
        if keep:
            a, b = s * hop, min(gap_end * hop, e * hop + int(MAX_TAIL_S * fs), len(v2))
            pieces.append(v2[a:b])
            mask[a:b] = True
            kept += b - a
    if not pieces:  # nothing passes: analyse everything rather than nothing
        return Selection(v2, len(v2) / fs, len(v2) / fs, sections, np.ones(len(v2), bool))
    return Selection(np.concatenate(pieces), kept / fs, len(v2) / fs, sections, mask)
