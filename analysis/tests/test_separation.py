"""Separation cache, full-mix detection, the worker's chunking and section selection, with
stand-ins for Demucs (the real model runs in the macOS install test and tests/listening)."""

import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from vocalmatch import separation as S
from vocalmatch.synth import synthetic_song, synthetic_vocal

FS = 44100


def _fake_separator(calls):
    def run(song, out_dir, model, progress=None):
        calls.append(song)
        x, fs = sf.read(str(song), always_2d=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / "vocals.flac"), x * 0.9 * S.STEM_GAIN, fs, subtype="PCM_24")
        sf.write(str(out_dir / "no_vocals.flac"), x * 0.1 * S.STEM_GAIN, fs, subtype="PCM_24")
        if progress:
            progress(1.0, "separated")
        return out_dir / "vocals.flac", out_dir / "no_vocals.flac"
    return run


def test_stems_are_cached_by_content_and_redone_when_the_song_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("VOCALMATCH_CACHE", str(tmp_path / "cache"))
    song = tmp_path / "song.wav"
    sf.write(str(song), synthetic_vocal(seed=1, duration_s=2.0, fs=FS), FS)
    calls = []
    v, r, cached = S.separate(song, separator=_fake_separator(calls))
    assert not cached and v.is_file() and r.is_file()
    assert v.parent.parent.parent == tmp_path / "cache" and not (tmp_path / "song.wav.stems").exists()
    x, _ = sf.read(str(song))
    y, _ = S.load_stem(v)
    assert np.allclose(y[:, 0], 0.9 * x, atol=1e-5)  # the stored gain is undone
    copy = tmp_path / "same song.wav"  # the same audio under another name: still cached
    copy.write_bytes(song.read_bytes())
    v2, _, cached2 = S.separate(copy, separator=_fake_separator(calls))
    assert cached2 and v2 == v and len(calls) == 1
    sf.write(str(song), synthetic_vocal(seed=2, duration_s=2.0, fs=FS), FS)  # edited song
    _, _, cached3 = S.separate(song, separator=_fake_separator(calls))
    assert not cached3 and len(calls) == 2


def test_an_interrupted_separation_is_not_taken_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VOCALMATCH_CACHE", str(tmp_path / "cache"))
    song = tmp_path / "song.wav"
    sf.write(str(song), synthetic_vocal(seed=1, duration_s=2.0, fs=FS), FS)

    def crash(song, out_dir, model, progress=None):
        (out_dir / "vocals.flac").write_bytes(b"partial")
        raise S.SeparationFailed("the source separation crashed (SIGSEGV)")

    with pytest.raises(S.SeparationFailed):
        S.separate(song, separator=crash)
    calls = []
    _, _, cached = S.separate(song, separator=_fake_separator(calls))
    assert not cached and len(calls) == 1


def test_the_cache_keeps_the_most_recent_songs(tmp_path):
    import os
    import time

    for i in range(5):
        d = tmp_path / f"song{i}" / "htdemucs-v2"
        d.mkdir(parents=True)
        os.utime(d.parent, (time.time() - 100 + i, time.time() - 100 + i))
    S.prune_cache(keep=2, root=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["song3", "song4"]


def test_missing_demucs_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "demucs_available", lambda: False)
    with pytest.raises(S.SeparationUnavailable, match="Source separation not installed: run the installer again"):
        S.run_demucs(tmp_path / "x.wav", tmp_path)


def test_a_full_mix_is_told_from_an_isolated_vocal():
    from vocalmatch.chain import render
    from vocalmatch.preset import Preset, Reverb

    mix, vocal, _ = synthetic_song(seed=5, duration_s=20.0, fs=FS)
    assert S.looks_like_full_mix(mix, FS)[0]
    assert S.looks_like_full_mix(synthetic_song(seed=6, duration_s=20.0, fs=FS, vocal_db=10)[0], FS)[0]
    assert not S.looks_like_full_mix(vocal, FS)[0]
    wet = Preset()
    wet.effects.reverb = Reverb(True, 1.0, 3.2, 60.0, 40.0, 5000.0)  # a long hall: hardly any silence left
    assert not S.looks_like_full_mix(render(wet, vocal, FS), FS)[0]


def test_without_demucs_a_full_mix_is_refused_with_a_clear_message(tmp_path):
    mix, _, _ = synthetic_song(seed=7, duration_s=20.0, fs=FS)
    sf.write(str(tmp_path / "song.mp3"), mix.astype(np.float32), FS, format="MP3")
    sf.write(str(tmp_path / "dry.wav"), synthetic_vocal(seed=8, duration_s=7.0, fs=FS), FS)
    code = ("import sys; import vocalmatch.separation as S; S.demucs_available = lambda: False; "
            "from vocalmatch.cli import main; raise SystemExit(main(sys.argv[1:]))")
    r = subprocess.run([sys.executable, "-c", code, "analyze", "--reference", str(tmp_path / "song.mp3"),
                        "--dry", str(tmp_path / "dry.wav"), "--out", str(tmp_path / "p.json"), "--progress"],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "ERROR Source separation not installed: run the installer again or use an isolated vocal" in r.stdout
    assert "--separate no" in r.stderr
    assert not (tmp_path / "p.json").exists()


def test_the_worker_separates_in_chunks_with_progress(tmp_path):
    """The chunking and stem writing of demucs_worker, with a stand-in model (identity on
    the vocal: the stem must come back whole, at the song's length, seams included)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("demucs")
    from vocalmatch import demucs_worker as W

    class Half(torch.nn.Module):  # vocals = half the mix, the rest = the other half
        samplerate, audio_channels, sources, segment = 44100, 2, ["drums", "bass", "other", "vocals"], 7.8

        def forward(self, x):
            return torch.stack([x * 0.5 / 3, x * 0.5 / 3, x * 0.5 / 3, x * 0.5], dim=1)

    mix, _, _ = synthetic_song(seed=9, duration_s=47.0, fs=FS)
    sf.write(str(tmp_path / "song.wav"), mix, FS, subtype="FLOAT")
    lines = []
    W.separate_file(tmp_path / "song.wav", tmp_path / "out", Half(), progress=lines.append)
    v, fs = S.load_stem(tmp_path / "out" / "vocals.flac")
    r, _ = S.load_stem(tmp_path / "out" / "no_vocals.flac")
    assert fs == FS and len(v) == len(mix)
    # (Demucs adds the mix's mean back to every stem: 0.5 * mean here, ~3e-4)
    assert np.max(np.abs(v - 0.5 * mix)) < 1e-3 and np.max(np.abs(r - 0.5 * mix)) < 1e-3
    assert len(lines) == 3 and lines[-1].startswith("PROGRESS 1.000")



def test_bleed_sections_are_dropped():
    vocal = synthetic_vocal(seed=3, duration_s=10.0, fs=FS)
    rng = np.random.default_rng(0)
    band = rng.standard_normal(len(vocal)) * 0.05  # the accompaniment
    stem = vocal.copy()
    # 4-7 s: an instrumental part; the "vocal stem" holds only bleed of the band there
    a, b = 4 * FS, 7 * FS
    stem[a:b] = 0.3 * band[a:b]
    stem[a - FS // 2 : a] = 0.0  # silences around it, as between sung phrases
    stem[b : b + FS // 2] = 0.0
    sel = S.select_sections(stem[:, None], band[:, None], FS)
    dropped = [s for s in sel.sections if not s[2]]
    assert dropped and any(s[0] >= 3.9 and s[1] <= 7.1 for s in dropped), sel.sections
    assert sel.kept_s < sel.total_s - 2.5


def test_clean_stem_is_kept_whole():
    vocal = synthetic_vocal(seed=4, duration_s=8.0, fs=FS)
    sel = S.select_sections(vocal[:, None], 0.01 * np.random.default_rng(1).standard_normal((len(vocal), 1)), FS)
    assert sel.kept_s > 0.9 * sel.total_s
