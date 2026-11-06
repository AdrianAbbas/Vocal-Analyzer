"""Separation cache and section selection, with a stand-in for Demucs (the real model is
too large for CI; tests/listening runs it on real songs)."""

import numpy as np
import soundfile as sf

from vocalmatch import separation as S
from vocalmatch.synth import synthetic_vocal

FS = 44100


def _fake_separator(calls):
    def run(song, out_dir, model):
        calls.append(song)
        x, fs = sf.read(str(song), always_2d=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / "vocals.wav"), x * 0.9, fs)
        sf.write(str(out_dir / "no_vocals.wav"), x * 0.1, fs)
    return run


def test_stems_are_cached_next_to_the_song_and_redone_when_it_changes(tmp_path):
    song = tmp_path / "song.wav"
    sf.write(str(song), synthetic_vocal(seed=1, duration_s=2.0, fs=FS), FS)
    calls = []
    v, r, cached = S.separate(song, separator=_fake_separator(calls))
    assert not cached and v.is_file() and r.is_file() and v.parent == tmp_path / "song.wav.stems" / S.MODEL
    v2, _, cached2 = S.separate(song, separator=_fake_separator(calls))
    assert cached2 and v2 == v and len(calls) == 1
    sf.write(str(song), synthetic_vocal(seed=2, duration_s=2.0, fs=FS), FS)  # edited song
    _, _, cached3 = S.separate(song, separator=_fake_separator(calls))
    assert not cached3 and len(calls) == 2


def test_missing_demucs_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "demucs_available", lambda: False)
    try:
        S.run_demucs(tmp_path / "x.wav", tmp_path)
    except S.SeparationUnavailable as exc:
        assert "vocalmatch[separation]" in str(exc)
    else:
        raise AssertionError("expected SeparationUnavailable")


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
