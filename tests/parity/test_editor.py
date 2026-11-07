"""The plug-in's editor and Analyze panel through VocalMatchRender: the layout fits from the
smallest to the largest window (no component outside its parent or overlapping another,
on both pages), the window size is saved with the song, and the reference preview plays
through the output."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from test_cpp_match import FS, REPO, render_binary

sys.path.insert(0, str(REPO / "tests/roundtrip"))


@pytest.fixture(scope="module")
def binary():
    b = render_binary()
    if b is None:
        pytest.skip("VocalMatchRender is not built")
    return b


def _run(binary, *args, **kw):
    cmd = [str(binary), *map(str, args)]
    # The editor needs a display on Linux: use a virtual one when there is none.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", *cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kw)


@pytest.fixture(scope="module")
def duet(tmp_path_factory):
    """A synthetic duet on disk and its singer scan (from the analyzer's own code)."""
    from duet_cases import synthetic_duet

    from vocalmatch.singers import detect_singers

    d = synthetic_duet()
    folder = tmp_path_factory.mktemp("duet")
    song = folder / "duet.wav"
    sf.write(str(song), d.mix.T, FS, subtype="FLOAT")
    scan = folder / "duet.singers.json"
    scan.write_text(json.dumps({"version": 1, "reference": str(song), "separated": False,
                                **detect_singers(d.mix.T, FS).to_dict()}))
    return song, scan


@pytest.mark.parametrize("size", ["900x580", "1040x660", "1100x700", "1440x900", "2080x1320", "1200x580", "900x900"])
def test_layout_fits(binary, duet, size):
    song, scan = duet
    r = _run(binary, "--preset", REPO / "shared/examples/example_preset.json", "--reference", song, "--scan", scan,
             "--regions", "10-20,30-35", "--singers", "A", "--size", size, "--check-layout")
    assert r.returncode == 0, r.stdout + r.stderr
    w, h = size.split("x")
    assert f"layout {w} x {h} analyze: ok" in r.stdout and f"layout {w} x {h} chain: ok" in r.stdout, r.stdout


def test_layout_with_collapsed_cards(binary):
    r = _run(binary, "--collapse", "eq,compressor,reverb", "--size", "900x580", "--check-layout")
    assert r.returncode == 0, r.stdout + r.stderr


def test_default_size_fits_a_13_inch_laptop_and_is_saved(binary, tmp_path):
    r = _run(binary, "--snapshot", tmp_path / "default.png")
    assert r.returncode == 0, r.stderr
    assert "(1040 x 660)" in r.stdout  # within 1100 x 700 points
    r = _run(binary, "--size", "1300x800", "--page", "chain", "--snapshot", tmp_path / "big.png")
    assert r.returncode == 0, r.stderr
    assert "saved editor size 1300 x 800, page 1" in r.stdout
    r = _run(binary, "--size", "400x300", "--snapshot", tmp_path / "small.png")
    assert "(900 x 580)" in r.stdout  # clamped to the minimum


def test_preview_plays_the_song_through_the_output(binary, tmp_path):
    rng = np.random.default_rng(3)
    song = 0.3 * rng.standard_normal((3 * FS, 2))
    sf.write(str(tmp_path / "song.wav"), song, FS, subtype="FLOAT")
    silence = np.zeros((2 * FS, 2))
    sf.write(str(tmp_path / "in.wav"), silence, FS, subtype="FLOAT")
    # input gain off etc. do not matter: the input is silent, the preview is added after the chain
    r = _run(binary, "--in", tmp_path / "in.wav", "--out", tmp_path / "out.wav", "--preview", tmp_path / "song.wav", "1.0", "2.5")
    assert r.returncode == 0, r.stderr
    y, _ = sf.read(str(tmp_path / "out.wav"), dtype="float64", always_2d=True)
    n = int(1.5 * FS)
    np.testing.assert_allclose(y[:n], song[FS : FS + n], atol=1e-6)
    assert np.max(np.abs(y[n + 10 :])) < 1e-6  # stops at the end of the span
