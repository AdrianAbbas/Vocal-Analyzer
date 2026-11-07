"""Milestone 6 features of the plug-in, through VocalMatchRender (the plug-in's processor
and its Analyze code, offline): A/B against the dry signal, capture of the input, and
Analyze running the installed analyzer as a separate process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from test_cpp_match import FS, full_preset, render_binary, residual_db

from vocalmatch.chain import render
from vocalmatch.preset import Compressor, EQBand, Preset, validate_preset_dict
from vocalmatch.synth import synthetic_vocal


@pytest.fixture(scope="module")
def binary():
    b = render_binary()
    if b is None:
        pytest.skip("VocalMatchRender is not built")
    return b


@pytest.fixture(scope="module")
def stereo_input(tmp_path_factory):
    x = synthetic_vocal(seed=6, duration_s=4.0, fs=FS)
    stereo = np.stack([x, 0.7 * np.roll(x, 50)], axis=1)
    path = tmp_path_factory.mktemp("features") / "in.wav"
    sf.write(str(path), stereo, FS, subtype="FLOAT")
    return path, sf.read(str(path), dtype="float64", always_2d=True)[0]


def test_ab_dry_is_the_input_bit_exact(binary, stereo_input, tmp_path):
    path, x = stereo_input
    preset = tmp_path / "p.json"
    full_preset().save(preset)
    out = tmp_path / "ab.wav"
    subprocess.run([str(binary), "--preset", str(preset), "--in", str(path), "--out", str(out),
                    "--param", "ab_dry=1"], check=True, capture_output=True)
    y, _ = sf.read(str(out), dtype="float64", always_2d=True)
    np.testing.assert_array_equal(y, x)


def test_capture_records_the_input_mono_mix(binary, stereo_input, tmp_path):
    path, x = stereo_input
    take = tmp_path / "take.wav"
    subprocess.run([str(binary), "--in", str(path), "--out", str(tmp_path / "o.wav"), "--capture", str(take), "--block", "333"],
                   check=True, capture_output=True)
    c, fs = sf.read(str(take), dtype="float64")
    assert fs == FS and c.ndim == 1 and len(c) == len(x)
    np.testing.assert_allclose(c, 0.5 * (x[:, 0] + x[:, 1]), atol=1e-7)


def analyzer_command() -> str | None:
    return os.environ.get("VOCALMATCH_ANALYZER") or shutil.which("vocalmatch")


def test_analyze_through_the_plugin(binary, tmp_path):
    exe = analyzer_command()
    if exe is None:
        if os.environ.get("VOCALMATCH_REQUIRE_CPP") == "1":
            pytest.fail("the vocalmatch command is not installed")
        pytest.skip("the vocalmatch command is not installed")
    dry = synthetic_vocal(seed=14, duration_s=6.0, fs=FS)
    truth = Preset()
    truth.effects.eq.enabled = True
    truth.effects.eq.bands = [EQBand("peak", 3000, 5.0, 1.2)]
    truth.effects.compressor = Compressor(enabled=True, threshold_db=-20, ratio=3, attack_ms=5, release_ms=100, makeup_db=4)
    sf.write(str(tmp_path / "dry.wav"), dry, FS, subtype="FLOAT")
    sf.write(str(tmp_path / "ref.wav"), render(truth, dry, FS), FS, subtype="FLOAT")
    preset_path = tmp_path / "analyzed.json"
    out = tmp_path / "out.wav"
    r = subprocess.run([str(binary), "--analyze", str(tmp_path / "ref.wav"), str(tmp_path / "dry.wav"),
                        "--analyze-out", str(preset_path), "--analyze-refine", "3", "--analyze-passes", "1",
                        "--in", str(tmp_path / "dry.wav"), "--out", str(out)],
                       capture_output=True, text=True, timeout=900, env={**os.environ, "VOCALMATCH_ANALYZER": exe})
    assert r.returncode == 0, r.stdout + r.stderr
    stages = [line for line in r.stdout.splitlines() if line.startswith("  ")]
    assert any("compressor" in s for s in stages) and "done" in stages[-1], r.stdout  # progress arrived
    d = json.loads(preset_path.read_text())
    validate_preset_dict(d)
    assert d["effects"]["compressor"]["enabled"]
    # The plug-in loaded exactly what the analyzer wrote: its render matches Python's.
    y, _ = sf.read(str(out), dtype="float64", always_2d=True)
    y_py = render(Preset.from_dict(d), sf.read(str(tmp_path / "dry.wav"), dtype="float64")[0], FS)
    assert residual_db(y, y_py) < -80.0


def test_missing_analyzer_is_reported(binary, tmp_path):
    sf.write(str(tmp_path / "a.wav"), np.zeros(FS), FS)
    env = {k: v for k, v in os.environ.items()}
    env["VOCALMATCH_ANALYZER"] = str(tmp_path / "nothing-here")
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = str(tmp_path)
    r = subprocess.run([str(binary), "--analyze", str(tmp_path / "a.wav"), str(tmp_path / "a.wav")],
                       capture_output=True, text=True, env=env)
    if "not installed" not in r.stderr:  # a system-wide copy (e.g. /usr/local/bin) may exist on this machine
        pytest.skip("an analyzer is installed system-wide here")
    assert r.returncode != 0
