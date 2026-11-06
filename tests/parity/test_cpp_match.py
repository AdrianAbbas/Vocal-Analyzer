"""Python vs C++ check (SPEC "Testing"): the plug-in's processor and the Python reference
chain, fed the same preset and input, must agree within -40 dB of residual.

The C++ side is VocalMatchRender (plugin/Tools/RenderMain.cpp): the plug-in's own
AudioProcessor run offline, with the preset loaded through the plug-in's "Load preset" code.
Build it with
    cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release && cmake --build plugin/build --target VocalMatchRender
or point $VOCALMATCH_RENDER at it. Without the binary these tests are skipped, unless
$VOCALMATCH_REQUIRE_CPP=1 (CI), which turns a missing binary into a failure.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocalmatch.chain import render
from vocalmatch.preset import EFFECT_ORDER, Compressor, EQBand, Preset
from vocalmatch.synth import synthetic_vocal

REPO = Path(__file__).resolve().parents[2]
FS = 44100
SPEC_LIMIT_DB = -40.0  # SPEC: residual within -40 dB
# Both chains run in double precision; what remains is float32 audio files and float32
# parameter storage in the plug-in. Anything above this is a real divergence.
TIGHT_LIMIT_DB = -80.0

# Effects the plug-in implements so far (the others are written disabled for this check).
PLUGIN_EFFECTS = {"input_gain", "eq", "compressor", "output_gain"}


def render_binary() -> Path | None:
    env = os.environ.get("VOCALMATCH_RENDER")
    candidates = [Path(env)] if env else []
    candidates += [
        REPO / "plugin/build/VocalMatchRender_artefacts/Release/VocalMatchRender",
        REPO / "plugin/build/VocalMatchRender_artefacts/Release/VocalMatchRender.exe",
        REPO / "plugin/build/VocalMatchRender_artefacts/VocalMatchRender",
    ]
    for c in candidates:
        if c.is_file():
            return c
    if os.environ.get("VOCALMATCH_REQUIRE_CPP") == "1":
        pytest.fail(f"VocalMatchRender not found (looked at {[str(c) for c in candidates]})")
    return None


@pytest.fixture(scope="module")
def binary() -> Path:
    b = render_binary()
    if b is None:
        pytest.skip("VocalMatchRender is not built (see this module's docstring)")
    return b


@pytest.fixture(scope="module")
def dry_file(tmp_path_factory) -> tuple[Path, np.ndarray]:
    x = synthetic_vocal(seed=5, duration_s=6.0, fs=FS)
    stereo = np.stack([x, 0.8 * np.roll(x, 37)], axis=1)  # distinct channels: checks the stereo link
    path = tmp_path_factory.mktemp("parity") / "dry.wav"
    sf.write(str(path), stereo, FS, subtype="FLOAT")
    data, _ = sf.read(str(path), dtype="float64", always_2d=True)  # exactly what C++ reads
    return path, data


def only_plugin_effects(preset: Preset) -> Preset:
    p = copy.deepcopy(preset)
    for name in EFFECT_ORDER:
        if name not in PLUGIN_EFFECTS:
            getattr(p.effects, name).enabled = False
    return p


def example_preset() -> Preset:
    return only_plugin_effects(Preset.load(REPO / "shared/examples/example_preset.json"))


def heavy_preset() -> Preset:
    """Hard knee, fast attack, deep compression, every EQ band, input and output gain."""
    p = Preset()
    e = p.effects
    e.input_gain.enabled, e.input_gain.gain_db = True, 4.5
    e.output_gain.enabled, e.output_gain.gain_db = True, -3.25
    e.eq.enabled = True
    e.eq.bands = [
        EQBand("low_shelf", 180.0, -6.0, 0.6),
        EQBand("peak", 400.0, 4.0, 2.5),
        EQBand("peak", 1200.0, -9.0, 6.0),
        EQBand("peak", 3100.0, 7.5, 0.8),
        EQBand("peak", 7000.0, -3.0, 12.0),
        EQBand("high_shelf", 12000.0, 9.0, 0.9),
    ]
    e.compressor = Compressor(enabled=True, confidence=1.0, threshold_db=-36.0, ratio=12.0, attack_ms=0.3,
                              release_ms=45.0, knee_db=0.0, makeup_db=14.0)
    return p


def bypassed_preset() -> Preset:
    p = heavy_preset()
    for name in PLUGIN_EFFECTS:
        getattr(p.effects, name).enabled = False
    return p


def residual_db(a: np.ndarray, b: np.ndarray) -> float:
    err = np.sqrt(np.mean((a - b) ** 2))
    ref = np.sqrt(np.mean(b**2))
    return float(20 * np.log10(max(err, 1e-30) / ref))


def run_cpp(binary: Path, preset: Preset, dry: Path, out: Path, match: float, block: int, tail: float = 0.0) -> np.ndarray:
    preset_path = out.with_suffix(".json")
    preset.save(preset_path)
    cmd = [str(binary), "--preset", str(preset_path), "--in", str(dry), "--out", str(out),
           "--match", str(match), "--block", str(block), "--tail", str(tail)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    y, fs = sf.read(str(out), dtype="float64", always_2d=True)
    assert fs == FS
    return y


CASES = {
    "example-100%-block512": (example_preset, 100.0, 512),
    "example-60%-block37": (example_preset, 60.0, 37),
    "heavy-100%-block64": (heavy_preset, 100.0, 64),
    "heavy-25%-block1024": (heavy_preset, 25.0, 1024),
    "heavy-0%-block512": (heavy_preset, 0.0, 512),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_cpp_matches_python(binary, dry_file, tmp_path, case):
    make, match, block = CASES[case]
    dry_path, dry = dry_file
    preset = make()
    y_cpp = run_cpp(binary, preset, dry_path, tmp_path / "cpp.wav", match, block)
    y_py = render(preset, dry, FS, match_amount=match / 100.0)
    assert y_cpp.shape == y_py.shape
    r = residual_db(y_cpp, y_py)
    print(f"{case}: residual {r:.1f} dB")
    assert r < SPEC_LIMIT_DB, f"{case}: residual {r:.1f} dB exceeds the spec's {SPEC_LIMIT_DB} dB"
    assert r < TIGHT_LIMIT_DB, f"{case}: residual {r:.1f} dB; the chains diverge (expected below {TIGHT_LIMIT_DB} dB)"


def test_bypassed_is_bit_exact(binary, dry_file, tmp_path):
    dry_path, dry = dry_file
    y = run_cpp(binary, bypassed_preset(), dry_path, tmp_path / "bypass.wav", 100.0, 512)
    np.testing.assert_array_equal(y, dry)


def test_match_amount_zero_is_neutral(binary, dry_file, tmp_path):
    """At 0 % every effect is at neutral: gains 0 dB, EQ flat, ratio 1:1, no make-up."""
    dry_path, dry = dry_file
    y = run_cpp(binary, heavy_preset(), dry_path, tmp_path / "zero.wav", 0.0, 512)
    assert residual_db(y, dry) < -120.0


def test_preset_file_fields_reach_the_plugin(binary, tmp_path):
    """Every parameter the preset sets is a plug-in parameter (no silently dropped keys)."""
    out = subprocess.run([str(binary), "--list-params"], check=True, capture_output=True, text=True).stdout
    ids = {line.split("\t")[0] for line in out.splitlines() if line.strip()}
    expected = {"match_amount", "input_on", "input_gain_db", "output_on", "output_gain_db", "eq_on", "comp_on",
                "comp_threshold_db", "comp_ratio", "comp_attack_ms", "comp_release_ms", "comp_knee_db", "comp_makeup_db"}
    expected |= {f"eq{b}_{k}" for b in range(1, 7) for k in ("on", "type", "freq_hz", "gain_db", "q")}
    assert expected <= ids, sorted(expected - ids)


def test_bad_preset_is_rejected(binary, dry_file, tmp_path):
    dry_path, _ = dry_file
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "effects": {}}))
    r = subprocess.run([str(binary), "--preset", str(bad), "--in", str(dry_path), "--out", str(tmp_path / "x.wav")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "schema_version" in r.stderr
