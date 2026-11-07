import json

import numpy as np
import soundfile as sf

from vocalmatch.chain import render
from vocalmatch.cli import main
from vocalmatch.preset import Compressor, EQBand, Preset
from vocalmatch.synth import synthetic_vocal

FS = 44100


def test_analyze_and_render_end_to_end(tmp_path):
    dry = synthetic_vocal(seed=9, duration_s=6.0, fs=FS)
    truth = Preset()
    truth.effects.eq.enabled = True
    truth.effects.eq.bands = [EQBand("peak", 2500, 4, 1.0)]
    truth.effects.compressor = Compressor(enabled=True, threshold_db=-20, ratio=3, attack_ms=5, release_ms=100, makeup_db=4)
    sf.write(tmp_path / "dry.wav", dry, FS, subtype="FLOAT")
    sf.write(tmp_path / "ref.flac", render(truth, dry, FS), FS)  # stereo FLAC, 24-bit

    out = tmp_path / "preset.json"
    assert main(["analyze", "--reference", str(tmp_path / "ref.flac"), "--dry", str(tmp_path / "dry.wav"),
                 "--out", str(out), "--render", str(tmp_path / "matched.wav")]) == 0
    d = json.loads(out.read_text())
    assert d["reference"]["filename"] == "ref.flac" and d["effects"]["compressor"]["enabled"]
    assert main(["validate", str(out)]) == 0

    matched, fs = sf.read(tmp_path / "matched.wav")
    assert fs == FS and matched.shape == (len(dry), 2)
    assert main(["render", "--preset", str(out), "--dry", str(tmp_path / "dry.wav"), "--out", str(tmp_path / "r.wav")]) == 0
    np.testing.assert_allclose(sf.read(tmp_path / "r.wav")[0], matched, atol=1e-6)


def test_synth_command(tmp_path):
    assert main(["synth", "--out", str(tmp_path / "s.wav"), "--duration", "2"]) == 0
    x, fs = sf.read(tmp_path / "s.wav")
    assert fs == FS and len(x) == 2 * FS


def test_selftest_command():
    assert main(["selftest"]) == 0


def test_analyze_progress_lines(tmp_path, capsys):
    dry = synthetic_vocal(seed=15, duration_s=4.0, fs=FS)
    sf.write(tmp_path / "dry.wav", dry, FS)
    sf.write(tmp_path / "ref.wav", dry * 0.5, FS)
    assert main(["analyze", "--reference", str(tmp_path / "ref.wav"), "--dry", str(tmp_path / "dry.wav"),
                 "--out", str(tmp_path / "p.json"), "--separate", "no", "--refine", "2", "--passes", "1", "--progress"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PROGRESS ")]
    fractions = [float(line.split()[1]) for line in lines]
    assert len(lines) >= 8 and fractions == sorted(fractions) and fractions[-1] == 1.0 and lines[-1].endswith("done")
