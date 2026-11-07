"""Songs with more than one singer, end to end: a synthetic duet of two different singers,
each processed by their own known chain (duet_cases.py). Choosing a singer, by singer
separation or by marking that singer's sections by hand, must recover that singer's chain
within the round-trip tolerances (EQ +-1.5 dB per 1/3-octave band, from the singer's lowest
note up: below it a voice has no energy to measure the EQ by; compressor threshold
+-2 dB, ratio +-20 %, attack / release x/÷ 2, make-up +-1.5 dB, output statistics 0.25 dB),
and both ways of choosing must give the same result."""

import functools

import pytest

from vocalmatch.preset import validate_preset_dict

from cases import compare_eq
from duet_cases import VOICE_B, eq_range_for, run_duet_case, synthetic_duet

pytestmark = pytest.mark.slow


@functools.lru_cache(maxsize=None)
def _singer_map():
    from duet_cases import FS
    from vocalmatch.singers import detect_singers

    return detect_singers(synthetic_duet().mix.T, FS)


@functools.lru_cache(maxsize=None)
def _case(who, mode):
    return run_duet_case(who, mode, singer_map=_singer_map() if mode == "singer" else None)


@pytest.mark.parametrize("mode", ["singer", "regions"])
@pytest.mark.parametrize("who", ["A", "B"])
def test_singer_chain_recovered(who, mode):
    r = _case(who, mode)
    validate_preset_dict(r.preset.to_dict())
    assert r.selection.analysed_s > 15.0
    assert r.ok, r.summary()


@pytest.mark.parametrize("who", ["A", "B"])
def test_region_selection_gives_the_same_result(who):
    a, b = _case(who, "singer"), _case(who, "regions")
    ea, eb = a.preset.effects, b.preset.effects
    same_eq = compare_eq(ea.eq, eb.eq, eq_range_for(VOICE_B) if who == "B" else (100.0, 12500.0))
    assert same_eq.max_err_db <= 1.5, f"EQ estimates differ by {same_eq.max_err_db:.2f} dB"
    assert abs(ea.compressor.threshold_db - eb.compressor.threshold_db) <= 2.0
    assert abs(ea.compressor.ratio / eb.compressor.ratio - 1.0) <= 0.2
    assert abs(ea.compressor.makeup_db - eb.compressor.makeup_db) <= 1.5


def test_the_singers_chains_differ():
    # the two singers were processed differently, and the estimates show it
    a, b = _case("A", "singer").preset.effects, _case("B", "singer").preset.effects
    assert compare_eq(a.eq, b.eq).max_err_db > 6.0
    assert b.compressor.ratio > a.compressor.ratio


def test_cli_one_preset_per_singer(tmp_path):
    """vocalmatch analyze --singer all --per-singer: a preset per singer, each saying what was
    analysed (the plug-in shows it), and a per-singer report."""
    import json
    import subprocess
    import sys

    import soundfile as sf

    from duet_cases import FS

    d = synthetic_duet()
    sf.write(str(tmp_path / "duet.wav"), d.mix.T, FS, subtype="FLOAT")
    sf.write(str(tmp_path / "dry.wav"), d.dry["A"], FS, subtype="FLOAT")
    out = tmp_path / "duet.json"
    r = subprocess.run([sys.executable, "-m", "vocalmatch", "analyze", "--reference", str(tmp_path / "duet.wav"),
                        "--dry", str(tmp_path / "dry.wav"), "--out", str(out), "--separate", "no",
                        "--singer", "all", "--per-singer", "--regions", "0-36", "--refine", "0", "--passes", "1"],
                       capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0, r.stdout + r.stderr
    for who in "AB":
        p = tmp_path / f"duet - Singer {who}.json"
        sel = json.loads(p.read_text())["reference"]["selection"]
        assert sel["singers"] == [who] and sel["regions"] == [[0.0, 36.0]] and 0.9 < sel["singer_confidence"] <= 1.0
        assert 10.0 < sel["analysed_s"] < 20.0  # two sections of that singer inside 0-36 s
        assert f"Singer {who}, singer confidence" in r.stdout
    assert json.loads(out.read_text())["reference"]["selection"]["singers"] == ["A"]  # --out: the first singer
    assert "2 singers found" in r.stdout
