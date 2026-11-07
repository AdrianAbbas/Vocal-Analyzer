"""Songs with more than one singer: region selection, singer separation and the CLI around
them (fast tests; the chain recovery per singer is test_duet_roundtrip.py)."""

import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from vocalmatch.chain import render
from vocalmatch.selection import join_masked, parse_regions, regions_mask, select_reference
from vocalmatch.singers import detect_singers
from vocalmatch.synth import synthetic_vocal

from duet_cases import FS, VOICE_B, synthetic_duet


# ------------------------------------------------------------------ regions
def test_parse_regions():
    assert parse_regions("12.5-30, 1:05-1:20.5") == [(12.5, 30.0), (65.0, 80.5)]
    assert parse_regions("10-20,15-25;40-41") == [(10.0, 25.0), (40.0, 41.0)]  # merged, sorted
    for bad in ("", "10", "20-10", "a-b", "1:2:3:4-5"):
        with pytest.raises(ValueError):
            parse_regions(bad)


def test_join_masked_keeps_only_the_regions():
    x = np.arange(1000, dtype=float)
    m = regions_mask([(0.1, 0.2), (0.5, 0.6)], 1000, 1000)
    y = join_masked(x, m, 1000)
    assert len(y) == 200
    assert y[50] == 150 and y[150] == 550  # inside, away from the fades


def test_select_reference_by_regions():
    d = synthetic_duet()
    sel = select_reference(d.mix.T, FS, regions=d.regions("A"))
    want = sum(b - a for a, b in d.regions("A"))
    assert abs(sel.analysed_s - want) < 0.01
    assert sel.to_dict()["regions"][0] == [0.0, 9.0]


# ------------------------------------------------------------------ singers
def _accuracy(m, d):
    """Share of the assigned solo frames given to the right singer, and the overlap found
    inside the section where both sing."""
    t = np.arange(len(m.labels)) * 0.05
    ok = total = 0
    overlap = 0.0
    for a, b, who in d.sections:
        sel = (t >= a) & (t < b)
        if who == "AB":
            overlap += (m.labels[sel] == -2).sum() * 0.05
            continue
        want = 0 if who == "A" else 1
        ok += (m.labels[sel] == want).sum()
        total += (m.labels[sel] >= 0).sum()
    return ok / max(total, 1), overlap


def test_two_singers_are_separated():
    d = synthetic_duet()
    m = detect_singers(d.mix.T, FS)
    assert [s.id for s in m.singers] == ["A", "B"], m.message
    assert not m.unsure and m.confidence > 0.8, m.message
    acc, overlap_found = _accuracy(m, d)
    assert acc > 0.97
    # each singer has most of their solo singing
    for s in m.singers:
        assert s.seconds > 18.0
    # the overlap is found and reported (the section where both sing is 6 s; both voices
    # sound together in about 4 s of it)
    assert overlap_found >= 2.0 and m.overlap_s >= 2.0
    assert "Overlap excluded" in m.message
    # previews lie inside that singer's sections
    for s, who in zip(m.singers, "AB"):
        assert any(a <= s.preview[0] and s.preview[1] <= b + 0.5 for a, b in d.regions(who))


@pytest.mark.parametrize("seed,voice", [(60, None), (61, VOICE_B)])
def test_one_singer_stays_one(seed, voice):
    from duet_cases import chain

    x = render(chain("lofi" if voice else "pop vocal"), synthetic_vocal(seed=seed, duration_s=60, fs=FS, voice=voice), FS)
    m = detect_singers(x, FS)
    assert len(m.singers) == 1, m.message
    assert not m.unsure, m.message
    assert m.message.startswith("One singer")


def test_unsure_detection_says_so():
    # In this duet the clusters are only just apart: the result says it is unsure and points
    # to region selection.
    d = synthetic_duet(43)
    m = detect_singers(d.mix.T, FS)
    assert m.unsure
    assert "unsure" in m.message and "region selection" in m.message


def test_singer_selection_excludes_the_overlap():
    d = synthetic_duet()
    m = detect_singers(d.mix.T, FS)
    sel = select_reference(d.mix.T, FS, singer_map=m, singers=["A"])
    info = sel.to_dict()
    assert info["singers"] == ["A"] and info["overlap_excluded_s"] >= 2.0
    assert 18.0 < info["analysed_s"] < 30.0
    # nothing of the overlap section beyond its edges
    mask = m.mask(["A"], FS, d.mix.shape[1])
    a, b = [(a, b) for a, b, w in d.sections if w == "AB"][0]
    inside = mask[int((a + 1.0) * FS) : int((b - 1.0) * FS)]
    assert inside.mean() < 0.6


def test_singer_map_json():
    d = synthetic_duet()
    m = detect_singers(d.mix.T, FS)
    j = m.to_dict()
    assert j["singers"][0]["id"] == "A" and len(j["activity"]) == len(m.labels)
    assert {w for _, _, w in j["segments"]} <= {"A", "B", "overlap"}
    json.dumps(j)


# --------------------------------------------------------------------- CLI
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "vocalmatch", *args], capture_output=True, text=True)


def test_cli_singers(tmp_path):
    d = synthetic_duet()
    song = tmp_path / "duet.wav"
    sf.write(str(song), d.mix.T, FS, subtype="FLOAT")
    out = tmp_path / "s.json"
    r = _cli("singers", "--reference", str(song), "--json", str(out), "--separate", "no", "--progress")
    assert r.returncode == 0, r.stderr
    j = json.loads(out.read_text())
    assert j["version"] == 1 and [s["id"] for s in j["singers"]] == ["A", "B"]
    assert "PROGRESS 1.000 done" in r.stdout and "Singer B" in r.stdout


def test_cli_rejects_bad_selection(tmp_path):
    x = synthetic_vocal(seed=2, duration_s=4.0, fs=FS)
    sf.write(str(tmp_path / "a.wav"), x, FS)
    base = ["analyze", "--reference", str(tmp_path / "a.wav"), "--dry", str(tmp_path / "a.wav"),
            "--out", str(tmp_path / "p.json"), "--separate", "no"]
    r = _cli(*base, "--regions", "20-10")
    assert r.returncode == 2 and "--regions" in r.stderr
    r = _cli(*base, "--per-singer")
    assert r.returncode == 2 and "--per-singer needs --singer" in r.stderr
    r = _cli(*base, "--regions", "100-120")
    assert r.returncode == 2 and "choose more of the song" in r.stderr
    r = _cli(*base, "--singer", "C")
    assert r.returncode == 2 and "no singer 'C'" in r.stderr


def test_licences_listed():
    """docs/LICENSES.md names every dependency of the analyzer (and its licence)."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    text = (repo / "docs/LICENSES.md").read_text(encoding="utf-8").lower()
    toml = (repo / "analysis/pyproject.toml").read_text(encoding="utf-8")
    project = toml[toml.index("[project]") :]  # not the build system
    deps = re.findall(r'"([A-Za-z0-9_.-]+)\s*[<>=!~]', project)
    assert deps
    missing = [d for d in deps if d.lower() not in text]
    assert not missing, f"not in docs/LICENSES.md: {missing}"
