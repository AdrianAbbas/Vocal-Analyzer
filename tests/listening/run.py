"""Listening tests on real songs (SPEC milestone 4).

The songs are yours and stay out of git: list them in tests/audio/private/manifest.json:

    [{"name": "song-a", "reference": "song-a.mp3", "dry": "my-take-a.wav", "tempo": 96},
     ...]

(paths relative to that folder; "reference" can be the full song when Demucs is installed,
"dry" is your own unprocessed vocal for it). Then

    python tests/listening/run.py [--refine 60] [--seconds 20]

For each song this runs the analyzer like `vocalmatch analyze` (separation, section
selection, estimators, refinement) and writes tests/audio/private/listening/<name>/:

    preset.json           load it in the plug-in
    1-reference-vocal.wav the reference vocal (the separated stem), an excerpt
    2-dry.wav             your dry vocal, the same length
    3-matched.wav         your dry vocal through the preset
    4-matched-50.wav      ... at 50 % match amount
    ab.wav                dry / matched / reference, 4 s each, three rounds
    report.md             the estimates with confidences, the refinement distance, timings,
                          and a rating sheet to fill in while listening

and tests/audio/private/listening/summary.md with one line per song. The ratings are a
person's to give: this script only prepares the material and measures what it can.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from vocalmatch.audio_io import load_audio, write_audio  # noqa: E402
from vocalmatch.chain import render, to_stereo  # noqa: E402
from vocalmatch.pipeline import analyze  # noqa: E402
from vocalmatch.separation import demucs_available, select_sections, separate  # noqa: E402

PRIVATE = ROOT / "tests" / "audio" / "private"


def _excerpt(x: np.ndarray, fs: int, seconds: float) -> np.ndarray:
    """The loudest window of *seconds* (by 1 s RMS), so the excerpt has singing in it."""
    st = to_stereo(x)
    n = int(seconds * fs)
    if st.shape[1] <= n:
        return st.T
    e = np.convolve(np.mean(st**2, axis=0), np.ones(fs) / fs, mode="valid")
    starts = np.arange(0, len(e) - n + fs, fs // 2)
    starts = starts[starts + n <= st.shape[1]]
    best = max(starts, key=lambda s: e[s : s + n - fs + 1].mean() if s + n - fs + 1 <= len(e) else 0)
    return st[:, best : best + n].T


def _norm(x: np.ndarray) -> np.ndarray:
    """Equal loudness (RMS) for fair A/B listening."""
    return x / (np.sqrt(np.mean(x**2)) + 1e-12) * 10 ** (-20 / 20)


def run_one(item: dict, out_root: Path, refine: int, seconds: float) -> str:
    name = item["name"]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dry, fs = load_audio(PRIVATE / item["dry"])
    ref_path = PRIVATE / item["reference"]
    if item.get("is_stem") or not demucs_available():
        reference, _ = load_audio(ref_path, target_fs=fs)
        sep = "none (stem given)" if item.get("is_stem") else "none (Demucs not installed: reference used as a stem)"
    else:
        v, r, cached = separate(ref_path)
        vocals, _ = load_audio(v, target_fs=fs)
        rest, _ = load_audio(r, target_fs=fs)
        sel = select_sections(vocals, rest, fs)
        reference = sel.audio
        sep = f"Demucs {'(cached)' if cached else ''}, kept {sel.kept_s:.0f} of {sel.total_s:.0f} s"
    preset = analyze(reference, dry, fs, reference_name=ref_path.name, tempo_bpm=item.get("tempo"), refine_evals=refine)
    preset.save(out / "preset.json")

    ref_x = _norm(_excerpt(reference, fs, seconds))
    dry_x = _excerpt(dry, fs, seconds)
    matched = _norm(render(preset, dry_x, fs))
    half = _norm(render(preset, dry_x, fs, match_amount=0.5))
    dry_n = _norm(dry_x)
    write_audio(out / "1-reference-vocal.wav", ref_x, fs)
    write_audio(out / "2-dry.wav", dry_n, fs)
    write_audio(out / "3-matched.wav", matched, fs)
    write_audio(out / "4-matched-50.wav", half, fs)
    seg = int(4 * fs)
    gap = np.zeros((int(0.5 * fs), 2))
    ab = []
    for k in range(3):
        for clip in (dry_n, matched, ref_x):
            s = (k * seg) % max(1, len(clip) - seg)
            ab += [clip[s : s + seg], gap]
    write_audio(out / "ab.wav", np.concatenate(ab), fs)

    e = preset.effects
    lines = [f"# {name}", "", f"reference: `{item['reference']}`, dry: `{item['dry']}`, separation: {sep}",
             f"analysis: {time.time() - t0:.0f} s", "", "| effect | on | confidence | settings |", "|---|---|---|---|"]
    for k in ("eq", "compressor", "saturation", "deesser", "delay", "reverb", "doubler"):
        fx = getattr(e, k)
        settings = ", ".join(f"{a}={v}" for a, v in vars(fx).items() if a not in ("enabled", "confidence", "bands"))
        if k == "eq":
            settings = "; ".join(f"{b.type} {b.freq_hz:.0f} Hz {b.gain_db:+.1f} dB Q{b.q:.2f}" for b in fx.bands if b.enabled)
        lines.append(f"| {k} | {'yes' if fx.enabled else 'no'} | {fx.confidence:.2f} | {settings} |")
    ref_d = preset.diagnostics.get("refinement", {})
    if ref_d:
        lines += ["", f"refinement distance: {ref_d['distance_before']} -> {ref_d['distance_after']} "
                      f"({ref_d['evaluations']} evaluations)"]
    lines += ["", "## Ratings (fill in while listening to ab.wav: dry, matched, reference, three times)", "",
              "| question | 1-5 | notes |", "|---|---|---|",
              "| tone (EQ) closer to the reference than the dry? | | |",
              "| dynamics (compression) closer? | | |",
              "| space (reverb, delay) closer? | | |",
              "| width / doubling closer? | | |",
              "| artefacts (harsh saturation, lisping de-esser, pumping)? | | |",
              "| overall: would you start from this preset? | | |"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    on = [k for k in ("eq", "compressor", "saturation", "deesser", "delay", "reverb", "doubler") if getattr(e, k).enabled]
    return f"| {name} | {', '.join(on)} | {ref_d.get('distance_before', '-')} -> {ref_d.get('distance_after', '-')} | {time.time() - t0:.0f} s |"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(PRIVATE / "manifest.json"))
    ap.add_argument("--refine", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        print(f"No {manifest}. Put your songs and dry takes in {PRIVATE} and list them there (see this script's docstring).")
        return 2
    items = json.loads(manifest.read_text())
    out_root = PRIVATE / "listening"
    rows = ["| song | effects found | refinement distance | time |", "|---|---|---|---|"]
    for item in items:
        print(f"== {item['name']}")
        rows.append(run_one(item, out_root, args.refine, args.seconds))
        print(rows[-1])
    (out_root / "summary.md").write_text("# Listening tests\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
