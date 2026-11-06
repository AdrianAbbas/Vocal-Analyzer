"""Print the synthetic round-trip results as Markdown tables.

    python tests/roundtrip/report.py [--out results.md] [--unpaired]

--unpaired adds an informational section where the reference is made from a
*different* synthetic performance than the dry vocal given to the analyzer
(closer to real use; no pass/fail tolerances are applied there).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

import cases as C  # noqa: E402
from vocalmatch.chain import render  # noqa: E402
from vocalmatch.estimators.compressor import estimate_compressor  # noqa: E402
from vocalmatch.estimators.eq import estimate_eq  # noqa: E402
from vocalmatch.preset import Preset  # noqa: E402


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def comp_row(name: str, r: C.CompResult) -> str:
    t, e, er = r.true, r.est, r.errors
    return (
        f"| {name} | {t.threshold_db:.0f} / {e.threshold_db - r.level_shift_db:.1f} ({er['threshold_db']:+.1f}) "
        f"| {t.ratio:.1f} / {e.ratio:.2f} ({er['ratio_rel'] * 100:+.0f}%) "
        f"| {t.attack_ms:.0f} / {e.attack_ms:.1f} (x{er['attack_factor']:.2f}) "
        f"| {t.release_ms:.0f} / {e.release_ms:.0f} (x{er['release_factor']:.2f}) "
        f"| {t.makeup_db:.0f} / {e.makeup_db + r.level_shift_db:.1f} ({er['makeup_db']:+.1f}) "
        f"| {r.output_stats_db:.3f} | {e.confidence:.2f} | {mark(r.ok)} |"
    )


COMP_HEADER = (
    "| case | threshold dB true/est (err) | ratio | attack ms | release ms | make-up dB | out-stats dist dB | conf | result |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--unpaired", action="store_true")
    args = ap.parse_args()
    lines: list[str] = []
    t0 = time.time()
    n_fail = 0

    lines += ["## EQ round trip", "",
              f"Worst 1/3-octave band error, {C.EQ_EVAL_RANGE_HZ[0]:.0f} Hz - {C.EQ_EVAL_RANGE_HZ[1] / 1000:.1f} kHz, "
              f"broadband offset removed (tolerance ±{C.EQ_BAND_TOL_DB} dB).", "",
              "| case | true bands | worst band err dB | offset dB (info) | est. bands | result |", "|---|---|---|---|---|---|"]
    for name in C.EQ_CASES:
        r = C.run_eq_case(name)
        n_fail += not r.ok
        est = [b for b in r.estimated.bands if b.enabled]
        lines.append(f"| {name} | {len(C.EQ_CASES[name])} | {r.max_err_db:.2f} | {r.offset_db:+.2f} | {len(est)} | {mark(r.ok)} |")

    lines += ["", "## Compressor round trip", "",
              f"Tolerances: threshold ±{C.COMP_THRESHOLD_TOL_DB} dB, ratio ±{C.COMP_RATIO_TOL_REL * 100:.0f}%, "
              f"attack/release within x/÷{C.COMP_TIME_TOL_FACTOR:g}, make-up ±{C.COMP_MAKEUP_TOL_DB} dB, "
              f"output envelope-stats distance ≤ {C.OUTPUT_STATS_TOL_DB} dB.", "", COMP_HEADER]
    for name in C.COMP_CASES:
        r = C.run_comp_case(name)
        n_fail += not r.ok
        lines.append(comp_row(name, r))

    lines += ["", "## Combined EQ + compressor (full `analyze` pipeline)", "",
              "Threshold/make-up compared relative to the level entering the compressor "
              "(the estimated EQ's broadband level is a convention, see README).", "",
              "| case | EQ worst band err dB | EQ result | level shift dB |", "|---|---|---|---|"]
    comp_lines = [COMP_HEADER]
    for name in C.COMBINED_CASES:
        eq_r, comp_r, _ = C.run_combined_case(name)
        n_fail += (not eq_r.ok) + (not comp_r.ok)
        lines.append(f"| {name} | {eq_r.max_err_db:.2f} | {mark(eq_r.ok)} | {comp_r.level_shift_db:+.2f} |")
        comp_lines.append(comp_row(name, comp_r))
    lines += [""] + comp_lines

    if args.unpaired:
        lines += ["", "## Informational: unpaired (reference from a different performance)", "",
                  "Reference = chain(synthetic voice seed 3); analyzer's dry = synthetic voice seed 4. "
                  "No tolerances: this shows how far the estimators are from the paired ideal.", ""]
        a = C.synthetic_vocal(seed=3, duration_s=20.0, fs=C.FS)
        b = C.synthetic_vocal(seed=4, duration_s=20.0, fs=C.FS)
        lines += ["| EQ case | worst band err dB (offset removed) |", "|---|---|"]
        for name in ("presence", "single-peak"):
            p = Preset()
            p.effects.eq = C.eq_preset(C.EQ_CASES[name])
            r = C.compare_eq(p.effects.eq, estimate_eq(render(p, a, C.FS), b, C.FS).eq)
            lines.append(f"| {name} | {r.max_err_db:.2f} |")
        lines += ["", COMP_HEADER]
        for name in ("medium 4:1", "slow attack 3:1"):
            p = Preset()
            p.effects.compressor = C.COMP_CASES[name]
            ref = render(p, a, C.FS)
            est = estimate_compressor(ref, b, C.FS).compressor
            q = Preset()
            q.effects.compressor = est
            r = C.CompResult(C.COMP_CASES[name], est, 0.0, C._output_stats_distance(q, ref, b))
            lines.append(comp_row(name, r).replace("PASS", "-").replace("FAIL", "-"))

    lines += ["", f"_{n_fail} failing check(s); {time.time() - t0:.0f} s; numpy {np.__version__}_"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
