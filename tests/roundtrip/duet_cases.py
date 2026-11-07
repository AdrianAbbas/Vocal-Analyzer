"""Synthetic duets for the multi-singer tests: two different dry singers, each processed by
its own known (clearly different) chain, singing in alternating sections with one section
where both sing at once. Synthetic material only: real songs stay in tests/audio/private.

    sections:  A | B | A | B | A+B | A | B
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from vocalmatch.chain import render, to_stereo
from vocalmatch.preset import Preset
from vocalmatch.synth import Voice, synthetic_vocal

from cases import COMBINED_CASES, FS, eq_preset

# Singer A: the default synthetic voice. Singer B: a voice an octave higher (as in a duet of a
# low and a high voice), with a shorter vocal tract (formants +20 %), faster vibrato, breathier.
VOICE_A = Voice()
VOICE_B = Voice(pitch_semitones=12.0, formant_scale=1.2, vibrato_scale=1.25, breath=2.0)
CHAIN_A = "pop vocal"  # presence EQ, 4:1 compressor
CHAIN_B = "lofi"       # telephone EQ, 6:1 compressor with +8 dB make-up

# (who, seconds). Each section is its own synthetic performance (starting and ending in a pause).
LAYOUT = (("A", 9.0), ("B", 9.0), ("A", 9.0), ("B", 9.0), ("AB", 6.0), ("A", 9.0), ("B", 9.0))


def chain(name: str) -> Preset:
    bands, c = COMBINED_CASES[name]
    p = Preset()
    p.effects.eq = eq_preset(bands)
    p.effects.compressor = c
    return p


@dataclass
class Duet:
    mix: np.ndarray  # (2, n) the processed vocal stem: chain A (dry A) + chain B (dry B)
    dry: dict  # "A" / "B" -> (n,) that singer's dry track (silent where they do not sing)
    truth: dict  # "A" / "B" -> Preset
    sections: list  # (start_s, end_s, who)

    def regions(self, who: str) -> list[tuple[float, float]]:
        """The sections where only *who* sings (what a user would mark by hand)."""
        return [(a, b) for a, b, w in self.sections if w == who]

    @property
    def overlap_s(self) -> float:
        return sum(b - a for a, b, w in self.sections if w == "AB")


@functools.lru_cache(maxsize=4)
def synthetic_duet(seed: int = 40, voice_b: Voice | None = None) -> Duet:
    total = int(sum(s for _, s in LAYOUT) * FS)
    dry = {"A": np.zeros(total), "B": np.zeros(total)}
    sections, pos = [], 0
    for k, (who, secs) in enumerate(LAYOUT):
        n = int(secs * FS)
        for singer in who:
            voice = VOICE_A if singer == "A" else (voice_b or VOICE_B)
            part = synthetic_vocal(seed=seed + 7 * k + (0 if singer == "A" else 3), duration_s=secs, fs=FS, voice=voice)
            dry[singer][pos : pos + n] += part[:n]
        sections.append((pos / FS, (pos + n) / FS, who))
        pos += n
    truth = {"A": chain(CHAIN_A), "B": chain(CHAIN_B)}
    mix = render(truth["A"], dry["A"], FS) + render(truth["B"], dry["B"], FS)
    return Duet(to_stereo(mix), dry, truth, sections)


# ------------------------------------------------------------------ analysis
@dataclass
class DuetResult:
    who: str
    mode: str  # "singer" or "regions"
    preset: Preset
    selection: object
    eq: object  # cases.EQResult
    comp: object  # cases.CompResult

    @property
    def ok(self) -> bool:
        return self.eq.ok and self.comp.ok

    def summary(self) -> str:
        c = self.comp
        return (f"{self.who} by {self.mode}: EQ worst band {self.eq.max_err_db:.2f} dB, compressor errors "
                f"{ {k: round(v, 3) for k, v in c.errors.items()} }, output stats {c.output_stats_db:.3f} dB, "
                f"analysed {self.selection.analysed_s:.1f} s, failed {[k for k, v in c.checks.items() if not v]}")


def run_duet_case(who: str, mode: str, seed: int = 40, singer_map=None) -> DuetResult:
    """Analyse singer *who* of the duet, chosen by singer detection ("singer") or by marking
    that singer's sections by hand ("regions"), against that singer's own dry track."""
    from vocalmatch.chain.effects import process_eq
    from vocalmatch.estimators.compressor import envelope_stats, stats_distance
    from vocalmatch.pipeline import analyze
    from vocalmatch.selection import select_reference
    from vocalmatch.singers import detect_singers

    from cases import CompResult, _rms_db, compare_eq

    d = synthetic_duet(seed)
    stem = d.mix.T
    if mode == "singer":
        m = singer_map or detect_singers(stem, FS)
        sel = select_reference(stem, FS, singer_map=m, singers=[who])
    else:
        sel = select_reference(stem, FS, regions=d.regions(who))
    # As in the other round trips, the analyser gets the dry signal the reference was made
    # from: this singer's dry track where they sing alone.
    from vocalmatch.selection import join_masked, regions_mask

    dry = join_masked(d.dry[who], regions_mask(d.regions(who), FS, len(d.dry[who])), FS)
    est = analyze(sel.audio, dry, FS, reference_name=f"synthetic duet: singer {who} ({mode})")
    truth = d.truth[who]
    eq_res = compare_eq(truth.effects.eq, est.effects.eq)
    x = to_stereo(dry)
    shift = _rms_db(process_eq(x, FS, est.effects.eq)) - _rms_db(process_eq(x, FS, truth.effects.eq))
    # output statistics: the estimate rendered on the dry vs that singer's processed sections
    ref_alone = to_stereo(render(truth, dry, FS))
    keep = np.abs(dry) > 0
    out = to_stereo(render(est, dry, FS))
    stats = stats_distance(envelope_stats(out[:, keep].T, FS), envelope_stats(ref_alone[:, keep].T, FS))
    return DuetResult(who, mode, est, sel, eq_res, CompResult(truth.effects.compressor, est.effects.compressor, shift, stats))
