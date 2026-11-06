"""End-to-end analysis: reference vocal + dry vocal -> Preset.

Milestone 2 status: EQ and compressor are estimated. Saturation, de-esser,
delay, reverb and doubler/width are written disabled with confidence 0
(estimators arrive in milestone 3). Source separation (Demucs) and the
refinement optimizer arrive in milestone 4; until then *reference* must be
an isolated vocal stem.
"""

from __future__ import annotations

import numpy as np

from .chain import to_stereo
from .chain.effects import process_eq
from .estimators.compressor import estimate_compressor
from .estimators.eq import estimate_eq
from .preset import Preset, Reference

NOT_YET_ESTIMATED = ("saturation", "deesser", "delay", "reverb", "doubler")


def analyze(
    reference: np.ndarray,
    dry: np.ndarray,
    fs: int,
    reference_name: str = "",
    tempo_bpm: float | None = None,
) -> Preset:
    preset = Preset(
        reference=Reference(
            filename=reference_name,
            sample_rate_hz=int(fs),
            tempo_bpm=tempo_bpm,
            duration_s=round(len(to_stereo(reference)[0]) / fs, 3),
        )
    )
    e = preset.effects
    # Gain stages are not estimated: level is matched by the compressor make-up.
    e.input_gain.enabled = True
    e.output_gain.enabled = True

    eq_est = estimate_eq(reference, dry, fs)
    e.eq = eq_est.eq

    dry_eq = process_eq(to_stereo(dry), fs, e.eq) if e.eq.enabled else to_stereo(dry)
    comp_est = estimate_compressor(reference, dry_eq, fs)
    e.compressor = comp_est.compressor

    preset.diagnostics = {
        "eq": eq_est.diagnostics,
        "compressor": comp_est.diagnostics,
        "not_yet_estimated": list(NOT_YET_ESTIMATED),
    }
    return preset
