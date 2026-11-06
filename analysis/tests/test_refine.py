"""The refinement optimizer: never worse, and it pulls a perturbed estimate back."""

import copy

from vocalmatch.chain import render
from vocalmatch.preset import EQ, Compressor, EQBand, Preset, Reverb
from vocalmatch.refine import Target, distance, refine
from vocalmatch.synth import synthetic_vocal

FS = 44100


def test_refine_improves_a_perturbed_preset_and_truth_is_a_minimum():
    dry = synthetic_vocal(seed=12, duration_s=6.0, fs=FS)
    truth = Preset()
    truth.effects.eq = EQ(enabled=True, bands=[EQBand("peak", 2500, 4.0, 1.0)])
    truth.effects.compressor = Compressor(True, 1.0, -24.0, 4.0, 5.0, 100.0, 6.0, 6.0)
    truth.effects.reverb = Reverb(True, 1.0, 1.5, 20.0, 20.0, 7000.0)
    ref = render(truth, dry, FS)
    target = Target.of(ref, FS)
    assert distance(ref, target) < 1e-9

    start = copy.deepcopy(truth)
    start.effects.compressor.threshold_db = -16.0
    start.effects.compressor.makeup_db = 2.0
    start.effects.reverb.mix_pct = 35.0
    res = refine(start, ref, dry, FS, max_evals=100, target=target)
    assert res.distance_after < 0.6 * res.distance_before
    c = res.preset.effects.compressor
    assert abs(c.threshold_db + 24.0) < abs(-16.0 + 24.0)
    assert abs(res.preset.effects.reverb.mix_pct - 20.0) < 15.0
    # the input preset is left alone
    assert start.effects.compressor.threshold_db == -16.0
