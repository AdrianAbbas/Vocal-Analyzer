"""Python reference chain.

Order (fixed, same as the plugin):
Input gain -> EQ -> Compressor -> Saturation -> De-esser -> Delay -> Reverb
-> Doubler/Width -> Output gain.
"""

from __future__ import annotations

import copy

import numpy as np

from ..preset import Preset
from .dynamics import process_compressor, process_deesser
from .effects import (
    process_delay,
    process_doubler,
    process_eq,
    process_gain,
    process_reverb,
    process_saturation,
)

__all__ = ["render", "to_stereo", "process_chain", "apply_match_amount"]


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Accept (n,), (n, ch) or (ch, n) with ch in {1, 2}; return (2, n) float64."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return np.vstack([a, a])
    if a.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {a.shape}")
    if a.shape[0] in (1, 2) and a.shape[1] > 2:
        chans = a
    elif a.shape[1] in (1, 2):
        chans = a.T
    else:
        raise ValueError(f"cannot interpret audio of shape {a.shape} as mono/stereo")
    if chans.shape[0] == 1:
        chans = np.vstack([chans[0], chans[0]])
    return np.ascontiguousarray(chans)


def apply_match_amount(preset: Preset, amount: float) -> Preset:
    """The plug-in's "Match amount" knob (shared/CHAIN.md): scale every effect's departure
    from neutral by *amount* (0..1). Returns a copy; 1 leaves the preset unchanged."""
    m = min(max(float(amount), 0.0), 1.0)
    p = copy.deepcopy(preset)
    e = p.effects
    e.input_gain.gain_db *= m
    for band in e.eq.bands:
        band.gain_db *= m
    e.compressor.ratio = 1.0 + (max(e.compressor.ratio, 1.0) - 1.0) * m
    e.compressor.makeup_db *= m
    e.saturation.mix_pct *= m
    e.deesser.reduction_db *= m
    e.delay.mix_pct *= m
    e.reverb.mix_pct *= m
    e.doubler.mix_pct *= m
    e.doubler.width_pct = 100.0 + (e.doubler.width_pct - 100.0) * m
    e.output_gain.gain_db *= m
    return p


def process_chain(x: np.ndarray, fs: float, preset: Preset) -> np.ndarray:
    """Run the full chain on a (2, n) array."""
    e = preset.effects
    y = x
    if e.input_gain.enabled:
        y = process_gain(y, e.input_gain.gain_db)
    if e.eq.enabled:
        y = process_eq(y, fs, e.eq)
    if e.compressor.enabled:
        y = process_compressor(y, fs, e.compressor)
    if e.saturation.enabled:
        y = process_saturation(y, e.saturation)
    if e.deesser.enabled:
        y = process_deesser(y, fs, e.deesser)
    if e.delay.enabled:
        y = process_delay(y, fs, e.delay)
    if e.reverb.enabled:
        y = process_reverb(y, fs, e.reverb)
    if e.doubler.enabled:
        y = process_doubler(y, fs, e.doubler)
    if e.output_gain.enabled:
        y = process_gain(y, e.output_gain.gain_db)
    return y


def render(
    preset: Preset, audio: np.ndarray, fs: float, tail_s: float = 0.0, match_amount: float = 1.0
) -> np.ndarray:
    """Render *audio* through the chain; returns (n [+ tail], 2) like soundfile.

    *tail_s* appends silence before processing so delay/reverb tails ring out.
    *match_amount* (0..1) is the plug-in's Match amount knob.
    """
    if match_amount != 1.0:
        preset = apply_match_amount(preset, match_amount)
    x = to_stereo(audio)
    if tail_s > 0:
        x = np.hstack([x, np.zeros((2, int(round(tail_s * fs))))])
    return process_chain(x, fs, preset).T
