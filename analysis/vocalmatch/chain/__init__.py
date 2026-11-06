"""Python reference chain.

Order (fixed, same as the plugin):
Input gain -> EQ -> Compressor -> Saturation -> De-esser -> Delay -> Reverb
-> Doubler/Width -> Output gain.
"""

from __future__ import annotations

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

__all__ = ["render", "to_stereo", "process_chain"]


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


def render(preset: Preset, audio: np.ndarray, fs: float, tail_s: float = 0.0) -> np.ndarray:
    """Render *audio* through the chain; returns (n [+ tail], 2) like soundfile.

    *tail_s* appends silence before processing so delay/reverb tails ring out.
    """
    x = to_stereo(audio)
    if tail_s > 0:
        x = np.hstack([x, np.zeros((2, int(round(tail_s * fs))))])
    return process_chain(x, fs, preset).T
