# Vocal Match

Analyze the lead vocal of a reference song and set up an effect chain so your
own vocal is processed the same way. Full spec: [`SPEC.md`](SPEC.md).

```
/analysis   Python prototype: analysis + parameter estimation + reference chain
/plugin     JUCE plugin (milestone 5)
/shared     preset schema (JSON) + exact DSP definition of the chain (CHAIN.md)
/tests      synthetic round-trip tests, test audio (private/ is git-ignored)
```

## Status

| Milestone | State |
|---|---|
| 1. Scaffold, preset schema, Python reference chain + render | done |
| 2. EQ + compression estimators, synthetic round-trip tests | done |
| 3. Reverb, delay, saturation, de-esser, width estimators | – |
| 4. Demucs separation, refinement optimizer, listening tests | – |
| 5–7. JUCE plugin, in-plugin Analyze, ONNX | – |

Until milestone 4, `--reference` must already be an **isolated vocal stem**:
no source separation runs yet.

## Quick start

```bash
python -m pip install -e "analysis[dev]"
```

```bash
python -m vocalmatch analyze --reference ref_vocal.wav --dry myvocal.wav --out preset.json --render matched.wav
```

```bash
python -m vocalmatch render --preset preset.json --dry myvocal.wav --out out.wav --tail 2
```

```bash
python -m vocalmatch validate preset.json
```

## Tests

```bash
python -m pytest
```

```bash
python tests/roundtrip/report.py --unpaired
```

The first command runs the unit tests and the synthetic round-trip tests.
The second prints a Markdown table of the round-trip results.

The round trip renders a deterministic synthetic vocal through known settings,
runs the estimators on the result, and checks that they recover the settings:

| | tolerance |
|---|---|
| EQ | each 1/3-octave band 100 Hz – 12.5 kHz within ±1.5 dB (broadband offset removed, see below) |
| Compressor | threshold ±2 dB, ratio ±20 %, attack and release within ×/÷ 2, make-up ±1.5 dB, and the rendered output's envelope statistics within 0.25 dB of the reference |

## How the estimators work

**EQ** (`analysis/vocalmatch/estimators/eq.py`):
1. Compute the long-term average spectrum (LTAS) of the reference and the dry vocal over active frames.
   Each frame is first divided by the geometric mean of its 1/3-octave band powers.
   That statistic changes by a constant under any EQ or gain, so compression does not bias the shape.
2. Fit six RBJ bands (low shelf, 4 peaks, high shelf, ±12 dB) so that `Smooth⅓(|H|²·LTAS_dry) ≈ Smooth⅓(LTAS_ref)`.
   Smoothing the model the same way as the data means the 1/3-octave smoothing doesn't flatten narrow bands.
3. The fit is multi-start: greedy band placement plus random restarts, then joint least squares.
   Small penalties keep it readable: an L1 penalty on gains, a penalty on nearby opposite-sign peaks, and merging of duplicate peaks.

Broadband level can't be observed from spectral shape. By convention the estimated EQ is therefore kept close to *level-neutral* on the dry vocal, and loudness goes to the compressor make-up.

**Compressor** (`analysis/vocalmatch/estimators/compressor.py`):
1. Compare only statistics of each signal's level envelope (quantiles of peak level, RMS level, crest factor, local contrast and level changes).
   Nothing relies on sample alignment, because the reference and your vocal are different performances.
2. Get a first threshold/ratio/make-up from a quantile–quantile fit of the two level distributions.
3. Refine with Nelder–Mead: simulate the actual chain compressor on the dry vocal and match the reference statistics.
   The search runs on a decimated detector first, then a short full-rate polish.

Threshold is relative to the level entering the compressor, so it transfers to your own vocal level.

## Known limitations

- The round-trip tests follow the spec: the analyzer gets the very dry signal the reference was made from.
  With a *different* performance (`report.py --unpaired`) the estimates are much less precise.
  EQ differences below ~500 Hz are dominated by the pitch range sung, and compression ratio is confounded by the performance's own dynamics.
  Milestone 4's listening tests are where this gets tuned.
- The knee is fixed at 6 dB during estimation.
- Saturation, de-esser, delay, reverb and doubler are implemented in the chain but not estimated yet.
  They are written disabled with `confidence: 0`.
