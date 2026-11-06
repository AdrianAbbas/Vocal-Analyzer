# Vocal Match — Project Spec

An audio plugin that analyzes the lead vocal in a reference song and sets up an effect chain so the user's own vocal track sounds processed the same way.

**Repository:** https://github.com/AdrianAbbas/Vocal-Analyzer — all work goes here. This spec lives at the repo root as `SPEC.md`.

## Goals
- The user loads a reference song (WAV/MP3/FLAC) and presses **Analyze**.
- The plugin isolates the reference vocal, estimates the processing (EQ, compression, saturation, de-essing, delay, reverb, doubling/width), and fills in the plugin's own effect chain with those settings.
- Every estimated setting stays visible and editable. The output is a starting point, not a black box.
- Formats: VST3 + AU (macOS) and VST3 (Windows), plus a standalone app for testing.

## Non-goals (v1)
- Pitch correction / tuning (maybe in v2).
- Matching harmonies or backing vocals; lead vocal only.
- Training a big model from scratch. Use DSP analysis plus a small optimizer first.

## Architecture

Build in two phases, all in one repo:

```
/analysis     Python prototype: separation + analysis + parameter estimation
/plugin       JUCE C++ plugin: real-time effect chain + UI
/shared       preset schema (JSON) used by both
/tests        test audio + scripts that compare results
```

### Phase 1 — Python prototype (`/analysis`)
CLI: `python -m vocalmatch analyze --reference song.wav --dry myvocal.wav --out preset.json --render matched.wav`

Pipeline:
1. **Separation:** Demucs (`htdemucs`) pulls out the vocal stem. Cache stems on disk.
2. **Pick the sections to analyze:** keep frames where the vocal is clearly active (RMS gate + voicing detection) and skip bleed-heavy sections.
3. **Estimators** (each one is its own module with a unit test):
   - **EQ:** long-term average spectrum (LTAS) of the reference vs the dry vocal, smoothed to 1/3 octave. Fit up to 6 parametric bands (low shelf, 4 peaks, high shelf) to the difference curve. Limit the gain to ±12 dB.
   - **Compression:** compare crest factor, short-term loudness range, and envelope statistics. Estimate threshold, ratio, attack, release and makeup gain.
   - **Saturation:** estimate harmonic distortion (THD) on sustained voiced frames and map it to drive + mix.
   - **De-esser:** compare the energy share in 5–10 kHz during sibilant frames and set frequency + reduction.
   - **Delay:** autocorrelation of the vocal envelope after phrases end gives time (also snap to the song's tempo if detected), feedback, mix and a high-cut.
   - **Reverb:** measure how fast the tail decays in the gaps after phrases (Schroeder backward integration) to get decay, pre-delay, wet/dry and damping.
   - **Width/doubling:** inter-channel correlation and mid/side ratio give doubler mix and stereo width.
4. **Refinement:** starting from the estimates, run an optimizer (CMA-ES or scipy) over the chain settings. It minimizes a multi-resolution STFT + loudness-statistics distance between the processed dry vocal and the reference vocal. The Python chain (step 5) has to match the C++ chain closely enough for this to carry over.
5. **Reference chain in Python:** the same effects in the same order, implemented in numpy/pedalboard, so the render matches the plugin.
6. Output `preset.json` that follows `/shared/preset.schema.json`.

### Phase 2 — JUCE plugin (`/plugin`)
- CMake + JUCE 8. Effect order: Input gain → EQ → Compressor → Saturation → De-esser → Delay → Reverb → Doubler/Width → Output gain.
- All settings are exposed through `AudioProcessorValueTreeState` (automatable, saved with the session).
- **Load preset** button reads the JSON from Phase 1.
- **Analyze** button (stretch goal for v1): runs the analysis in a background thread. Option A: launch the bundled Python analyzer as a separate process. Option B: export Demucs to ONNX and run the analysis in C++ with ONNX Runtime. Start with A and move to B later.
- UI: one row per effect with a bypass button, a bar showing the estimated vs current value, a "match amount" master knob (0–100% scales every effect's departure from neutral), and A/B against the dry signal.
- Real-time safe: no allocation or locking on the audio thread. Report latency correctly.

## Preset schema (`/shared/preset.schema.json`)
JSON Schema with a version field, info about the reference (filename, tempo, sample rate), a confidence score from 0 to 1 for each effect, and the settings for each effect in real units (Hz, dB, ms, ratio, %).

## Testing
- **Synthetic round-trip:** process a dry vocal with known chain settings, run the analyzer on the result, and check that it recovers the settings within tolerance (e.g. EQ ±1.5 dB per band, reverb decay ±20%). This is the main correctness test.
- **Python vs C++ check:** the C++ chain and the Python chain, fed the same preset and input, must agree within -40 dB of residual.
- **pluginval** at strictness 5+ on the built plugin.
- GitHub Actions: Python tests + plugin build on macOS and Windows.

## Milestones
1. Repo scaffold, preset schema, Python reference chain + render.
2. EQ + compression estimators + synthetic round-trip tests.
3. Reverb, delay, saturation, de-esser and width estimators.
4. Demucs integration + refinement optimizer; listening tests on real songs.
5. JUCE plugin with the full chain + preset loading; Python vs C++ check passing.
6. In-plugin Analyze (separate-process option), UI polish, match-amount knob.
7. (Later) ONNX analysis inside the plugin, pitch-correction estimation.

## Notes for the implementer
- Work milestone by milestone and keep tests passing before moving on.
- Prefer simple, explainable DSP estimators. Add ML only where it measurably improves the round-trip tests or listening results.
- Keep the user's reference songs out of git (`.gitignore` the `/tests/audio/private` folder).
