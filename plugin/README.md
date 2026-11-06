# Vocal Match plug-in (JUCE 8)

An AU (`aufx Vcmt Wnwd`, "Windward Audio: Vocal Match") and VST3 effect, bundle ID
`com.windwardaudio.vocalmatch`. It implements the chain in [`/shared/CHAIN.md`](../shared/CHAIN.md),
the same algorithms as the Python reference chain (input gain → EQ → compressor → saturation →
de-esser → delay → reverb → doubler/width → output gain), and loads the analyzer's `preset.json`
([`/shared/preset.schema.json`](../shared/preset.schema.json)).

```
Source/DSP/      the chain, plain C++ (no JUCE): Biquad.h, Chain.h
Source/          processor, parameters (APVTS), preset.json reader, editor
Tools/           VocalMatchRender: the processor offline, for the Python-vs-C++ check
```

- Every setting is an `AudioProcessorValueTreeState` parameter: automatable, saved with the song.
- **Load preset...** (or drop a `.json` on the window) sets every parameter from the file as
  host-recorded gestures; the estimates stay marked on the knobs (orange dots) with each
  effect's confidence.
- **Match amount** scales every effect's departure from neutral (CHAIN.md); **On** is each effect's bypass.
- Real-time safe: the audio thread reads parameter atomics and recomputes coefficients only on
  change; buffers are allocated in `prepareToPlay`. No look-ahead, latency 0.

## Build

```bash
cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build plugin/build --parallel
```

Pass `-DJUCE_PATH=/path/to/JUCE` to use a local JUCE 8.0.16 instead of fetching it.
On macOS the build is universal (arm64 + x86_64, macOS 10.15+).

## Python vs C++ check

```bash
python -m pytest tests/parity -v -s
```

It renders the same input and presets through `VocalMatchRender` and the Python chain and
requires the residual below -40 dB (the spec) and below -80 dB (both run in double precision,
so anything above that is a real divergence). Currently about -145 dB for every effect.

## Package and install (macOS)

`scripts/package-mac.sh` signs the bundles ad hoc and zips them with the analyzer and
`install-mac.sh` (the TAPEWORN FX setup from AdrianAbbas/LIMINAL); CI installs that zip as a
quarantined download and runs auval (arm64 and x86_64) and pluginval on the installed copies.
