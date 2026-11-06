# vocalmatch (analysis prototype)

The Python half of Vocal Match: reference effect chain, estimators and CLI.
See the repository [README](../README.md) and [SPEC](../SPEC.md).

```
vocalmatch/
  chain/          reference chain (must match the plugin; see shared/CHAIN.md)
  estimators/     eq.py, compressor.py (more in milestone 3)
  features.py     framing, RMS gate, voicing, LTAS, fractional-octave smoothing
  pipeline.py     analyze(reference, dry) -> Preset
  preset.py       dataclasses mirroring shared/preset.schema.json
  synth.py        deterministic synthetic vocal for tests
  cli.py          python -m vocalmatch {analyze,render,synth,validate}
```

The preset schema is read from `../shared/preset.schema.json` (override with
`VOCALMATCH_SCHEMA`), so install in editable mode: `pip install -e "analysis[dev]"`.
