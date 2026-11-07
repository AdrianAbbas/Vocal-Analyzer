# Licences of the code and models Vocal Match uses

Rule for the analyzer: only permissively licensed code and models (MIT, BSD, Apache-2.0, or
similar). No gated models (that need an account or an accepted agreement to download) and no
non-commercial ones. `tests/roundtrip/test_singers.py::test_licences_listed` checks that every
dependency in `analysis/pyproject.toml` is listed here.

## Singer separation (songs with more than one singer)

| What | Where | Licence | Notes |
|---|---|---|---|
| Singer separation: segmentation, singer embedding, clustering, overlap detection | `analysis/vocalmatch/singers.py` | this project's own code | Hand-written DSP (pitch tracking, cepstra, harmonic-series analysis) and clustering. **No pretrained model**: nothing is downloaded, nothing is gated. |
| Region selection | `analysis/vocalmatch/selection.py` | this project's own code | |
| numpy | PyPI `numpy` | BSD-3-Clause | arrays, FFT |
| scipy | PyPI `scipy` | BSD-3-Clause | `scipy.cluster.hierarchy` (Ward clustering), `scipy.fft`, `scipy.ndimage` |

Pretrained speaker-embedding models were considered and left out. Gated models (they need an
account and accepted conditions) and non-commercial ones are excluded by the rule above. Models
trained on speech also distinguish singing voices poorly. If a permissively licensed singing-voice
embedding model is added later, it goes in this table with its weights' licence and source.

## The analyzer's other dependencies

| Package | Licence | Used for |
|---|---|---|
| numpy | BSD-3-Clause | everything numeric |
| scipy | BSD-3-Clause | filters, optimisation, resampling, clustering |
| soundfile | BSD-3-Clause | reading and writing audio; bundles libsndfile (LGPL-2.1, dynamically linked) |
| jsonschema | MIT | checking presets against `shared/preset.schema.json` |
| pytest (development only) | MIT | tests |
| demucs (optional: `vocalmatch[separation]`) | MIT (code and the `htdemucs` weights) | source separation of full songs |
| torch, torchaudio (pulled in by demucs) | BSD-3-Clause | running Demucs |

## The plug-in

| What | Licence | Notes |
|---|---|---|
| JUCE 8 | AGPLv3 or the JUCE commercial licence | The plug-in framework (fetched at build time, not in this repository). Distributing the plug-in needs one of the two. |
| pluginval (CI only) | GPLv3 | validation in CI; not shipped |

## Test material

Synthetic only: `vocalmatch.synth` generates every vocal the tests use, including the duets
(`tests/roundtrip/duet_cases.py`). Real songs never go in the repository: they stay in
`tests/audio/private/`, which git ignores.
