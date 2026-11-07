# Vocal Match

Analyze the lead vocal of a reference song and set up an effect chain so your
own vocal is processed the same way. Full spec: [`SPEC.md`](SPEC.md).

```
/analysis   Python prototype: analysis + parameter estimation + reference chain
/plugin     JUCE plug-in: AU + VST3 effect, loads preset.json (see plugin/README.md)
/shared     preset schema (JSON) + exact DSP definition of the chain (CHAIN.md)
/tests      synthetic round-trip tests, test audio (private/ is git-ignored)
```

## Status

| Milestone | State |
|---|---|
| 1. Scaffold, preset schema, Python reference chain + render | done |
| 2. EQ + compression estimators, synthetic round-trip tests | done |
| 5 (early). JUCE plug-in: AU + VST3, the whole chain, Load preset, Match amount, Python vs C++ check | done |
| 3. Saturation, de-esser, delay, reverb, doubler/width estimators (+ each effect in the plug-in) | done |
| 4. Demucs separation + section selection, refinement optimizer, listening-test tool | done (listening itself: yours, see below) |
| 6. In-plugin Analyze (separate process), capture of your take, A/B against dry, meters, resizable window | done |
| 7a. Pitch-correction estimation (how hard the reference is tuned, retune speed, key) | done |
| 7b. ONNX analysis inside the plug-in | not started: see below |
| Songs with more than one singer: region selection, singer separation, a preset per singer; compact resizable window | done |

The plug-in was moved ahead of milestones 3–4 so it can be tried in GarageBand early;
every effect is in the Python chain and the plug-in alike, checked sample by sample.

## Install on macOS (plug-in + analyzer)

Download **`VocalMatch-macOS-Universal`** from the latest successful [CI run](https://github.com/AdrianAbbas/vocal-analyzer/actions/workflows/ci.yml?query=is%3Asuccess).
The run's summary page has the direct link. Quit GarageBand, unzip it, then follow `INSTALL.txt`:

```bash
bash install-mac.sh
```

The installer copies `Vocal Match.component` to `~/Library/Audio/Plug-Ins/Components` and
`Vocal Match.vst3` to `~/Library/Audio/Plug-Ins/VST3`, removes the browser's quarantine flag,
signs both ad hoc, resets the Audio Unit registry, runs `auval`, then installs the analyzer
(below). In GarageBand: Smart Controls (B) > Plug-ins > an empty slot > Audio Units >
Windward Audio > Vocal Match. Then either **Analyze** in the plug-in (below) or **Load preset...**
with a preset made in Terminal.

### Analyze inside the plug-in

1. **Reference song...**: the song whose vocal sound you want (a full song needs Demucs:
   `bash install.sh --with-separation`; otherwise a vocal stem).
2. **Capture**, then play your vocal track: the plug-in records its input (your dry take, up to
   5 minutes; Shift-click starts over). Or pick *Dry vocal: a file...*.
3. **Analyze**: the analyzer runs as a separate program (progress bar, Cancel); its preset loads
   when it is done and is kept in `~/Music/Vocal Match/Presets`.

GarageBand runs plug-ins in a sandbox, which may not allow starting another program. Then Analyze
says so and copies the exact Terminal command; run it, then **Load preset...**. Logic Pro and most
other hosts allow it. **A/B: dry** switches to your untouched input (20 ms crossfade; the effects keep
running), **Reset to estimates** undoes your edits, double-click a knob to return it to its estimate,
and the compressor and de-esser show their gain reduction.

**The window** opens at 1040 × 660 points (fits a 13" laptop with the menu bar and Dock) and
resizes from 900 × 580 to 2080 × 1320 with the host's resize corner or the plug-in's own; the
size is saved with the song. The header is always visible: **Analyze | Chain** tabs, Load preset,
Reset, A/B, the status line, IN/OUT meters and **Match** amount. **Chain** shows every effect as a
card (click a card's title to fold it; the others grow). Each knob has its value below it: click
the value to type one.

### Songs with more than one singer

On a duet the vocal stem holds every singer, so analysing all of it averages their processing.
On the **Analyze** page:

- **Regions**: drag on the song's waveform to mark the sections to analyse (several allowed).
  Drag a region's edge to change it, right-click (or Alt-click) to remove it, double-click or
  **Play** to hear it. Only the marked sections are analysed.
- **Find singers** separates the singers automatically: they get a colour each on the singer lane
  under the waveform (hatched: where they sing at once). **Preview** plays a few seconds of each;
  tick who to match. Sections where singers overlap are always left out (the panel says how much).
  When the detection is unsure it says so: mark the sections with regions instead.
- Tick several singers to analyse them together, or tick **One preset per singer** to get a
  preset each (`<name> - Singer A.json`, ...; the first one loads). The status line shows what
  each preset was made from: singer, seconds analysed, overlap excluded, singer confidence.

Preview plays the reference through this track's output: the track must be playing or monitored
for the host to run the plug-in. From Terminal:

```bash
vocalmatch singers --reference song.wav --json song.singers.json
```

```bash
vocalmatch analyze --reference song.wav --dry my_vocal.wav --out preset.json --singer A
```

```bash
vocalmatch analyze --reference song.wav --dry my_vocal.wav --out preset.json --regions 0:12-0:41,1:30-2:05
```

`--singer A,B` analyses both together; `--singer all --per-singer` writes one preset per singer.
`--regions` and `--singer` combine (that singer, within the regions). How the singer separation
works: `analysis/vocalmatch/singers.py` (own DSP code, no pretrained model; licences in
[`docs/LICENSES.md`](docs/LICENSES.md)).

## Install the analyzer alone (Linux, or macOS without the plug-in)

Download **`VocalMatch-Analyzer`** from the same CI run, unzip it, then:

```bash
bash install.sh
```

The installer:
- finds Python 3.10 or newer;
- installs the `vocalmatch` command into a private environment and links it into `~/.local/bin`;
- runs `vocalmatch selftest` and prints PASS or FAIL.

CI runs exactly this on every push before it publishes the zips: on macOS a quarantined download is installed with `install-mac.sh`,
validated with `auval` (arm64 and x86_64 under Rosetta) and pluginval (strictness 5), the analyzer's preset is rendered by the plug-in, and everything is uninstalled; on Linux the analyzer package is installed, used and uninstalled.
Tagged versions (`v*`) are also published as a GitHub Release, which needs no GitHub sign-in:
`https://github.com/AdrianAbbas/vocal-analyzer/releases/latest/download/VocalMatch-macOS-Universal.zip`
(and `VocalMatch-Analyzer.zip`).

## Develop

```bash
python -m pip install -e "analysis[dev]"
```

```bash
python -m vocalmatch analyze --reference song.mp3 --dry myvocal.wav --out preset.json --render matched.wav
```

`--reference` is the full song when Demucs is installed (`pip install -e "analysis[separation]"`,
or `bash install.sh --with-separation`): the vocal is separated (stems cached in `song.mp3.stems/`)
and only the sections where it is clearly present are analysed. Without Demucs, or with
`--separate no`, give an isolated vocal stem. `--refine N` sets the refinement budget
(default 60 whole-chain evaluations, about a minute; 0 = off). Long material is analysed on
45 s of excerpts.

```bash
python -m vocalmatch render --preset preset.json --dry myvocal.wav --out out.wav --tail 2 --match 100
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

The first command runs the unit tests, the synthetic round-trip tests, and the Python-vs-C++
check (`tests/parity`, skipped unless the plug-in's `VocalMatchRender` is built: see `plugin/README.md`).
The second prints a Markdown table of the round-trip results.

The round trip renders a deterministic synthetic vocal through known settings,
runs the estimators on the result, and checks that they recover the settings:

| | tolerance |
|---|---|
| EQ | each 1/3-octave band 100 Hz – 12.5 kHz within ±1.5 dB (broadband offset removed, see below) |
| Compressor | threshold ±2 dB, ratio ±20 %, attack and release within ×/÷ 2, make-up ±1.5 dB, and the rendered output's envelope statistics within 0.25 dB of the reference |
| Delay | time ±1 ms, feedback ±5, mix ±2 (percentage points), high-cut ×/÷ 1.25 |
| Reverb | decay ±20 %, predelay ±15 ms, mix ±3, damping ×/÷ 1.5 |
| Doubler / width | mix ±5, width ±10 |
| Saturation | drive ±3 dB, mix ±10 |
| De-esser | frequency ±15 %, threshold ±3 dB, reduction ±2 dB |
| Whole chain (all eight effects at once, with refinement) | refinement lowers the distance by 30 %; EQ ±2.5 dB per band; compressor threshold ±2 dB, make-up ±1.5 dB, ratio ±30 %; delay time ±1 ms, mix ±3; reverb decay ±25 %, mix ±4; doubler mix ±5, width ±15; de-esser found |
| Duet (two singers, two chains; `test_duet_roundtrip.py`) | each singer's EQ and compressor within the EQ / compressor tolerances above (EQ bands from the singer's lowest note up: below it there is nothing to measure by), chosen by singer separation and by regions; both choices agree |

The duet is synthetic (`tests/roundtrip/duet_cases.py`): two different synthetic singers (the second
an octave higher, with shorter vocal tract, faster vibrato, more breath), each through its own chain
(presence EQ + 4:1, telephone EQ + 6:1), in alternating sections and one section where both sing.
Real songs never go in the repository: put them in `tests/audio/private/` (git-ignored).
`tests/parity/test_editor.py` checks the window's layout at sizes from 900 × 580 to 2080 × 1320
(nothing outside its parent or overlapping, on both pages), that the size is saved, and the preview.

The milestone-3 cases put each effect behind a realistic EQ + compressor (`tests/roundtrip/cases.py`).
The milestone-3 and whole-chain round trips take minutes (`-m slow`); CI runs them on one Linux job.

The **Python vs C++ check** (`tests/parity`) renders the same presets through the plug-in's own processor
(`VocalMatchRender`) and the Python chain: every effect, edge cases included (half-sample delay times,
odd predelays, clipping drive, match amount, odd block sizes). Residual about −145 dB (spec: −40 dB).

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

The milestone-3 estimators all follow the compressor's recipe: level-independent features on the
reference, the chain's own effect simulated on (the dry vocal through the estimates of everything before
it), and the settings fitted until the features match. Nothing needs sample alignment.

- **Delay** (`delay.py`): an echo is a sharp autocorrelation peak at its lag that the dry vocal lacks, in
  at least two of four bands, *throughout* the song (a sung note's own pitch periodicity only shows in a
  few places); the reverb's early reflections (its signature: predelay + its delay lines) are rejected.
  Feedback, mix and high-cut from the echo train's band autocorrelations at D, 2D, 3D. With `--tempo`
  the time snaps to a note value within 2 %.
- **Reverb** (`reverb.py`): the fastest level drops (L(t + τ) − L(t), τ = 30 ms … 1 s, lowest quantiles,
  four bands): without reverb the level falls into the noise after a phrase, with it only as fast as the
  tail decays. One FDN render per (decay, damping) serves every predelay and mix.
- **Doubler / width** (`width.py`): side-to-mid energy over loud frames (doubler voices) and tail frames
  (the reverb's side, scaled by the width), and the left-right correlation. A mono reference has none.
- **Saturation** (`saturation.py`): crest factor and high-frequency tilt in level-ranked frames (tanh
  flattens peaks and adds harmonics only when loud). Estimated in every pass, but switched on only by
  the refinement's whole-chain model selection: mild saturation and small compressor-estimate errors look
  alike to a single-effect fit.
- **De-esser** (`deesser.py`): the shapes (relative to the median) of the high-band level and tilt
  distributions; the de-esser caps their top, an EQ only shifts them. The effects after it enter as an
  energy kernel of their impulse response. The threshold may not go below the median high-band level
  (that would be a static high cut, the EQ's job).

They run as a **coordinate descent** over the chain, two passes (`pipeline.py`): time-based effects, EQ
(against the dry vocal through those, which commute with it; the de-esser's spectral change handed over
as a correction), compressor (scored after the effects that follow it), saturation, de-esser.

**Refinement** (`refine.py`): a block-coordinate Nelder–Mead over each effect's settings against one
alignment-free distance of the *whole-chain render* to the reference: multi-resolution (256/1024/4096)
1/3-octave spectral statistics, loudness statistics, and every estimator's features. First it tries
switching on saturation and a de-esser that the estimators left off; it keeps them only when the
distance drops by 3 % and 0.1. A result is kept only if it lowers the distance.

**Separation** (`separation.py`): Demucs `htdemucs` (`--two-stems vocals`), stems cached next to the song
with its SHA-256. Sections (cut at silences) are kept when mostly voiced and the vocal stem is not mostly
bleed (median vocal-to-accompaniment ratio ≥ −15 dB), each with up to 2 s of the silence after it (tails).

**Pitch correction** (`estimators/pitch.py`, milestone 7): a YIN pitch track of both vocals, split into
notes. A corrector takes away the singer's own intonation: how far notes land from the semitone grid and
how much they move (vibrato, drift) around their centre. The *amount* is how much of the dry vocal's
intonation the reference lacks (your singing is the baseline), the *retune* speed comes from how fast the
reference settles onto each note (fast / medium / slow relative to you, and an approximate time), plus
the key and tuning offset. It is written to `preset.json` as `pitch_correction` and shown in the plug-in;
nothing is applied (the spec keeps pitch correction out of v1): set your own tuner to match.

**ONNX analysis inside the plug-in (7b)** is not started. It means exporting Demucs to ONNX and porting the
whole analyzer (eight estimators and the optimizer) to C++ with ONNX Runtime in the plug-in: a large
project. Its main benefit would be hosts whose sandbox stops the plug-in from starting the analyzer (see
"Analyze inside the plug-in"); try Analyze in your GarageBand first, since that decides whether it is needed.

## Listening tests (real songs)

Put your songs and dry takes in `tests/audio/private/` (git-ignored), list them in
`tests/audio/private/manifest.json` (format in `tests/listening/run.py`), then:

```bash
python tests/listening/run.py
```

For each song it writes the preset, loudness-matched excerpts of the reference vocal, your dry vocal and
the match (also at 50 %), an A/B file and a report with every estimate, its confidence and a rating
sheet. The listening is yours to do: nothing in CI has real songs, and the synthetic round trips are
the automated measure.

## Known limitations

- The round-trip tests follow the spec: the analyzer gets the very dry signal the reference was made from.
  With a *different* performance (`report.py --unpaired`, and real use) the estimates are much less precise:
  EQ differences below ~500 Hz are dominated by the pitch range sung, and compressor ratio is confounded by
  the performance's own dynamics (when the fit explains nothing, the compressor is set to 1:1 with only a
  level match). The time-based effects (delay, reverb, width) hold up best unpaired.
- Reverb damping is the least certain setting; the whole-chain round trip does not check it.
- Mild saturation (a few dB of drive) is hard to tell from compressor detail; it is only switched on when
  the whole chain clearly matches better with it.
- The compressor knee is fixed at 6 dB during estimation.
- Pitch correction: the amount is relative to your own singing; the retune time is approximate (within a
  factor of 2: the tracker's 40 ms window blurs note onsets), the fast / medium / slow class more reliable.
- Singer separation uses pitch range, timbre and level dynamics per phrase, and that singers take
  turns. It separates clearly different voices (e.g. a low and a high voice) and singers with
  different processing; two similar voices singing line by line can stay one singer (it says it is
  unsure): use regions then. Harmonies an octave apart are not found as overlap.
- Analysis takes one to a few minutes (Python); the refinement budget (`--refine`) is most of it.
- Windows: the analyzer's tests run on Windows; the plug-in is built for macOS (AU + VST3) and Linux (VST3,
  for the parity check), not yet for Windows.
