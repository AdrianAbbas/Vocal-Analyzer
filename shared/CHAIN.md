# Vocal Match effect chain — DSP reference

This is the contract between the Python reference chain
(`analysis/vocalmatch/chain/`) and the JUCE plugin (`plugin/Source/DSP/`).
Both must implement exactly this so that, fed the same preset and input, they
agree within -40 dB of residual (SPEC "Python vs C++ check",
`tests/parity/test_cpp_match.py`, which runs the plug-in's own processor).
Both run in double precision.

Conventions: processing is stereo `(L, R)`; a mono input is copied to both
channels. All state starts at zero on reset. `fs` is the sample rate.
`dB→lin(x) = 10^(x/20)`. `round()` is Python's: half to even. Times in
milliseconds become samples as `round(t·fs/1000)` with `t` first rounded to
0.001 ms (the plug-in stores parameters as float; without this a value
such as 375 ms = 16537.5 samples at 44.1 kHz could land one sample off). A disabled effect is a bit-exact bypass. An
effect whose wet amount is zero (`mix_pct = 0`, `reduction_db = 0`) is
also skipped. All of these are latency-free (no look-ahead).

Order: **input gain → EQ → compressor → saturation → de-esser → delay →
reverb → doubler/width → output gain.**

## Biquads (RBJ Audio EQ Cookbook)
`A = 10^(gain_db/40)`, `w0 = 2π·f/fs` with `f` clamped to `[1, 0.49·fs]`,
`α = sin(w0)/(2Q)`. Peak, low shelf and high shelf use the cookbook formulas
with this Q-form α; shelves use `2√A·α`. Coefficients are normalised by
`a0` and run as transposed direct form II. See `chain/biquad.py:rbj_sos`.

## Input / output gain
`y = x · dB→lin(gain_db)`.

## EQ
Up to 6 bands in list order, each an RBJ `low_shelf` / `peak` / `high_shelf`.
Bands with `enabled = false` or `gain_db = 0` are skipped.

## Compressor (feed-forward, linked stereo)
1. Detector `d[n] = max(|L[n]|, |R[n]|)`, `x_dB = 20·log10(max(d, 10^-6))`.
2. Static soft-knee gain computer (Giannoulis et al. 2012), threshold `T`,
   ratio `R`, knee `W`:
   - `2(x−T) < −W`: `y = x`
   - `2|x−T| ≤ W`: `y = x + (1/R − 1)(x − T + W/2)² / (2W)`
   - `2(x−T) > W`: `y = T + (x − T)/R`

   Gain reduction `gr = x − y ≥ 0`. (`W = 0`: hard knee.)
3. Ballistics on the gain reduction, in dB ("smooth branching"):
   `αA = exp(−1/(attack_s·fs))`, `αR = exp(−1/(release_s·fs))`;
   `s = αA·s + (1−αA)·gr` if `gr > s`, else `s = αR·s + (1−αR)·gr`.
4. `y = x · dB→lin(makeup_db − s)` on both channels.

## Saturation
`g = dB→lin(drive_db)`, `wet = tanh(g·x)/g` (unity small-signal gain,
soft-clips at ±1/g), `y = (1 − m)·x + m·wet` with `m = mix_pct/100`.

## De-esser (split band)
1. Linkwitz-Riley 4 crossover at `frequency_hz`: `low` = two cascaded RBJ
   low-pass sections (Q = 1/√2), `high` = two cascaded RBJ high-pass sections.
2. Detector `max(|highL|, |highR|)` in dB (floor -120 dB),
   `gr = clamp(level − threshold_db, 0, reduction_db)`.
3. Smooth `gr` as in the compressor with attack 0.5 ms, release 40 ms.
4. `y = low + dB→lin(−s)·high`. (With no reduction the de-esser is an
   all-pass; it is skipped entirely when `reduction_db = 0`.)

## Delay (per channel)
`D = max(1, round(time_ms·fs/1000))` samples; one-pole low-pass
`w += a(r − w)`, `a = 1 − exp(−2π·highcut_hz/fs)`.
`r[n] = buf[n − D]`, `w[n] = LP(r)[n]`, `buf[n] = x[n] + fb·w[n]`,
`y = (1 − m)·x + m·w`, with `fb = feedback_pct/100`, `m = mix_pct/100`.
`sync_note` is informational only.

## Reverb (8-line FDN)
- Line lengths `d_i = round(ms_i·fs/1000)`,
  `ms = 29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.7, 73.3`.
- Feedback matrix: 8×8 Sylvester Hadamard / √8.
- Per line: one-pole low-pass at `damping_hz` (same form as the delay)
  and gain `g_i = 10^(−3·d_i/(fs·decay_s))`, so `decay_s` is the RT60 at
  low frequencies.
- Input `u[n]` = mid `(L+R)/2` delayed by `round(predelay_ms·fs/1000)`.
- Each sample: `r_i = buf_i[n − d_i]`, `f_i = g_i·LP_i(r_i)`,
  `buf_i[n] = Σ_j H_ij·f_j + u[n]/√8`.
- Wet `L = Σ cL_i·f_i`, `R = Σ cR_i·f_i` with
  `cL = (+,−,+,−,+,−,+,−)/√8`, `cR = (+,+,−,−,+,+,−,−)/√8`, both scaled by
  `√(1 − mean(g)²)` to keep the wet energy roughly independent of decay.
- `y = (1 − m)·x + m·wet`.

## Doubler / width
- Voice per channel `c`: `v_c[n] = x_c(n − D_c(n))`,
  `D_c(n) = (base_c + depth_c·sin(2π·rate_c·n/fs))·fs/1000` samples, read
  with linear interpolation (zero before the start of the signal).
  L: base 14 ms, depth 1.6 ms, rate 0.31 Hz; R: base 21 ms, depth 2.1 ms,
  rate 0.23 Hz. LFO phase is 0 at reset.
- `y_c = x_c + m·v_c`, `m = mix_pct/100`.
- Width: `M = (L+R)/2`, `S = (L−R)/2 · width_pct/100`, `L = M+S`, `R = M−S`.

## Match amount (plug-in knob, `render(..., match_amount=)`)
`m` in `[0, 1]` scales every effect's departure from neutral before processing:
gains (`input_gain`, `output_gain`, every EQ band `gain_db`, compressor
`makeup_db`) are multiplied by `m`; compressor `ratio → 1 + (ratio − 1)·m`;
`mix_pct` of saturation, delay, reverb and doubler and the de-esser's
`reduction_db` are multiplied by `m`; doubler `width_pct → 100 + (width_pct − 100)·m`.
Frequencies, Q, threshold, knee, times, feedback and decay are unchanged.
`m = 0` is neutral (the dry signal, apart from compressor ballistics of a
1:1 ratio, which are an exact identity). `apply_match_amount` in
`chain/__init__.py` and `applyMatchAmount` in `plugin/Source/DSP/Chain.h`.

## Plug-in I/O
Mono → mono, mono → stereo and stereo → stereo. A mono input feeds both
channels; a mono output is `(L + R)/2`. Latency is 0.
