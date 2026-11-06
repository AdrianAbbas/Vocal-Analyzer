"""EQ estimator.

1. Long-term average spectrum (LTAS) of reference and dry over active frames,
   each frame normalised to unit power (so level/compression does not matter).
2. Both smoothed to 1/3 octave on a 1/12-octave log grid.
3. Fit up to 6 RBJ bands (low shelf, 4 peaks, high shelf) plus a free
   broadband offset so that  Smooth(|H_eq|^2 * LTAS_dry) ~= Smooth(LTAS_ref)
   in dB. Fitting the *smoothed model* to the *smoothed data* means the 1/3
   octave smoothing does not bias narrow bands. Bands are added greedily at
   the largest remaining error, then refined jointly (scipy least_squares).
   Gains are bounded to +-12 dB with a small L2 penalty so unused bands stay
   at 0 dB; nearby peaks of opposite sign are discouraged.
4. Broadband level is not observable from frame-normalised spectra, so the
   EQ is only defined up to a constant. Convention: a weak penalty keeps the
   EQ's overall level change on the dry vocal near 0 dB; the free offset is
   *not* part of the EQ and level is left to compressor make-up/output gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..chain.biquad import rbj_sos, sos_power_response
from ..features import fractional_octave_matrix, log_grid, ltas, power_db
from ..preset import EQ, EQBand, default_eq_bands

GAIN_LIMIT_DB = 12.0
GAIN_PENALTY = 0.15  # weight of the (smoothed L1) band-gain penalty
FIGHT_PENALTY = 0.5  # weight of the opposite-sign neighbouring-peaks term
FIGHT_OCTAVES = 1.0  # ... and its reach in octaves
LEVEL_PENALTY = 0.3  # pulls the EQ towards not changing overall level
GROUPS_PER_OCTAVE = 48
MAX_NFEV = 80
PRUNE_DB = 0.3  # bands smaller than this are switched off
WEIGHT_RANGE_DB = 60.0  # bins this far below the dry LTAS peak get no weight

# (type, f_lo, f_hi, q_lo, q_hi, f_init, q_init)
BAND_LAYOUT = (
    ("low_shelf", 30.0, 600.0, 0.4, 1.5, 120.0, 0.7071),
    ("peak", 60.0, 16000.0, 0.3, 8.0, 1000.0, 1.4),
    ("peak", 60.0, 16000.0, 0.3, 8.0, 1000.0, 1.4),
    ("peak", 60.0, 16000.0, 0.3, 8.0, 1000.0, 1.4),
    ("peak", 60.0, 16000.0, 0.3, 8.0, 1000.0, 1.4),
    ("high_shelf", 1000.0, 16000.0, 0.4, 1.5, 8000.0, 0.7071),
)


@dataclass
class EQEstimate:
    eq: EQ
    offset_db: float
    residual_db: float
    grid_hz: np.ndarray
    target_db: np.ndarray  # smoothed reference/dry difference curve
    fitted_db: np.ndarray  # smoothed fitted EQ response (incl. offset)
    weights: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class _Problem:
    def __init__(self, freqs, p_dry, p_ref, fs, f_lo, f_hi, points_per_octave):
        self.fs = fs
        self.f_hi_band = min(16000.0, 0.45 * fs)
        self.grid = log_grid(f_lo, min(f_hi, self.f_hi_band), points_per_octave)
        smooth = fractional_octave_matrix(freqs, self.grid, 3.0)
        used = smooth.any(axis=0)
        smooth, freqs, p_dry, p_ref = smooth[:, used], freqs[used], p_dry[used], p_ref[used]
        ref_s = smooth @ p_ref
        dry_s = smooth @ p_dry
        self.dry_lin = dry_s
        self.target = power_db(ref_s)
        self.dry_db = power_db(dry_s)
        self.diff = self.target - self.dry_db
        rel = self.dry_db - (self.dry_db.max() - WEIGHT_RANGE_DB)
        self.w = np.clip(rel / 10.0, 0.0, 1.0)
        # The model is Smooth(|H|^2 * P_dry) = (Smooth * P_dry) @ |H|^2.
        # |H| is effectively constant within 1/48 octave, so FFT bins are
        # grouped and |H| is evaluated once per group (exact up to that).
        group = np.floor(np.log2(freqs / freqs[0]) * GROUPS_PER_OCTAVE).astype(int)
        uniq, inv = np.unique(group, return_inverse=True)
        weighted = smooth * p_dry[None, :]
        self.model_matrix = np.zeros((len(self.grid), len(uniq)))
        np.add.at(self.model_matrix.T, inv, weighted.T)
        f_group = np.zeros(len(uniq))
        np.add.at(f_group, inv, freqs)
        f_group /= np.bincount(inv)
        w = 2 * np.pi * f_group / fs
        self.z1 = np.exp(-1j * w)
        self.z2 = self.z1 * self.z1

    def band_limits(self, i):
        kind, f_lo, f_hi, q_lo, q_hi, f0, q0 = BAND_LAYOUT[i]
        return kind, f_lo, min(f_hi, self.f_hi_band), q_lo, q_hi, f0, q0

    def power(self, bands):
        h = np.ones_like(self.z1)
        for kind, f, g, q in bands:
            if g == 0.0:
                continue
            b0, b1, b2, _, a1, a2 = rbj_sos(kind, f, g, q, self.fs)
            h *= (b0 + b1 * self.z1 + b2 * self.z2) / (1.0 + a1 * self.z1 + a2 * self.z2)
        return np.abs(h) ** 2

    def model_db(self, bands, offset):
        return power_db(self.model_matrix @ self.power(bands)) + offset

    def residuals(self, bands, offset):
        """Weighted fit error, regularisers, and the level-neutrality term."""
        model_lin = self.model_matrix @ self.power(bands)
        r = self.w * (power_db(model_lin) + offset - self.target)
        # EQ's overall level change on the dry vocal (power-weighted, dB).
        level = power_db(np.sum(self.w * model_lin) / np.sum(self.w * self.dry_lin))
        return np.concatenate([r, _penalty(bands), [LEVEL_PENALTY * level]])


def _unpack(theta, active, prob):
    offset = theta[0]
    bands = []
    for j, i in enumerate(active):
        kind = BAND_LAYOUT[i][0]
        lf, g, lq = theta[1 + 3 * j : 4 + 3 * j]
        bands.append((kind, 2.0**lf, g, 2.0**lq))
    return offset, bands


def _fit(prob: _Problem, active: list[int], init: dict[int, tuple[float, float, float]], offset0: float):
    x0 = [offset0]
    lo = [-60.0]
    hi = [60.0]
    for i in active:
        kind, f_lo, f_hi, q_lo, q_hi, _, _ = prob.band_limits(i)
        f, g, q = init[i]
        x0 += [np.log2(np.clip(f, f_lo, f_hi)), np.clip(g, -GAIN_LIMIT_DB, GAIN_LIMIT_DB), np.log2(np.clip(q, q_lo, q_hi))]
        lo += [np.log2(f_lo), -GAIN_LIMIT_DB, np.log2(q_lo)]
        hi += [np.log2(f_hi), GAIN_LIMIT_DB, np.log2(q_hi)]
    x0 = np.clip(np.array(x0), np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    def resid(theta):
        offset, bands = _unpack(theta, active, prob)
        return prob.residuals(bands, offset)

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=MAX_NFEV)
    offset, bands = _unpack(sol.x, active, prob)
    params = {i: (b[1], b[2], b[3]) for i, b in zip(active, bands)}
    return offset, params


def _penalty(bands) -> np.ndarray:
    """Regularisation residuals.

    * smoothed L1 on band gains (sum of squares ~ GAIN_PENALTY^2 * sum|g|):
      unlike L2 it does not reward splitting one boost over several
      overlapping bands, and it drives unneeded bands to ~0 dB.
    * a "fighting bands" term for nearby peaks of opposite sign, which
      otherwise cancel each other to chase fine LTAS detail.
    """
    gains = np.array([b[2] for b in bands])
    out = [GAIN_PENALTY * np.sqrt(np.sqrt(gains * gains + 0.01))]
    peaks = [b for b in bands if b[0] == "peak"]
    fight = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            gi, gj = peaks[i][2], peaks[j][2]
            if gi * gj < 0:
                dist = np.log2(peaks[i][1] / peaks[j][1])
                fight.append(FIGHT_PENALTY * np.sqrt(-gi * gj) * np.exp(-(dist / FIGHT_OCTAVES) ** 2))
            else:
                fight.append(0.0)
    if fight:
        out.append(np.array(fight))
    return np.concatenate(out)


def _cost(prob: _Problem, active, params, offset) -> float:
    bands = [(BAND_LAYOUT[k][0], *params[k]) for k in active if k in params]
    r = prob.residuals(bands, offset)
    return float(np.sum(r * r))


def _greedy_peaks(prob: _Problem, peak_ids, params, offset):
    active = [0, 5]
    for i in peak_ids:
        bands = [(BAND_LAYOUT[k][0], *params[k]) for k in active]
        err = prob.target - prob.model_db(bands, offset)
        score = prob.w * np.abs(err)
        for k in active:  # avoid stacking on existing peaks
            if BAND_LAYOUT[k][0] == "peak":
                score[np.abs(np.log2(prob.grid / params[k][0])) < 1 / 3] = 0
        idx = int(np.argmax(score))
        if score[idx] < PRUNE_DB:
            break
        params[i] = (float(prob.grid[idx]), float(err[idx]), 1.4)
        active.append(i)
        offset, params = _fit(prob, active, params, offset)
    return _cost(prob, active, params, offset), offset, params


MERGE_COST_TOL = 0.02  # a merge may raise the fit cost by this fraction (+ a small absolute slack)


def _merge_duplicate_peaks(prob: _Problem, peak_ids, params, offset):
    """Two peaks closer than 1/6 octave usually do the job of one: merge them (gains add),
    refit, and keep the merge when the fit is (nearly) as good, so the preset stays readable.
    Deciding by the fit rather than by a Q threshold keeps the result stable when numerical
    differences (BLAS, CPU) leave the two peaks with somewhat different Q."""
    rejected: set[tuple[int, int]] = set()
    while True:
        live = [i for i in peak_ids if i in params and abs(params[i][1]) >= PRUNE_DB]
        pairs = sorted(
            ((abs(np.log2(params[a][0] / params[b][0])), a, b) for a in live for b in live
             if a < b and (a, b) not in rejected and abs(np.log2(params[a][0] / params[b][0])) < 1 / 6),
        )
        if not pairs:
            return offset, params
        _, a, b = pairs[0]
        (fa, ga, qa), (fb, gb, qb) = params[a], params[b]
        merged = dict(params)
        # gain-weighted centre and Q of the pair
        wa, wb = abs(ga) + 1e-9, abs(gb) + 1e-9
        merged[a] = (float(np.exp((wa * np.log(fa) + wb * np.log(fb)) / (wa + wb))), ga + gb,
                     float(np.exp((wa * np.log(qa) + wb * np.log(qb)) / (wa + wb))))
        del merged[b]
        off_m, par_m = _fit(prob, sorted(merged), merged, offset)
        before = _cost(prob, sorted(params), params, offset)
        after = _cost(prob, sorted(par_m), par_m, off_m)
        if after <= before * (1.0 + MERGE_COST_TOL) + 1e-4:
            offset, params = off_m, par_m
        else:
            rejected.add((a, b))


def estimate_eq(
    reference: np.ndarray,
    dry: np.ndarray,
    fs: float,
    n_peaks: int = 4,
    f_lo: float = 60.0,
    f_hi: float = 16000.0,
    points_per_octave: int = 12,
    n_shelf_starts: int = 3,
    n_random_starts: int = 4,
    seed: int = 0,
    dry_correction: np.ndarray | None = None,
) -> EQEstimate:
    """*dry_correction*: optional power ratio per LTAS bin applied to the dry LTAS: the
    spectral effect of the (non-linear) effects after the EQ, so they are not fitted as EQ."""
    freqs, p_ref = ltas(reference, fs)
    _, p_dry = ltas(dry, fs)
    if dry_correction is not None:
        p_dry = p_dry * dry_correction
    prob = _Problem(freqs, p_dry, p_ref, fs, f_lo, f_hi, points_per_octave)

    offset0 = float(np.sum(prob.w * prob.diff) / np.sum(prob.w))
    peak_ids = [1, 2, 3, 4][:n_peaks]
    rng = np.random.default_rng(seed)

    candidates = []
    # (a) greedy: shelves from a small grid of start frequencies, then peaks
    #     added one at a time where the remaining error is largest.
    shelf_fits = []
    for f_ls in (80.0, 160.0, 320.0):
        for f_hs in (2500.0, 6000.0, 12000.0):
            init = {0: (f_ls, 0.0, 0.7071), 5: (f_hs, 0.0, 0.7071)}
            off, par = _fit(prob, [0, 5], init, offset0)
            shelf_fits.append((_cost(prob, [0, 5], par, off), off, par))
    shelf_fits.sort(key=lambda t: t[0])
    for _, off, par in shelf_fits[:n_shelf_starts]:
        candidates.append(_greedy_peaks(prob, peak_ids, dict(par), off))
    # (b) random restarts with every band active.
    active_all = [0, 5] + peak_ids
    for _ in range(n_random_starts):
        init = {}
        for i in active_all:
            kind, f_lo_i, f_hi_i, q_lo, q_hi, f0, q0 = prob.band_limits(i)
            f = float(np.exp(rng.uniform(np.log(max(f_lo_i, prob.grid[0])), np.log(min(f_hi_i, prob.grid[-1])))))
            g = float(np.interp(f, prob.grid, prob.diff) - offset0)
            init[i] = (f, g / 2.0, q0)
        off, par = _fit(prob, active_all, init, offset0)
        candidates.append((_cost(prob, active_all, par, off), off, par))

    _, offset, params = min(candidates, key=lambda t: t[0])
    offset, params = _merge_duplicate_peaks(prob, peak_ids, params, offset)

    # Prune negligible bands and produce the fixed 6-band layout.
    bands_out = default_eq_bands()
    peaks = sorted((params[i] for i in peak_ids if i in params), key=lambda p: p[0])
    slots = {0: params.get(0), 5: params.get(5)}
    for j, p in enumerate(peaks):
        slots[1 + j] = p
    for slot, p in slots.items():
        if p is None:
            continue
        f, g, q = p
        b = bands_out[slot]
        b.freq_hz, b.gain_db, b.q = round(float(f), 2), round(float(g), 2), round(float(q), 3)
        b.enabled = bool(abs(g) >= PRUNE_DB)
        if not b.enabled:
            b.gain_db = 0.0

    eq = EQ(enabled=True, bands=bands_out)
    final_bands = [(b.type, b.freq_hz, b.gain_db, b.q) for b in bands_out if b.enabled]
    fitted = prob.model_db(final_bands, 0.0) - prob.dry_db
    # re-solve the offset for the pruned/rounded bands
    offset = float(np.sum(prob.w * (prob.diff - fitted)) / np.sum(prob.w))
    fitted = fitted + offset
    resid = float(np.sqrt(np.sum(prob.w * (fitted - prob.diff) ** 2) / np.sum(prob.w)))
    eq.confidence = round(float(np.exp(-resid / 1.5)), 3)
    if not any(b.enabled for b in bands_out):
        eq.enabled = False

    return EQEstimate(
        eq=eq,
        offset_db=offset,
        residual_db=resid,
        grid_hz=prob.grid,
        target_db=prob.diff,
        fitted_db=fitted,
        weights=prob.w,
        diagnostics={"offset_db": round(offset, 2), "residual_db": round(resid, 3)},
    )


def eq_response_db(eq: EQ, freqs_hz: np.ndarray, fs: float) -> np.ndarray:
    """Unsmoothed magnitude response of an EQ in dB."""
    rows = [rbj_sos(b.type, b.freq_hz, b.gain_db, b.q, fs) for b in eq.bands if b.enabled and b.gain_db != 0]
    if not rows or not eq.enabled:
        return np.zeros(len(np.atleast_1d(freqs_hz)))
    return power_db(sos_power_response(np.vstack(rows), freqs_hz, fs))
