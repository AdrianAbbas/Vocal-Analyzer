"""Main correctness test (SPEC "Synthetic round-trip"): render a dry vocal
with known EQ / compressor settings, analyse it, check recovery."""

import pytest

from vocalmatch.preset import validate_preset_dict

from cases import (
    COMBINED_CASES,
    COMP_CASES,
    EQ_BAND_TOL_DB,
    EQ_CASES,
    run_combined_case,
    run_comp_case,
    run_eq_case,
)


@pytest.mark.parametrize("name", list(EQ_CASES))
def test_eq_roundtrip(name):
    res = run_eq_case(name)
    worst = res.centres_hz[abs(res.band_err_db).argmax()]
    assert res.ok, f"{name}: worst 1/3-oct band error {res.max_err_db:.2f} dB at {worst:.0f} Hz (tol {EQ_BAND_TOL_DB})"


@pytest.mark.parametrize("name", list(COMP_CASES))
def test_compressor_roundtrip(name):
    res = run_comp_case(name)
    assert res.ok, f"{name}: failed {[k for k, v in res.checks.items() if not v]}; errors {res.errors}; est {res.est}"


@pytest.mark.parametrize("name", list(COMBINED_CASES))
def test_combined_roundtrip(name):
    eq_res, comp_res, preset = run_combined_case(name)
    validate_preset_dict(preset.to_dict())
    assert eq_res.ok, f"{name}: EQ worst band error {eq_res.max_err_db:.2f} dB"
    assert comp_res.ok, f"{name}: failed {[k for k, v in comp_res.checks.items() if not v]}; errors {comp_res.errors}"
