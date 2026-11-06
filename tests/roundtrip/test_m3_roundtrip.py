"""Milestone 3 synthetic round trips: saturation, de-esser, delay, reverb, doubler/width,
each on a realistic EQ + compressor vocal (see cases.run_m3_case)."""

import pytest

from cases import M3_CASES, run_m3_case


@pytest.mark.parametrize("name", list(M3_CASES))
def test_m3_roundtrip(name):
    res = run_m3_case(name)
    failed = [k for k, v in res.checks.items() if not v]
    assert res.ok, f"{name}: failed {failed}; errors {res.errors}; estimated {res.est}"
