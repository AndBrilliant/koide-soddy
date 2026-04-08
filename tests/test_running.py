"""Tests for QCD running of quark masses."""

import pytest
from koide_soddy.running import run_ms_to_scale, run_mc_to_scale, run_mb_to_scale


def test_ms_roundtrip():
    """Running m_s from 2 GeV to 2 GeV returns ~93.5 MeV."""
    ms = run_ms_to_scale(93.5, 2000.0)
    assert abs(ms - 93.5) < 0.01, f"Round-trip gave {ms}, expected ~93.5"


def test_ms_at_lepton_sum():
    """m_s(1883 MeV) ~ 95.1 MeV — the central observation."""
    ms = run_ms_to_scale(93.5, 1883.0)
    assert abs(ms - 95.1) < 0.5, f"m_s(1883 MeV) = {ms}, expected ~95.1"


def test_mc_at_mb():
    """m_c(m_b) ~ 924 MeV — published PDG cross-check."""
    mc = run_mc_to_scale(1273.0, 4183.0)
    assert abs(mc - 924) < 15, f"m_c(m_b) = {mc}, expected ~924"


@pytest.mark.parametrize("scale_mev", [1000, 2000, 5000, 10000, 50000, 100000])
def test_ms_positive_finite(scale_mev):
    """All running calls produce finite, positive values for scales in [1, 100] GeV."""
    ms = run_ms_to_scale(93.5, scale_mev)
    assert ms > 0, f"m_s({scale_mev} MeV) = {ms}, must be positive"
    assert ms < 1000, f"m_s({scale_mev} MeV) = {ms}, suspiciously large"
