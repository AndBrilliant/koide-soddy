"""Tests for lepton mass computations and Soddy curvature."""

from koide_soddy.leptons import (
    M_E_MEV, M_MU_MEV, M_TAU_MEV,
    koide_Q, F_from_soddy, F_from_diff,
    F_central, F_squared_central, F_uncertainty_from_leptons,
)


def test_koide_Q_pdg_value():
    """Koide Q at PDG central values lies in [0.66665, 0.66667]."""
    Q = koide_Q(M_E_MEV, M_MU_MEV, M_TAU_MEV)
    assert 0.66665 < Q < 0.66667, f"Koide Q = {Q}, expected in [0.66665, 0.66667]"


def test_F_soddy_diff_agreement():
    """F_from_soddy and F_from_diff agree to better than 1e-4 relative.

    They are only exactly equal when Koide holds exactly; at PDG values
    Q - 2/3 ~ -2.2e-6, so the two forms differ at ~2e-5 relative.
    """
    F_s = F_from_soddy(M_E_MEV, M_MU_MEV, M_TAU_MEV)
    F_d = F_from_diff(M_E_MEV, M_MU_MEV, M_TAU_MEV)
    rel = abs(F_s - F_d) / abs(F_s)
    assert rel < 1e-4, f"Relative difference = {rel}, expected < 1e-4"


def test_F_central_value():
    """F_central() (Soddy form) returns ~9.7526 sqrt(MeV).

    The Soddy form gives 9.75261; the Koide-simplified diff form gives
    9.75282 (matching the user's quoted 9.7528). The 2e-5 relative
    difference is due to Q != 2/3 exactly.
    """
    F = F_central()
    assert abs(F - 9.7526) < 0.0003, f"F = {F}, expected ~9.7526"


def test_F_squared_central_value():
    """F_squared_central() returns ~95.113 MeV (Soddy form)."""
    F2 = F_squared_central()
    assert abs(F2 - 95.113) < 0.006, f"F^2 = {F2}, expected ~95.113"


def test_F_uncertainty_lepton_dominated():
    """sigma_F2 from leptons is small (< 0.02 MeV), confirming
    the dominant uncertainty is from the quark side, not leptons.
    """
    sigma_F, sigma_F2 = F_uncertainty_from_leptons()
    assert sigma_F2 < 0.02, f"sigma_F2 = {sigma_F2}, expected < 0.02 MeV"
    assert sigma_F > 0, "sigma_F must be positive"
