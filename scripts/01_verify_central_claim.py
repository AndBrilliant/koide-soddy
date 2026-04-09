#!/usr/bin/env python3
"""Verify the central claim: F^2 ≈ m_s(sum_lepton)."""

from koide_soddy.leptons import (
    M_E_MEV, M_MU_MEV, M_TAU_MEV, SUM_LEPTON_MEV,
    koide_Q, F_from_soddy, F_from_diff,
    F_central, F_squared_central, F_uncertainty_from_leptons,
)
from koide_soddy.running import run_ms_to_scale

# Quark mass inputs
M_S_2GEV = 93.44  # MeV, FLAG 2024 Table 11 Nf=2+1+1 (arXiv:2411.04268)
DM_S_2GEV = 0.68  # MeV, FLAG 2024 Table 11 uncertainty

print("=" * 60)
print("CENTRAL CLAIM VERIFICATION")
print("=" * 60)

# Koide relation
Q = koide_Q(M_E_MEV, M_MU_MEV, M_TAU_MEV)
print(f"\nKoide Q = {Q:.8f}")
print(f"Q - 2/3 = {Q - 2/3:.2e}")

# Soddy curvature
F = F_central()
F2 = F_squared_central()
F_diff = F_from_diff(M_E_MEV, M_MU_MEV, M_TAU_MEV)
print(f"\nF (Soddy form)  = {F:.6f} sqrt(MeV)")
print(f"F (diff form)   = {F_diff:.6f} sqrt(MeV)")
print(f"F^2 (Soddy)     = {F2:.4f} MeV")
print(f"F^2 (diff)      = {F_diff**2:.4f} MeV")

# Lepton uncertainty
sigma_F, sigma_F2 = F_uncertainty_from_leptons()
print(f"\nsigma_F  = {sigma_F:.2e} sqrt(MeV)")
print(f"sigma_F2 = {sigma_F2:.4f} MeV")

# Running m_s
mu = SUM_LEPTON_MEV
ms_2gev = M_S_2GEV
ms_mu = run_ms_to_scale(ms_2gev, mu)
print(f"\nmu = sum(m_lepton) = {mu:.4f} MeV")
print(f"m_s(2 GeV) = {ms_2gev:.1f} MeV")
print(f"m_s(mu)    = {ms_mu:.4f} MeV")

# Propagate lattice uncertainty through running
ms_up = run_ms_to_scale(ms_2gev + DM_S_2GEV, mu)
ms_dn = run_ms_to_scale(ms_2gev - DM_S_2GEV, mu)
sigma_ms_mu = (ms_up - ms_dn) / 2.0
print(f"sigma_ms(mu) = {sigma_ms_mu:.4f} MeV")

# Residual
residual_mev = F2 - ms_mu
residual_sigma = residual_mev / sigma_ms_mu
print(f"\n{'='*60}")
print(f"RESIDUAL: F^2 - m_s(mu) = {residual_mev:.4f} MeV")
print(f"RESIDUAL: F^2 - m_s(mu) = {residual_sigma:.2f} sigma")
print(f"{'='*60}")
