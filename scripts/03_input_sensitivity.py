#!/usr/bin/env python3
"""Input sensitivity analysis: full error budget on F^2 - m_s."""

from koide_soddy.sensitivity import input_sensitivity

result = input_sensitivity()

print("Input sensitivity / error budget")
print(f"F^2           = {result['F_squared_mev']} MeV")
print(f"m_s(mu*)      = {result['ms_at_mu_mev']} MeV")
print(f"Residual      = {result['residual_mev']} MeV")
print()
eb = result["error_budget"]
print("Error budget:")
print(f"  sigma(F^2) from leptons  = {eb['sigma_F2_lepton_mev']:.6f} MeV ({eb['lepton_fraction_pct']:.1f}%)")
print(f"  sigma(m_s) from running  = {eb['sigma_ms_running_mev']:.4f} MeV ({eb['quark_fraction_pct']:.1f}%)")
print(f"  sigma(total)             = {eb['sigma_total_mev']:.4f} MeV")
print()
print(f"Residual / sigma(total) = {result['residual_in_sigma']:.4f}")
print("\nOutput written to results/input_sensitivity.json")
