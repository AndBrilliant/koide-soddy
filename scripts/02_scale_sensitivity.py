#!/usr/bin/env python3
"""Scale sensitivity analysis: how does the residual vary with mu?"""

from koide_soddy.sensitivity import scale_sensitivity

result = scale_sensitivity()

print("Scale sensitivity analysis")
print(f"F^2 = {result['F_squared_mev']} MeV")
print(f"mu* = {result['mu_star_mev']} MeV")
print()

# Print a selection of rows
print(f"{'mult':>5s}  {'mu (MeV)':>10s}  {'ms (MeV)':>10s}  {'resid (MeV)':>12s}  {'resid (σ)':>10s}")
print("-" * 55)
for row in result["data"]:
    if row["scale_multiplier"] in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0]:
        print(f"{row['scale_multiplier']:5.1f}  {row['mu_mev']:10.1f}  {row['ms_mu_mev']:10.4f}  "
              f"{row['residual_mev']:12.4f}  {row['residual_sigma']:10.4f}")

print()
if result["one_sigma_window_mev"]:
    lo, hi = result["one_sigma_window_mev"]
    print(f"1-sigma window: [{lo:.0f}, {hi:.0f}] MeV")
    print(f"  = [{lo/result['mu_star_mev']:.2f}, {hi/result['mu_star_mev']:.2f}] × sum(m_lepton)")
if result["two_sigma_window_mev"]:
    lo, hi = result["two_sigma_window_mev"]
    print(f"2-sigma window: [{lo:.0f}, {hi:.0f}] MeV")
    print(f"  = [{lo/result['mu_star_mev']:.2f}, {hi/result['mu_star_mev']:.2f}] × sum(m_lepton)")

print("\nOutput written to results/scale_sensitivity.json")
