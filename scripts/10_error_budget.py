#!/usr/bin/env python3
"""Error budget decomposition for m_s at lepton sum scale."""

from koide_soddy.error_budget import run_error_budget

result = run_error_budget()

print("=" * 60)
print("ERROR BUDGET")
print("=" * 60)
c = result["central"]
print(f"  m_s(mu*) = {c['ms_mu_star_mev']} MeV at mu* = {c['mu_star_mev']} MeV")
print(f"  Phase 1 sigma_total = {c['sigma_total_phase1_mev']} MeV\n")

print("  Components:")
for comp in result["components"]:
    src = comp["source"]
    sig = comp["sigma_contribution_mev"]
    if "shift_up_mev" in comp:
        print(f"    {src:20s}: +/- {sig:.4f} MeV  "
              f"(+{comp['shift_up_mev']:.4f} / {comp['shift_dn_mev']:.4f})")
    else:
        print(f"    {src:20s}: +/- {sig:.4f} MeV  "
              f"(4L-3L = {comp['shift_mev']:.4f})")

print(f"\n  Quadrature sum:     {result['quadrature_sum_mev']:.4f} MeV")
print(f"  Phase 1 total:      {c['sigma_total_phase1_mev']:.4f} MeV")
print(f"  {result['note']}")
print(f"\nOutput: results/error_budget.json")
