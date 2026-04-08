#!/usr/bin/env python3
"""Scale alternatives: m_s at multiple lepton-derived scales."""

from koide_soddy.scale_alternatives import run_scale_alternatives

result = run_scale_alternatives()

print("=" * 60)
print("SCALE ALTERNATIVES")
print("=" * 60)
print(f"F^2 = {result['F_squared_mev']} MeV\n")
print(f"{'Label':>14s}  {'mu (MeV)':>10s}  {'m_s (MeV)':>10s}  {'resid (sigma)':>14s}  {'Hit':>4s}")
print("-" * 62)
for s in result["scales"]:
    if s["valid"]:
        print(f"{s['label']:>14s}  {s['mu_mev']:10.2f}  {s['ms_mu_mev']:10.4f}  "
              f"{s['residual_sigma']:14.4f}  {'YES' if s['hit'] else 'no':>4s}")
    else:
        print(f"{s['label']:>14s}  {s['mu_mev']:10.2f}  {'N/A':>10s}  {'N/A':>14s}  {'N/A':>4s}  "
              f"({s.get('note', '')})")

print(f"\nHits: {result['hit_labels']}")
print(f"\nOutput: results/scale_alternatives.json")
