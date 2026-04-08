#!/usr/bin/env python3
"""Filter sensitivity: tight null with different mu* lower cutoffs."""

from koide_soddy.filter_sensitivity import run_filter_sensitivity

result = run_filter_sensitivity(n_samples=100_000, seed=42)

print("=" * 60)
print("FILTER SENSITIVITY")
print("=" * 60)
for c in result["cutoffs"]:
    ci = c["ci_95"]
    note = f"  ({c['note']})" if "note" in c else ""
    print(f"  mu_min = {c['mu_min_mev']:>6.0f} MeV: "
          f"{100*c['hit_fraction']:.2f}%  "
          f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]  "
          f"n_valid={c['n_valid']}{note}")

print(f"\n  Stable (max/min < 2): {result['stable']}")
print(f"\nOutput: results/filter_sensitivity.json")
