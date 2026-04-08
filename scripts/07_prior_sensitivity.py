#!/usr/bin/env python3
"""Prior sensitivity: tight null with four alternative priors."""

from koide_soddy.prior_sensitivity import run_prior_sensitivity

result = run_prior_sensitivity(n_samples=100_000, seed=42)

print("=" * 60)
print("PRIOR SENSITIVITY")
print("=" * 60)
for name, r in result["priors"].items():
    ci = r["ci_95"]
    print(f"  {name:20s}: {100*r['hit_fraction']:.2f}%  "
          f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]  "
          f"(n_valid={r['n_valid']})")

s = result["summary"]
print(f"\n  Min hit fraction: {100*s['min_hit_fraction']:.2f}%")
print(f"  Max hit fraction: {100*s['max_hit_fraction']:.2f}%")
print(f"  Fold variation:   {s['fold_variation']:.1f}x")

if s["fold_variation"] < 5:
    print("  Verdict: ROBUST to prior choice")
elif s["fold_variation"] < 20:
    print("  Verdict: MODERATE prior sensitivity — discuss in paper")
else:
    print("  Verdict: HIGHLY PRIOR-SENSITIVE — rarity claim must be dropped")

print(f"\nOutput: results/prior_sensitivity.json")
