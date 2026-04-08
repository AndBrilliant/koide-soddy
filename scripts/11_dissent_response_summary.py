#!/usr/bin/env python3
"""Dissent response summary: read all Phase 2 results and print a single report."""

import json


def load(path):
    with open(path) as f:
        return json.load(f)


prior = load("results/prior_sensitivity.json")
scale = load("results/scale_alternatives.json")
filt = load("results/filter_sensitivity.json")
budget = load("results/error_budget.json")
inp = load("results/input_sensitivity.json")

print("=" * 50)
print("DISSENT RESPONSE SUMMARY")
print("=" * 50)

# [1] Prior sensitivity
print("\n[1] Prior sensitivity:")
for name, r in prior["priors"].items():
    ci = r["ci_95"]
    print(f"  {name:20s}: {100*r['hit_fraction']:.2f}%  "
          f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]")
s = prior["summary"]
print(f"  Fold variation: {s['fold_variation']:.1f}x")
if s["fold_variation"] < 5:
    print("  Verdict: ROBUST")
elif s["fold_variation"] < 20:
    print("  Verdict: MODERATE — discuss in paper")
else:
    print("  Verdict: HIGHLY PRIOR-SENSITIVE — drop rarity claim")

# [2] Scale alternatives
print("\n[2] Scale alternatives:")
for sc in scale["scales"]:
    if sc["valid"]:
        tag = "HIT" if sc["hit"] else f"{sc['residual_sigma']:.1f} sigma"
        print(f"  {sc['label']:>14s}: mu={sc['mu_mev']:.0f} MeV -> {tag}")
    else:
        print(f"  {sc['label']:>14s}: mu={sc['mu_mev']:.1f} MeV -> INVALID (non-perturbative)")
print(f"  Hits: {scale['hit_labels']}")
if scale["n_hits"] <= 2 and "sum" in scale["hit_labels"]:
    print("  Verdict: ONLY sum hits (mu_plus_tau is trivially equivalent)")
elif scale["n_hits"] <= 4:
    print("  Verdict: MULTIPLE HITS — specificity weakened")
else:
    print("  Verdict: CATASTROPHIC — many scales match")

# [3] Filter sensitivity
print("\n[3] Filter sensitivity:")
for c in filt["cutoffs"]:
    baseline = " (baseline)" if c["mu_min_mev"] == 1000 else ""
    note = f" [{c['note']}]" if "note" in c else ""
    print(f"  {c['mu_min_mev']:.0f} MeV: {100*c['hit_fraction']:.2f}%{baseline}{note}")
# Compute max/min excluding zero
fracs = [c["hit_fraction"] for c in filt["cutoffs"] if c["hit_fraction"] > 0]
if len(fracs) >= 2:
    ratio = max(fracs) / min(fracs)
    print(f"  Max/min (nonzero): {ratio:.1f}x")
if filt["stable"]:
    print("  Verdict: STABLE")
else:
    print("  Verdict: DRIFTING — filter is doing real work")

# [4] Error budget
print("\n[4] Error budget decomposition:")
for comp in budget["components"]:
    print(f"  {comp['source']:20s}: +/- {comp['sigma_contribution_mev']:.4f} MeV")
print(f"  {'Quadrature sum':20s}:     {budget['quadrature_sum_mev']:.4f} MeV")
print(f"  {'Phase 1 reported':20s}:     {budget['central']['sigma_total_phase1_mev']:.4f} MeV")
diff = abs(budget["quadrature_sum_mev"] - budget["central"]["sigma_total_phase1_mev"])
if diff < 0.05:
    print("  Match: YES")
else:
    print(f"  Match: NO (diff = {diff:.4f} MeV)")

# [5] F_alg vs F_geo
print("\n[5] F_alg vs F_geo:")
fvf = inp["F_exact_vs_approximate"]
print(f"  F_alg:    {fvf['F_alg']:.6f}")
print(f"  F_geo:    {fvf['F_geo']:.6f}")
print(f"  Diff F^2: {fvf['diff_F2_mev']:.4f} MeV")
print(f"  Exceeds lepton budget: {'YES' if fvf['exceeds_lepton_budget'] else 'NO'}")

print("\n" + "=" * 50)
