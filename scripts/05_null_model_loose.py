#!/usr/bin/env python3
"""Loose null model: does F^n match ANY quark mass, for any small power?

Tests F^n for n in {1,2,3} against {u,d,s,c,b} at the natural scale.
Restricts to mu* > 1 GeV where perturbative QCD is valid.
"""

import json
import sys
import time
import os
import numpy as np
from scipy.interpolate import interp1d

from koide_soddy.null_model import (
    sample_koide_triples, compute_natural_scale, clopper_pearson_ci,
)
from koide_soddy.leptons import F_from_soddy
from koide_soddy.running import run_ms_to_scale, run_mc_to_scale, run_mb_to_scale

N_SAMPLES = 100_000
SEED = 42

MU_MIN_MEV = 1000.0
MU_MAX_MEV = 50_000.0

# Quark inputs at reference scales (MeV)
M_U_2GEV = 2.16;  DM_U = 0.07
M_D_2GEV = 4.70;  DM_D = 0.07
M_S_2GEV = 93.5;  DM_S = 0.8
M_C_MC = 1273.0;  DM_C = 4.6
M_B_MB = 4183.0;  DM_B = 7.0

QUARKS = ["u", "d", "s", "c", "b"]
POWERS = [1, 2, 3]

print(f"Loose null model: sampling {N_SAMPLES} Koide triples...", flush=True)
t0 = time.time()

# Step 1: Sample and filter
triples = sample_koide_triples(N_SAMPLES, seed=SEED)
n_total = len(triples)
mu_values = compute_natural_scale(triples)

mask = (mu_values >= MU_MIN_MEV) & (mu_values <= MU_MAX_MEV)
n_testable = int(mask.sum())
print(f"  {n_testable} testable triples ({100*n_testable/n_total:.1f}%)", flush=True)

triples_test = triples[mask]
mu_test = mu_values[mask]

# Compute F for each triple
F_values = np.array([F_from_soddy(float(m1), float(m2), float(m3))
                     for m1, m2, m3 in triples_test])

# Step 2: Build running caches for all 5 quarks
print("  Building running caches for all quarks...", flush=True)

n_grid = 300
log_mu_grid = np.linspace(np.log(MU_MIN_MEV), np.log(MU_MAX_MEV), n_grid)
mu_grid = np.exp(log_mu_grid)

# Suppress C-level stderr
stderr_fd = sys.stderr.fileno()
old_stderr = os.dup(stderr_fd)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, stderr_fd)

quark_cache = {}
for q, m_ref, dm_ref, runner in [
    ("u", M_U_2GEV, DM_U, lambda m, mu: run_ms_to_scale(m, mu)),
    ("d", M_D_2GEV, DM_D, lambda m, mu: run_ms_to_scale(m, mu)),
    ("s", M_S_2GEV, DM_S, lambda m, mu: run_ms_to_scale(m, mu)),
    ("c", M_C_MC,   DM_C, lambda m, mu: run_mc_to_scale(m, mu)),
    ("b", M_B_MB,   DM_B, lambda m, mu: run_mb_to_scale(m, mu)),
]:
    m_grid = np.empty(n_grid)
    sigma_grid = np.empty(n_grid)
    for i, mu in enumerate(mu_grid):
        m_grid[i] = runner(m_ref, float(mu))
        m_up = runner(m_ref + dm_ref, float(mu))
        m_dn = runner(m_ref - dm_ref, float(mu))
        sigma_grid[i] = (m_up - m_dn) / 2.0
    quark_cache[q] = (
        interp1d(log_mu_grid, m_grid, kind='cubic'),
        interp1d(log_mu_grid, sigma_grid, kind='cubic'),
    )

# Restore stderr
os.dup2(old_stderr, stderr_fd)
os.close(devnull)
os.close(old_stderr)

print(f"  Caches built in {time.time()-t0:.1f}s", flush=True)

# Step 3: Test all (quark, power) combinations
per_quark_hits = {q: 0 for q in QUARKS}
per_power_hits = {str(n): 0 for n in POWERS}
joint_hits = np.zeros((len(QUARKS), len(POWERS)), dtype=int)
overall_hits = 0

for i in range(n_testable):
    log_mu = np.log(mu_test[i])
    F = F_values[i]
    hit_this_triple = False

    for qi, q in enumerate(QUARKS):
        m_interp, s_interp = quark_cache[q]
        mq = float(m_interp(log_mu))
        sigma_q = float(s_interp(log_mu))
        if sigma_q <= 0:
            continue

        for ni, n in enumerate(POWERS):
            Fn = F ** n
            if abs(Fn - mq) < sigma_q:
                joint_hits[qi, ni] += 1
                per_quark_hits[q] += 1
                per_power_hits[str(n)] += 1
                hit_this_triple = True

    if hit_this_triple:
        overall_hits += 1

overall_frac = overall_hits / n_testable if n_testable > 0 else 0.0
ci_lo, ci_hi = clopper_pearson_ci(overall_hits, n_testable)

elapsed = time.time() - t0

result = {
    "n_samples": N_SAMPLES,
    "n_valid_triples": n_testable,
    "overall_hits": overall_hits,
    "overall_hit_fraction": round(overall_frac, 6),
    "overall_hit_fraction_95ci": [round(ci_lo, 6), round(ci_hi, 6)],
    "per_quark_hits": {q: int(per_quark_hits[q]) for q in QUARKS},
    "per_power_hits": {str(n): int(per_power_hits[str(n)]) for n in POWERS},
    "joint_hit_table": joint_hits.tolist(),
    "joint_hit_table_rows": QUARKS,
    "joint_hit_table_cols": [str(n) for n in POWERS],
    "runtime_seconds": round(elapsed, 1),
    "seed": SEED,
}

with open("results/null_loose.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}", flush=True)
print(f"LOOSE NULL RESULT", flush=True)
print(f"{'='*60}", flush=True)
print(f"N testable:         {n_testable}")
print(f"Overall hits:       {overall_hits}")
print(f"Overall hit frac:   {overall_frac:.4f} ({100*overall_frac:.2f}%)")
print(f"95% CI:             [{ci_lo:.6f}, {ci_hi:.6f}]")
print()
print("Per-quark hits:")
for q in QUARKS:
    frac = per_quark_hits[q] / n_testable if n_testable > 0 else 0
    print(f"  {q}: {per_quark_hits[q]:>6d} ({100*frac:.2f}%)")
print()
print("Per-power hits:")
for n in POWERS:
    frac = per_power_hits[str(n)] / n_testable if n_testable > 0 else 0
    print(f"  F^{n}: {per_power_hits[str(n)]:>6d} ({100*frac:.2f}%)")
print()
print("Joint hit table (rows=quarks, cols=powers):")
print(f"       {'F^1':>8s} {'F^2':>8s} {'F^3':>8s}")
for qi, q in enumerate(QUARKS):
    print(f"  {q:>3s}: {joint_hits[qi,0]:>8d} {joint_hits[qi,1]:>8d} {joint_hits[qi,2]:>8d}")
print(f"\nRuntime: {elapsed:.1f}s")
print(f"Output: results/null_loose.json")
