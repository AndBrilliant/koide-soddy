#!/usr/bin/env python3
"""Tight null model: does F^2 match m_s at the natural scale, by chance?

Restricts comparison to mu* > 1 GeV where perturbative QCD is valid.
Triples with mu* < 1 GeV are excluded since perturbative running is
unreliable near Lambda_QCD.
"""

import json
import sys
import time
import os
import numpy as np
from scipy.interpolate import interp1d

from koide_soddy.null_model import (
    sample_koide_triples, compute_F_squared, compute_natural_scale,
    clopper_pearson_ci,
)
from koide_soddy.running import run_ms_to_scale

N_SAMPLES = 100_000
SEED = 42

# Perturbative QCD cutoff: only compare at scales above this
MU_MIN_MEV = 1000.0
MU_MAX_MEV = 50_000.0

# Quark inputs
M_S_2GEV = 93.44  # FLAG 2024 Table 11, Nf=2+1+1 (arXiv:2411.04268)
DM_S_2GEV = 0.68  # FLAG 2024 Table 11 uncertainty

print(f"Tight null model: sampling {N_SAMPLES} Koide triples...", flush=True)
t0 = time.time()

# Step 1: Sample triples
triples = sample_koide_triples(N_SAMPLES, seed=SEED)
n_total = len(triples)
print(f"  Generated {n_total} valid triples in {time.time()-t0:.1f}s", flush=True)

# Step 2: Compute F^2 and natural scales
F2_values = compute_F_squared(triples)
mu_values = compute_natural_scale(triples)

# Filter to perturbative regime
mask = (mu_values >= MU_MIN_MEV) & (mu_values <= MU_MAX_MEV)
n_testable = int(mask.sum())
print(f"  {n_testable} triples with mu* in [{MU_MIN_MEV:.0f}, {MU_MAX_MEV:.0f}] MeV "
      f"({100*n_testable/n_total:.1f}%)", flush=True)

F2_test = F2_values[mask]
mu_test = mu_values[mask]

# Step 3: Build cached running function on a safe grid
print("  Building running cache...", flush=True)

n_grid = 300
log_mu_grid = np.linspace(np.log(MU_MIN_MEV), np.log(MU_MAX_MEV), n_grid)
mu_grid = np.exp(log_mu_grid)

ms_grid = np.empty(n_grid)
sigma_ms_grid = np.empty(n_grid)

# Suppress C-level stderr from rundec warnings during cache build
stderr_fd = sys.stderr.fileno()
old_stderr = os.dup(stderr_fd)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, stderr_fd)

for i, mu in enumerate(mu_grid):
    ms_grid[i] = run_ms_to_scale(M_S_2GEV, float(mu))
    ms_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, float(mu))
    ms_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, float(mu))
    sigma_ms_grid[i] = (ms_up - ms_dn) / 2.0

# Restore stderr
os.dup2(old_stderr, stderr_fd)
os.close(devnull)
os.close(old_stderr)

ms_interp = interp1d(log_mu_grid, ms_grid, kind='cubic')
sigma_interp = interp1d(log_mu_grid, sigma_ms_grid, kind='cubic')

print(f"  Cache built in {time.time()-t0:.1f}s", flush=True)

# Step 4: Count hits
n_hits = 0
residuals_sigma = np.empty(n_testable)

for i in range(n_testable):
    log_mu = np.log(mu_test[i])
    ms_mu = float(ms_interp(log_mu))
    sigma_ms_mu = float(sigma_interp(log_mu))

    residuals_sigma[i] = (F2_test[i] - ms_mu) / sigma_ms_mu
    if abs(F2_test[i] - ms_mu) < sigma_ms_mu:
        n_hits += 1

hit_fraction = n_hits / n_testable if n_testable > 0 else 0.0
ci_lo, ci_hi = clopper_pearson_ci(n_hits, n_testable)

elapsed = time.time() - t0

# Distribution summaries
ms_at_mu = np.array([float(ms_interp(np.log(mu))) for mu in mu_test])

result = {
    "n_samples": N_SAMPLES,
    "n_valid_triples": n_testable,
    "n_excluded_below_mu_min": int((mu_values < MU_MIN_MEV).sum()),
    "n_excluded_above_mu_max": int((mu_values > MU_MAX_MEV).sum()),
    "mu_min_mev": MU_MIN_MEV,
    "mu_max_mev": MU_MAX_MEV,
    "n_hits": n_hits,
    "hit_fraction": round(hit_fraction, 6),
    "hit_fraction_95ci": [round(ci_lo, 6), round(ci_hi, 6)],
    "F2_distribution_summary": {
        "mean": round(float(np.mean(F2_test)), 4),
        "median": round(float(np.median(F2_test)), 4),
        "std": round(float(np.std(F2_test)), 4),
        "p05": round(float(np.percentile(F2_test, 5)), 4),
        "p95": round(float(np.percentile(F2_test, 95)), 4),
    },
    "ms_at_natural_scale_distribution_summary": {
        "mean": round(float(np.mean(ms_at_mu)), 4),
        "median": round(float(np.median(ms_at_mu)), 4),
        "std": round(float(np.std(ms_at_mu)), 4),
        "p05": round(float(np.percentile(ms_at_mu, 5)), 4),
        "p95": round(float(np.percentile(ms_at_mu, 95)), 4),
    },
    "runtime_seconds": round(elapsed, 1),
    "seed": SEED,
}

with open("results/null_tight.json", "w") as f:
    json.dump(result, f, indent=2)

np.save("results/null_tight_residuals.npy", residuals_sigma)

print(f"\n{'='*60}", flush=True)
print(f"TIGHT NULL RESULT", flush=True)
print(f"{'='*60}", flush=True)
print(f"N samples:     {N_SAMPLES}")
print(f"N testable:    {n_testable} ({100*n_testable/n_total:.1f}% of total)")
print(f"N hits:        {n_hits}")
print(f"Hit fraction:  {hit_fraction:.6f}")
print(f"95% CI:        [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"Runtime:       {elapsed:.1f}s")
print(f"\nF^2 distribution (testable): mean={np.mean(F2_test):.1f}, median={np.median(F2_test):.1f}, std={np.std(F2_test):.1f}")
print(f"m_s(mu*) dist (testable):    mean={np.mean(ms_at_mu):.1f}, median={np.median(ms_at_mu):.1f}, std={np.std(ms_at_mu):.1f}")
print(f"\nOutput: results/null_tight.json")
