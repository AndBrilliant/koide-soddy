"""Prior sensitivity: re-run tight null with alternative sampler priors."""

import math
import json
import sys
import os
import numpy as np
from scipy.interpolate import interp1d

from koide_soddy.null_model import (
    _solve_m2_for_koide, compute_F_squared, compute_natural_scale,
    clopper_pearson_ci,
)
from koide_soddy.running import run_ms_to_scale

M_S_2GEV = 93.5
DM_S_2GEV = 0.8
MU_MIN_MEV = 1000.0
MU_MAX_MEV = 50_000.0


def _sample_with_prior(n: int, seed: int,
                       m3_range: tuple[float, float],
                       ratio_range: tuple[float, float],
                       log_sampling: bool = True) -> np.ndarray:
    """Generic Koide triple sampler with configurable prior.

    Parameters
    ----------
    m3_range : (lo, hi) for m3 in MeV
    ratio_range : (lo, hi) for m1/m3 ratio
    log_sampling : if True, sample log-uniformly; if False, linear-uniformly
    """
    rng = np.random.default_rng(seed)
    triples = []

    while len(triples) < n:
        batch_size = min(n - len(triples), 10000) * 2

        if log_sampling:
            log_m3 = rng.uniform(math.log(m3_range[0]), math.log(m3_range[1]), batch_size)
            m3_batch = np.exp(log_m3)
            log_ratio = rng.uniform(math.log(ratio_range[0]), math.log(ratio_range[1]), batch_size)
            m1_batch = np.exp(log_ratio) * m3_batch
        else:
            m3_batch = rng.uniform(m3_range[0], m3_range[1], batch_size)
            ratio_batch = rng.uniform(ratio_range[0], ratio_range[1], batch_size)
            m1_batch = ratio_batch * m3_batch

        for m1, m3 in zip(m1_batch, m3_batch):
            if len(triples) >= n:
                break
            m2 = _solve_m2_for_koide(float(m1), float(m3))
            if m2 is None:
                continue
            s = m1 + m2 + m3
            sr = math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3)
            Q = s / (sr * sr)
            if abs(Q - 2.0 / 3.0) < 1e-5:
                triples.append((m1, m2, m3))

    return np.array(triples[:n])


# Prior definitions
PRIORS = {
    "A_baseline": {
        "m3_range": (1.0, 1e4),
        "ratio_range": (1e-3, 1e-1),
        "log_sampling": True,
        "description": "Phase 1 baseline: m3 log-U[1, 10^4], ratio log-U[10^-3, 10^-1]",
    },
    "B_wider_range": {
        "m3_range": (0.1, 1e5),
        "ratio_range": (1e-3, 1e-1),
        "log_sampling": True,
        "description": "Wider m3: log-U[10^-1, 10^5], same ratio",
    },
    "C_smaller_ratio": {
        "m3_range": (1.0, 1e4),
        "ratio_range": (1e-4, 1e-2),
        "log_sampling": True,
        "description": "More hierarchical: m3 log-U[1, 10^4], ratio log-U[10^-4, 10^-2]",
    },
    "D_linear": {
        "m3_range": (10.0, 10000.0),
        "ratio_range": (0.001, 0.1),
        "log_sampling": False,
        "description": "Linear-uniform stress test: m3 U[10, 10^4], ratio U[10^-3, 10^-1]",
    },
}


def _build_ms_cache(mu_min: float = MU_MIN_MEV, mu_max: float = MU_MAX_MEV,
                    n_grid: int = 300):
    """Build interpolation cache for m_s running and its uncertainty."""
    log_mu_grid = np.linspace(np.log(mu_min), np.log(mu_max), n_grid)
    mu_grid = np.exp(log_mu_grid)

    ms_grid = np.empty(n_grid)
    sigma_grid = np.empty(n_grid)

    # Suppress C-level stderr from rundec
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)

    for i, mu in enumerate(mu_grid):
        ms_grid[i] = run_ms_to_scale(M_S_2GEV, float(mu))
        ms_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, float(mu))
        ms_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, float(mu))
        sigma_grid[i] = (ms_up - ms_dn) / 2.0

    os.dup2(old_stderr, stderr_fd)
    os.close(devnull)
    os.close(old_stderr)

    ms_interp = interp1d(log_mu_grid, ms_grid, kind='cubic')
    sigma_interp = interp1d(log_mu_grid, sigma_grid, kind='cubic')
    return ms_interp, sigma_interp


def _run_tight_null_on_triples(triples: np.ndarray, ms_interp, sigma_interp,
                                mu_min: float = MU_MIN_MEV,
                                mu_max: float = MU_MAX_MEV) -> dict:
    """Run the tight null test on a set of triples. Returns result dict."""
    F2_values = compute_F_squared(triples)
    mu_values = compute_natural_scale(triples)

    mask = (mu_values >= mu_min) & (mu_values <= mu_max)
    n_testable = int(mask.sum())

    if n_testable == 0:
        return {"n_valid": 0, "n_hits": 0, "hit_fraction": 0.0,
                "ci_95": [0.0, 1.0]}

    F2_test = F2_values[mask]
    mu_test = mu_values[mask]

    n_hits = 0
    for i in range(n_testable):
        log_mu = np.log(mu_test[i])
        ms_mu = float(ms_interp(log_mu))
        sigma_ms = float(sigma_interp(log_mu))
        if sigma_ms > 0 and abs(F2_test[i] - ms_mu) < sigma_ms:
            n_hits += 1

    hit_frac = n_hits / n_testable
    ci_lo, ci_hi = clopper_pearson_ci(n_hits, n_testable)
    return {
        "n_valid": n_testable,
        "n_hits": n_hits,
        "hit_fraction": round(hit_frac, 6),
        "ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
    }


def run_prior_sensitivity(n_samples: int = 100_000, seed: int = 42,
                          output_path: str = "results/prior_sensitivity.json") -> dict:
    """Run the tight null with all four alternative priors."""
    ms_interp, sigma_interp = _build_ms_cache()

    results = {}
    for name, cfg in PRIORS.items():
        triples = _sample_with_prior(
            n_samples, seed=seed,
            m3_range=cfg["m3_range"],
            ratio_range=cfg["ratio_range"],
            log_sampling=cfg["log_sampling"],
        )
        r = _run_tight_null_on_triples(triples, ms_interp, sigma_interp)
        r["description"] = cfg["description"]
        results[name] = r

    # Summary
    fracs = [r["hit_fraction"] for r in results.values() if r["hit_fraction"] > 0]
    min_f = min(fracs) if fracs else 0.0
    max_f = max(fracs) if fracs else 0.0
    fold = max_f / min_f if min_f > 0 else float('inf')

    output = {
        "n_samples_per_prior": n_samples,
        "priors": results,
        "summary": {
            "min_hit_fraction": round(min_f, 6),
            "max_hit_fraction": round(max_f, 6),
            "fold_variation": round(fold, 2),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
