"""Filter sensitivity: re-run tight null with different mu* lower cutoffs."""

import json
import numpy as np

from koide_soddy.null_model import sample_koide_triples
from koide_soddy.prior_sensitivity import (
    _build_ms_cache, _run_tight_null_on_triples,
    MU_MAX_MEV,
)

CUTOFFS_MEV = [700.0, 1000.0, 1500.0, 2000.0]


def run_filter_sensitivity(n_samples: int = 100_000, seed: int = 42,
                           output_path: str = "results/filter_sensitivity.json") -> dict:
    """Run tight null with multiple mu* lower cutoffs."""
    # Build cache covering the widest range needed
    mu_min_cache = min(CUTOFFS_MEV)
    ms_interp, sigma_interp = _build_ms_cache(mu_min=mu_min_cache, mu_max=MU_MAX_MEV)

    # Sample once, reuse for all cutoffs
    triples = sample_koide_triples(n_samples, seed=seed)

    results = []
    for mu_min in CUTOFFS_MEV:
        r = _run_tight_null_on_triples(
            triples, ms_interp, sigma_interp,
            mu_min=mu_min, mu_max=MU_MAX_MEV,
        )
        entry = {
            "mu_min_mev": mu_min,
            "n_valid": r["n_valid"],
            "n_hits": r["n_hits"],
            "hit_fraction": r["hit_fraction"],
            "ci_95": r["ci_95"],
        }
        if mu_min < 1000.0:
            entry["note"] = "below 1 GeV; running extrapolated"
        results.append(entry)

    # Stability check
    fracs = [r["hit_fraction"] for r in results if r["hit_fraction"] > 0]
    if len(fracs) >= 2:
        stable = max(fracs) / min(fracs) < 2.0
    else:
        stable = True

    output = {
        "cutoffs": results,
        "stable": stable,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
