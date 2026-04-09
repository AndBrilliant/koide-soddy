"""Scale alternatives: test m_s at multiple lepton-derived scale prescriptions."""

import json
import math
import sys
import os

from koide_soddy.leptons import (
    M_E_MEV, M_MU_MEV, M_TAU_MEV, F_squared_central,
)
from koide_soddy.running import run_ms_to_scale

M_S_2GEV = 93.44  # FLAG 2024 Table 11, Nf=2+1+1, MSbar at 2 GeV (arXiv:2411.04268)
DM_S_2GEV = 0.68  # FLAG 2024 Table 11 uncertainty

# Minimum scale for reliable perturbative running (MeV)
PERTURBATIVE_MIN = 700.0


def _compute_scales() -> list[dict]:
    """Define all lepton-derived scale prescriptions."""
    me, mmu, mtau = M_E_MEV, M_MU_MEV, M_TAU_MEV
    return [
        {"label": "sum", "definition": "m_e + m_mu + m_tau",
         "mu_mev": me + mmu + mtau},
        {"label": "tau_only", "definition": "m_tau",
         "mu_mev": mtau},
        {"label": "mean", "definition": "(m_e + m_mu + m_tau) / 3",
         "mu_mev": (me + mmu + mtau) / 3.0},
        {"label": "geo_mean_2", "definition": "sqrt(m_mu * m_tau)",
         "mu_mev": math.sqrt(mmu * mtau)},
        {"label": "geo_mean_3", "definition": "(m_e * m_mu * m_tau)^(1/3)",
         "mu_mev": (me * mmu * mtau) ** (1.0 / 3.0)},
        {"label": "mu_plus_tau", "definition": "m_mu + m_tau",
         "mu_mev": mmu + mtau},
        {"label": "2gev_ref", "definition": "2000 (conventional reference)",
         "mu_mev": 2000.0},
        {"label": "harmonic", "definition": "3 / (1/m_e + 1/m_mu + 1/m_tau)",
         "mu_mev": 3.0 / (1.0 / me + 1.0 / mmu + 1.0 / mtau)},
    ]


def run_scale_alternatives(output_path: str = "results/scale_alternatives.json") -> dict:
    """Compute m_s at each lepton-derived scale and compare to F^2."""
    F2 = F_squared_central()
    scales = _compute_scales()
    results = []

    # Suppress rundec warnings
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)

    for s in scales:
        mu = s["mu_mev"]
        entry = {
            "label": s["label"],
            "definition": s["definition"],
            "mu_mev": round(mu, 4),
        }

        if mu < PERTURBATIVE_MIN:
            entry.update({
                "ms_mu_mev": None,
                "sigma_ms_mev": None,
                "residual_mev": None,
                "residual_sigma": None,
                "hit": False,
                "valid": False,
                "note": f"Scale {mu:.2f} MeV is below {PERTURBATIVE_MIN} MeV; "
                        f"perturbative running unreliable",
            })
        else:
            ms = run_ms_to_scale(M_S_2GEV, mu)
            ms_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, mu)
            ms_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, mu)
            sigma = (ms_up - ms_dn) / 2.0

            residual = F2 - ms
            residual_sigma = residual / sigma if sigma > 0 else float('inf')
            hit = abs(residual) < sigma

            entry.update({
                "ms_mu_mev": round(ms, 4),
                "sigma_ms_mev": round(sigma, 4),
                "residual_mev": round(residual, 4),
                "residual_sigma": round(residual_sigma, 4),
                "hit": hit,
                "valid": True,
            })

        results.append(entry)

    # Restore stderr
    os.dup2(old_stderr, stderr_fd)
    os.close(devnull)
    os.close(old_stderr)

    hit_labels = [r["label"] for r in results if r["hit"]]

    output = {
        "F_squared_mev": round(F2, 4),
        "scales": results,
        "n_hits": len(hit_labels),
        "hit_labels": hit_labels,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
