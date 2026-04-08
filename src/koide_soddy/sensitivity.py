"""Scale and input sensitivity analyses."""

import json
import numpy as np

from koide_soddy.leptons import (
    SUM_LEPTON_MEV, F_squared_central, F_uncertainty_from_leptons,
)
from koide_soddy.running import run_ms_to_scale

# Quark mass inputs
M_S_2GEV = 93.5
DM_S_2GEV = 0.8


def scale_sensitivity(output_path: str = "results/scale_sensitivity.json") -> dict:
    """Compute m_s(mu) for a range of scales around sum(m_lepton).

    Returns the full results dict and writes it to output_path.
    """
    F2 = F_squared_central()
    mu_star = SUM_LEPTON_MEV

    # Coarse grid (0.5–4.0 in 0.1 steps) plus fine grid near 1.0 (0.01 steps)
    coarse = np.arange(0.5, 4.05, 0.1)
    fine = np.arange(0.85, 1.16, 0.01)
    multipliers = np.unique(np.sort(np.concatenate([coarse, fine])))
    rows = []

    for mult in multipliers:
        mu = mult * mu_star
        ms = run_ms_to_scale(M_S_2GEV, mu)
        ms_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, mu)
        ms_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, mu)
        sigma_ms = (ms_up - ms_dn) / 2.0

        residual_mev = F2 - ms
        residual_sigma = residual_mev / sigma_ms if sigma_ms > 0 else float('inf')

        rows.append({
            "scale_multiplier": round(float(mult), 2),
            "mu_mev": round(mu, 2),
            "ms_mu_mev": round(ms, 4),
            "ms_uncertainty_mev": round(sigma_ms, 4),
            "residual_mev": round(residual_mev, 4),
            "residual_sigma": round(residual_sigma, 4),
        })

    # Find 1-sigma and 2-sigma windows
    one_sigma = [r for r in rows if abs(r["residual_sigma"]) < 1.0]
    two_sigma = [r for r in rows if abs(r["residual_sigma"]) < 2.0]

    one_sigma_window = (
        [min(r["mu_mev"] for r in one_sigma), max(r["mu_mev"] for r in one_sigma)]
        if one_sigma else None
    )
    two_sigma_window = (
        [min(r["mu_mev"] for r in two_sigma), max(r["mu_mev"] for r in two_sigma)]
        if two_sigma else None
    )

    result = {
        "F_squared_mev": round(F2, 4),
        "mu_star_mev": round(mu_star, 4),
        "data": rows,
        "one_sigma_window_mev": one_sigma_window,
        "two_sigma_window_mev": two_sigma_window,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def input_sensitivity(output_path: str = "results/input_sensitivity.json") -> dict:
    """Full error budget on F^2 - m_s(sum_lepton).

    Combines lepton-side uncertainty (from F) with quark-side
    uncertainty (from m_s lattice input propagated through running).
    """
    F2 = F_squared_central()
    mu = SUM_LEPTON_MEV
    ms_mu = run_ms_to_scale(M_S_2GEV, mu)

    # Lepton-side uncertainty on F^2
    _, sigma_F2_lepton = F_uncertainty_from_leptons()

    # Quark-side: propagate m_s(2 GeV) uncertainty through running
    ms_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, mu)
    ms_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, mu)
    sigma_ms_mu = (ms_up - ms_dn) / 2.0

    # Total uncertainty on (F^2 - m_s): add in quadrature
    sigma_total = np.sqrt(sigma_F2_lepton**2 + sigma_ms_mu**2)

    residual = F2 - ms_mu

    result = {
        "F_squared_mev": round(F2, 4),
        "ms_at_mu_mev": round(ms_mu, 4),
        "residual_mev": round(residual, 4),
        "error_budget": {
            "sigma_F2_lepton_mev": round(sigma_F2_lepton, 6),
            "sigma_ms_running_mev": round(sigma_ms_mu, 4),
            "sigma_total_mev": round(sigma_total, 4),
            "lepton_fraction_pct": round(100 * sigma_F2_lepton**2 / sigma_total**2, 2),
            "quark_fraction_pct": round(100 * sigma_ms_mu**2 / sigma_total**2, 2),
        },
        "residual_in_sigma": round(residual / sigma_total, 4),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
