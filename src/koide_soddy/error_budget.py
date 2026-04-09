"""Error budget decomposition for m_s at the lepton sum scale."""

import json
import math
import sys
import os

import rundec

from koide_soddy.leptons import SUM_LEPTON_MEV
from koide_soddy.running import run_ms_to_scale, ALPHA_S_MZ, M_C_AT_MC_GEV, NLOOPS

M_S_2GEV = 93.44  # FLAG 2024 Table 11, Nf=2+1+1, MSbar at 2 GeV (arXiv:2411.04268)
DM_S_2GEV = 0.68  # FLAG 2024 Table 11 uncertainty
DM_ALPHA_S = 0.0009
DM_MC_GEV = 0.0046


def _run_ms_with_nloops(ms_mev: float, mu_mev: float, alpha_s_mz: float,
                        nloops: int) -> float:
    """Run m_s from 2 GeV to mu using specified loop order."""
    crd = rundec.CRunDec()
    mb = 4.183  # GeV
    mc = 1.273  # GeV

    # alpha_s at 2 GeV: M_Z -> m_b (nf=5) -> decouple -> m_c (nf=4) -> decouple -> nf=3
    as_mb_5 = crd.AlphasExact(alpha_s_mz, 91.1876, mb, 5, nloops)
    as_mb_4 = crd.DecAsDownMS(as_mb_5, mb, mb, 4, nloops)

    mu_gev = mu_mev / 1000.0
    ms_gev = ms_mev / 1000.0

    if mu_gev >= mc:
        # nf=4 region: run alpha_s from m_b to 2 GeV, then run m_s
        as_2gev = crd.AlphasExact(as_mb_4, mb, 2.0, 4, nloops)
        result = crd.AsmMSrunexact(ms_gev, as_2gev, 2.0, mu_gev, 4, nloops)
        return result.mMSexact * 1000.0
    else:
        # Need to cross m_c threshold
        as_2gev = crd.AlphasExact(as_mb_4, mb, 2.0, 4, nloops)
        result_mc = crd.AsmMSrunexact(ms_gev, as_2gev, 2.0, mc, 4, nloops)
        ms_mc = result_mc.mMSexact
        as_mc = result_mc.Asexact
        ms_mc_3 = crd.DecMqDownMS(ms_mc, as_mc, mc, mc, 3, nloops)
        as_mc_3 = crd.DecAsDownMS(as_mc, mc, mc, 3, nloops)
        result = crd.AsmMSrunexact(ms_mc_3, as_mc_3, mc, mu_gev, 3, nloops)
        return result.mMSexact * 1000.0


def run_error_budget(output_path: str = "results/error_budget.json") -> dict:
    """Decompose uncertainty on m_s(mu*) by source."""
    mu = SUM_LEPTON_MEV

    # Suppress rundec warnings
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)

    # Central value
    ms_central = run_ms_to_scale(M_S_2GEV, mu)

    # 1. Lattice m_s input
    ms_lat_up = run_ms_to_scale(M_S_2GEV + DM_S_2GEV, mu)
    ms_lat_dn = run_ms_to_scale(M_S_2GEV - DM_S_2GEV, mu)
    sigma_lat = (ms_lat_up - ms_lat_dn) / 2.0

    # 2. alpha_s(M_Z)
    ms_as_up = run_ms_to_scale(M_S_2GEV, mu, alpha_s_mz=ALPHA_S_MZ + DM_ALPHA_S)
    ms_as_dn = run_ms_to_scale(M_S_2GEV, mu, alpha_s_mz=ALPHA_S_MZ - DM_ALPHA_S)
    sigma_as = (ms_as_up - ms_as_dn) / 2.0

    # 3. Charm threshold — need to modify M_C_AT_MC_GEV in running
    # Run manually with shifted m_c
    from koide_soddy.running import _alpha_s_at_scale, _run_mass_between_scales
    import rundec as rd

    def _run_with_mc(mc_gev):
        crd = rd.CRunDec()
        ms_gev = M_S_2GEV / 1000.0
        mu_gev = mu / 1000.0
        mb = 4.183

        # alpha_s at 2 GeV with this m_c
        as_mb_5 = crd.AlphasExact(ALPHA_S_MZ, 91.1876, mb, 5, 4)
        as_mb_4 = crd.DecAsDownMS(as_mb_5, mb, mb, 4, 4)
        as_mc_4 = crd.AlphasExact(as_mb_4, mb, mc_gev, 4, 4)
        as_mc_3 = crd.DecAsDownMS(as_mc_4, mc_gev, mc_gev, 3, 4)
        as_2gev = crd.AlphasExact(as_mc_3, mc_gev, 2.0, 3, 4)

        # Wait — 2 GeV > m_c, so nf=4 at 2 GeV. Need different path.
        # alpha_s(2 GeV, nf=4) from m_b
        as_2gev_4 = crd.AlphasExact(as_mb_4, mb, 2.0, 4, 4)
        result = crd.AsmMSrunexact(ms_gev, as_2gev_4, 2.0, mu_gev, 4, 4)
        return result.mMSexact * 1000.0

    ms_mc_up = _run_with_mc(M_C_AT_MC_GEV + DM_MC_GEV)
    ms_mc_dn = _run_with_mc(M_C_AT_MC_GEV - DM_MC_GEV)
    sigma_mc = (ms_mc_up - ms_mc_dn) / 2.0

    # 4. Truncation: four-loop vs three-loop
    ms_4loop = _run_ms_with_nloops(M_S_2GEV, mu, ALPHA_S_MZ, 4)
    ms_3loop = _run_ms_with_nloops(M_S_2GEV, mu, ALPHA_S_MZ, 3)
    sigma_trunc = abs(ms_4loop - ms_3loop)

    # Restore stderr
    os.dup2(old_stderr, stderr_fd)
    os.close(devnull)
    os.close(old_stderr)

    # Quadrature sum
    quad_sum = math.sqrt(sigma_lat**2 + sigma_as**2 + sigma_mc**2 + sigma_trunc**2)

    # Phase 1 total was computed as (ms_up - ms_dn)/2 varying only m_s input
    phase1_total = sigma_lat  # Phase 1 only varied m_s, so its "total" = lattice component

    output = {
        "central": {
            "ms_mu_star_mev": round(ms_central, 4),
            "mu_star_mev": round(mu, 4),
            "sigma_total_phase1_mev": round(sigma_lat, 4),
        },
        "components": [
            {"source": "lattice_ms_input",
             "shift_up_mev": round(ms_lat_up - ms_central, 4),
             "shift_dn_mev": round(ms_lat_dn - ms_central, 4),
             "sigma_contribution_mev": round(sigma_lat, 4)},
            {"source": "alpha_s",
             "shift_up_mev": round(ms_as_up - ms_central, 4),
             "shift_dn_mev": round(ms_as_dn - ms_central, 4),
             "sigma_contribution_mev": round(sigma_as, 4)},
            {"source": "mc_threshold",
             "shift_up_mev": round(ms_mc_up - ms_central, 4),
             "shift_dn_mev": round(ms_mc_dn - ms_central, 4),
             "sigma_contribution_mev": round(sigma_mc, 4)},
            {"source": "loop_truncation",
             "shift_mev": round(ms_4loop - ms_3loop, 4),
             "sigma_contribution_mev": round(sigma_trunc, 4)},
        ],
        "quadrature_sum_mev": round(quad_sum, 4),
        "matches_phase1_total": abs(quad_sum - sigma_lat) < 0.05
                                if sigma_as < 0.1 and sigma_mc < 0.1
                                else False,
        "note": ("Phase 1 only varied m_s(2 GeV); this decomposition adds "
                 "alpha_s, m_c threshold, and truncation components."),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
