"""QCD running of quark masses using rundec (four-loop)."""

try:
    import rundec
except ImportError:
    raise ImportError(
        "rundec is required but not installed. Install it with:\n"
        "  pip install rundec\n"
        "See https://github.com/DavidMStraub/rundec-python"
    )

# QCD inputs
ALPHA_S_MZ = 0.1180
M_Z_GEV = 91.1876
M_C_AT_MC_GEV = 1.2730
M_B_AT_MB_GEV = 4.183

NLOOPS = 4  # four-loop running throughout


def _alpha_s_at_scale(scale_gev: float, alpha_s_mz: float) -> tuple[float, int]:
    """Compute alpha_s at the given scale with proper threshold matching.

    Runs alpha_s from M_Z (nf=5) through m_b and m_c thresholds.
    Returns (alpha_s, nf) at the requested scale.
    """
    crd = rundec.CRunDec()
    mb = M_B_AT_MB_GEV
    mc = M_C_AT_MC_GEV

    if scale_gev >= mb:
        # nf=5 region: run directly from M_Z
        alpha_s = crd.AlphasExact(alpha_s_mz, M_Z_GEV, scale_gev, 5, NLOOPS)
        return alpha_s, 5

    # Run from M_Z to m_b in nf=5
    as_mb_5 = crd.AlphasExact(alpha_s_mz, M_Z_GEV, mb, 5, NLOOPS)
    # Decouple nf=5 -> nf=4 at m_b
    as_mb_4 = crd.DecAsDownMS(as_mb_5, mb, mb, 4, NLOOPS)

    if scale_gev >= mc:
        # nf=4 region
        alpha_s = crd.AlphasExact(as_mb_4, mb, scale_gev, 4, NLOOPS)
        return alpha_s, 4

    # Run from m_b to m_c in nf=4
    as_mc_4 = crd.AlphasExact(as_mb_4, mb, mc, 4, NLOOPS)
    # Decouple nf=4 -> nf=3 at m_c
    as_mc_3 = crd.DecAsDownMS(as_mc_4, mc, mc, 3, NLOOPS)

    # nf=3 region
    alpha_s = crd.AlphasExact(as_mc_3, mc, scale_gev, 3, NLOOPS)
    return alpha_s, 3


def _run_mass_between_scales(m_gev: float, as_start: float, mu0_gev: float,
                             mu_gev: float, nf: int) -> tuple[float, float]:
    """Run a quark mass from mu0 to mu at fixed nf using AsmMSrunexact.

    Returns (m_at_mu_gev, alpha_s_at_mu).
    """
    crd = rundec.CRunDec()
    result = crd.AsmMSrunexact(m_gev, as_start, mu0_gev, mu_gev, nf, NLOOPS)
    return result.mMSexact, result.Asexact


def run_ms_to_scale(ms_at_2gev_mev: float, target_scale_mev: float,
                    alpha_s_mz: float = ALPHA_S_MZ) -> float:
    """Run m_s from 2 GeV to target_scale via four-loop QCD.

    m_s(2 GeV) is treated as being in the nf=4 MSbar scheme (since 2 GeV > m_c).
    Uses N_f = 4 between m_c and m_b, N_f = 3 below m_c, N_f = 5 above m_b.

    Parameters
    ----------
    ms_at_2gev_mev : m_s(2 GeV) in MeV
    target_scale_mev : target scale in MeV
    alpha_s_mz : alpha_s(M_Z)

    Returns
    -------
    m_s at the target scale, in MeV
    """
    crd = rundec.CRunDec()
    target_gev = target_scale_mev / 1000.0
    ms_gev = ms_at_2gev_mev / 1000.0
    mu0_gev = 2.0
    mc = M_C_AT_MC_GEV
    mb = M_B_AT_MB_GEV

    # alpha_s at 2 GeV in nf=4 scheme
    as_2gev, _ = _alpha_s_at_scale(mu0_gev, alpha_s_mz)

    # Case 1: target is in the nf=4 region (between m_c and m_b)
    if mc <= target_gev <= mb:
        m, _ = _run_mass_between_scales(ms_gev, as_2gev, mu0_gev, target_gev, 4)
        return m * 1000.0

    # Case 2: target is below m_c — run to m_c in nf=4, decouple, continue in nf=3
    if target_gev < mc:
        m_mc, as_mc = _run_mass_between_scales(ms_gev, as_2gev, mu0_gev, mc, 4)
        # Decouple at m_c: nf=4 -> nf=3
        m_mc_3 = crd.DecMqDownMS(m_mc, as_mc, mc, mc, 3, NLOOPS)
        as_mc_3 = crd.DecAsDownMS(as_mc, mc, mc, 3, NLOOPS)
        m, _ = _run_mass_between_scales(m_mc_3, as_mc_3, mc, target_gev, 3)
        return m * 1000.0

    # Case 3: target is above m_b — run to m_b in nf=4, decouple, continue in nf=5
    m_mb, as_mb = _run_mass_between_scales(ms_gev, as_2gev, mu0_gev, mb, 4)
    m_mb_5 = crd.DecMqUpMS(m_mb, as_mb, mb, mb, 4, NLOOPS)
    as_mb_5 = crd.DecAsUpMS(as_mb, mb, mb, 4, NLOOPS)
    m, _ = _run_mass_between_scales(m_mb_5, as_mb_5, mb, target_gev, 5)
    return m * 1000.0


def run_mc_to_scale(mc_at_mc_mev: float, target_scale_mev: float,
                    alpha_s_mz: float = ALPHA_S_MZ) -> float:
    """Run m_c from m_c to target scale via four-loop QCD."""
    crd = rundec.CRunDec()
    mc_gev = mc_at_mc_mev / 1000.0
    target_gev = target_scale_mev / 1000.0
    mb = M_B_AT_MB_GEV
    mc_thresh = M_C_AT_MC_GEV

    # alpha_s at m_c in nf=4
    as_mc, _ = _alpha_s_at_scale(mc_thresh, alpha_s_mz)

    # Case 1: target between m_c and m_b (nf=4)
    if mc_thresh <= target_gev <= mb:
        m, _ = _run_mass_between_scales(mc_gev, as_mc, mc_thresh, target_gev, 4)
        return m * 1000.0

    # Case 2: target below m_c — decouple to nf=3
    if target_gev < mc_thresh:
        m_mc_3 = crd.DecMqDownMS(mc_gev, as_mc, mc_thresh, mc_thresh, 3, NLOOPS)
        as_mc_3 = crd.DecAsDownMS(as_mc, mc_thresh, mc_thresh, 3, NLOOPS)
        m, _ = _run_mass_between_scales(m_mc_3, as_mc_3, mc_thresh, target_gev, 3)
        return m * 1000.0

    # Case 3: target above m_b — run to m_b in nf=4, decouple to nf=5
    m_mb, as_mb = _run_mass_between_scales(mc_gev, as_mc, mc_thresh, mb, 4)
    m_mb_5 = crd.DecMqUpMS(m_mb, as_mb, mb, mb, 4, NLOOPS)
    as_mb_5 = crd.DecAsUpMS(as_mb, mb, mb, 4, NLOOPS)
    m, _ = _run_mass_between_scales(m_mb_5, as_mb_5, mb, target_gev, 5)
    return m * 1000.0


def run_mb_to_scale(mb_at_mb_mev: float, target_scale_mev: float,
                    alpha_s_mz: float = ALPHA_S_MZ) -> float:
    """Run m_b from m_b to target scale via four-loop QCD."""
    crd = rundec.CRunDec()
    mb_gev = mb_at_mb_mev / 1000.0
    target_gev = target_scale_mev / 1000.0
    mc = M_C_AT_MC_GEV
    mb_thresh = M_B_AT_MB_GEV

    # alpha_s at m_b in nf=5 (just above threshold)
    as_mb_5, _ = _alpha_s_at_scale(mb_thresh, alpha_s_mz)

    # Case 1: target above m_b (nf=5)
    if target_gev >= mb_thresh:
        m, _ = _run_mass_between_scales(mb_gev, as_mb_5, mb_thresh, target_gev, 5)
        return m * 1000.0

    # Running down: decouple to nf=4 at m_b
    mb_at_mb_4 = crd.DecMqDownMS(mb_gev, as_mb_5, mb_thresh, mb_thresh, 4, NLOOPS)
    as_mb_4 = crd.DecAsDownMS(as_mb_5, mb_thresh, mb_thresh, 4, NLOOPS)

    # Case 2: target between m_c and m_b (nf=4)
    if target_gev >= mc:
        m, _ = _run_mass_between_scales(mb_at_mb_4, as_mb_4, mb_thresh, target_gev, 4)
        return m * 1000.0

    # Case 3: target below m_c — run to m_c in nf=4, decouple to nf=3
    m_mc, as_mc = _run_mass_between_scales(mb_at_mb_4, as_mb_4, mb_thresh, mc, 4)
    m_mc_3 = crd.DecMqDownMS(m_mc, as_mc, mc, mc, 3, NLOOPS)
    as_mc_3 = crd.DecAsDownMS(as_mc, mc, mc, 3, NLOOPS)
    m, _ = _run_mass_between_scales(m_mc_3, as_mc_3, mc, target_gev, 3)
    return m * 1000.0
