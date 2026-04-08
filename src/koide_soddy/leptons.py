"""Lepton masses, Koide relation, and Soddy curvature computations."""

import math

# PDG 2024 charged lepton pole masses, MeV
M_E_MEV = 0.51099895
M_MU_MEV = 105.6583755
M_TAU_MEV = 1776.93

# PDG 2024 lepton uncertainties (1 sigma), MeV
DM_E_MEV = 1.5e-10
DM_MU_MEV = 2.3e-6
DM_TAU_MEV = 0.09

# Sum of lepton masses in MeV
SUM_LEPTON_MEV = M_E_MEV + M_MU_MEV + M_TAU_MEV


def koide_Q(me: float, mmu: float, mtau: float) -> float:
    """Compute Q = (sum m) / (sum sqrt(m))**2. Should be ~ 0.666661 for PDG values."""
    s = me + mmu + mtau
    sr = math.sqrt(me) + math.sqrt(mmu) + math.sqrt(mtau)
    return s / sr**2


def F_from_soddy(me: float, mmu: float, mtau: float) -> float:
    """Outer Soddy curvature: F = e1 - 2*sqrt(e2). Units: sqrt(MeV)."""
    e1 = math.sqrt(me) + math.sqrt(mmu) + math.sqrt(mtau)
    e2 = (math.sqrt(me * mmu) + math.sqrt(me * mtau)
           + math.sqrt(mmu * mtau))
    return e1 - 2.0 * math.sqrt(e2)


def F_from_diff(me: float, mmu: float, mtau: float) -> float:
    """Equivalent form under Koide: F = e1 - sqrt(sum m). Units: sqrt(MeV)."""
    e1 = math.sqrt(me) + math.sqrt(mmu) + math.sqrt(mtau)
    return e1 - math.sqrt(me + mmu + mtau)


def F_central() -> float:
    """F at PDG central values. Units: sqrt(MeV)."""
    return F_from_soddy(M_E_MEV, M_MU_MEV, M_TAU_MEV)


def F_squared_central() -> float:
    """F^2 at PDG central values, in MeV."""
    return F_central() ** 2


def F_uncertainty_from_leptons() -> tuple[float, float]:
    """Propagate PDG lepton uncertainties to (sigma_F, sigma_F2).

    Linear error propagation via partial derivatives.
    Returns (sigma_F, sigma_F2) both in appropriate units.
    """
    me, mmu, mtau = M_E_MEV, M_MU_MEV, M_TAU_MEV
    dme, dmmu, dmtau = DM_E_MEV, DM_MU_MEV, DM_TAU_MEV

    # F = e1 - 2*sqrt(e2)
    # e1 = sqrt(me) + sqrt(mmu) + sqrt(mtau)
    # e2 = sqrt(me*mmu) + sqrt(me*mtau) + sqrt(mmu*mtau)
    #
    # dF/dmi = d(e1)/dmi - 2 * d(sqrt(e2))/dmi
    # d(e1)/dmi = 1/(2*sqrt(mi))
    # d(e2)/dmi = (sqrt(mj) + sqrt(mk)) / (2*sqrt(mi))  for j,k != i
    # d(sqrt(e2))/dmi = d(e2)/dmi / (2*sqrt(e2))

    e2 = (math.sqrt(me * mmu) + math.sqrt(me * mtau)
          + math.sqrt(mmu * mtau))
    sqrt_e2 = math.sqrt(e2)

    masses = [me, mmu, mtau]
    uncertainties = [dme, dmmu, dmtau]

    var_F = 0.0
    for i in range(3):
        mi = masses[i]
        # other two masses
        others = [masses[j] for j in range(3) if j != i]
        mj, mk = others

        de1_dmi = 1.0 / (2.0 * math.sqrt(mi))
        de2_dmi = (math.sqrt(mj) + math.sqrt(mk)) / (2.0 * math.sqrt(mi))
        dsqrt_e2_dmi = de2_dmi / (2.0 * sqrt_e2)

        dF_dmi = de1_dmi - 2.0 * dsqrt_e2_dmi
        var_F += (dF_dmi * uncertainties[i]) ** 2

    sigma_F = math.sqrt(var_F)
    F = F_central()
    sigma_F2 = 2.0 * abs(F) * sigma_F

    return sigma_F, sigma_F2
