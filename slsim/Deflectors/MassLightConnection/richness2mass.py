import numpy as np

"""
Module to convert richness to mass using mass-richness relations. This is used in ClusterDeflector
to assign a mass to each cluster based on its richness.
"""


def general_mass_richness(rich, alpha, rich_0, m_0, ln_m_scatter, min_mass=0.0):
    """Mass-richness relation:
    M(N) = M_0 * (N / N_0) ** alpha VAR[ln(M)] = alpha**2 / N + lnM_scatter**2.
    :param rich: richness
    :type rich: float
    :param alpha: slope
    :type alpha: float
    :param rich_0: pivot richness
    :type rich_0: float
    :param m_0: mass at pivot richness
    :type m_0: float
    :param ln_m_scatter: scatter in log mass
    :type ln_m_scatter: float
    :param min_mass: minimum mass
    :type min_mass: float
    """
    m = m_0 * (rich / rich_0) ** alpha

    ln_m_var = alpha**2 / rich + ln_m_scatter**2
    m_var = ln_m_var * m**2  # VAR[log(X)] ~ VAR[X] / E[X]^2
    m += np.random.normal(0, np.sqrt(m_var))
    m = np.maximum(m, min_mass)
    return m


def mass_richness_simet2017(rich, min_mass=1e12):
    """Mass-richness relation from Simet et al.

    2017
    :param rich: richness
    :type rich: float
    :param min_mass: minimum mass
    :type min_mass: float
    """
    return general_mass_richness(
        rich,
        alpha=1.33,
        rich_0=40,
        m_0=10**14.344,
        ln_m_scatter=0.25,
        min_mass=min_mass,
    )


def mass_richness_abdullah2022(rich, min_mass=1e12):
    """Mass-richness relation from Abdullah et al.

    2022 (sred13)
    :param rich: richness
    :type rich: float
    :param min_mass: minimum mass
    :type min_mass: float
    """
    return general_mass_richness(
        rich, alpha=0.95, rich_0=1, m_0=11.4e12, ln_m_scatter=0.25, min_mass=min_mass
    )


def mass_richness_relation(rich, relation="Abdullah2022"):
    """Mass-richness relation :param rich: richness :param relation: mass-
    richness relation :type relation: str."""
    if relation == "simet2017":
        return mass_richness_simet2017(rich)
    elif relation == "Abdullah2022":
        return mass_richness_abdullah2022(rich)
    else:
        raise ValueError(f"Unknown mass-richness relation: {relation}")
