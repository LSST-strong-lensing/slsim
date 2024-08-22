import numpy as np


def general_mass_richness(rich, alpha, rich_0, m_0, ln_m_scatter, min_mass=0.0):
    """M(N) = M_0 * (N / N_0) ** alpha VAR[ln(M)] = alpha**2 / N + lnM_scatter**2."""
    m = m_0 * (rich / rich_0) ** alpha

    ln_m_var = alpha**2 / rich + ln_m_scatter**2
    m_var = ln_m_var * m**2  # VAR[log(X)] ~ VAR[X] / E[X]^2
    m += np.random.normal(0, np.sqrt(m_var))
    m = np.maximum(m, min_mass)
    return m


def mass_richness_simet2017(rich, min_mass=1e11):
    """Mass-richness relation from Simet et al.

    2017
    """
    return general_mass_richness(
        rich,
        alpha=1.33,
        rich_0=40,
        m_0=10**14.344,
        ln_m_scatter=0.25,
        min_mass=min_mass,
    )


def mass_richness_abdullah2022(rich, min_mass=1e11):
    """Mass-richness relation from Abdullah et al.

    2022 (sred13)
    """
    return general_mass_richness(
        rich, alpha=0.95, rich_0=1, m_0=11.4e12, ln_m_scatter=0.25, min_mass=min_mass
    )


def mass_richness_relation(rich, relation="Abdullah2022"):
    if relation == "simet2017":
        return mass_richness_simet2017(rich)
    elif relation == "Abdullah2022":
        return mass_richness_abdullah2022(rich)
    else:
        raise ValueError(f"Unknown mass-richness relation: {relation}")
