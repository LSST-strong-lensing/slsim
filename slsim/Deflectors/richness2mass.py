import numpy as np


def mass_richness_simet2017(rich, min_mass=1e11):
    """
    mass-richness relation from Simet et al. 2017
    M(N) = M_0 * (N / N_0) ** alpha
    VAR[ln(M)] = alpha**2 / N + lnM_scatter**2
    """
    alpha = 1.33
    m_0 = 10**14.344
    rich_0 = 40
    ln_m_scatter = 0.25

    m = m_0 * (rich / rich_0) ** alpha

    ln_m_var = alpha**2 / rich + ln_m_scatter**2
    m_var = ln_m_var * m**2  # VAR[log(X)] ~ VAR[X] / E[X]^2
    m += np.random.normal(0, np.sqrt(m_var))
    m = np.maximum(m, min_mass)
    return m
