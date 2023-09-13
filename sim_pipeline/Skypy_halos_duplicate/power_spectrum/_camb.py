import numpy as np
from astropy import units
from ._base import TabulatedPowerSpectrum

# TODO: This code has been temporarily borrowed from SkyPy.  Once the feature is available in the main branch or
#  release version of SkyPy,  this code should be deprecated and replaced with the official implementation. Original
#  Source: [https://github.com/skypyproject/skypy/tree/module/halos]

__all__ = [
    "CAMB",
]


class CAMB(TabulatedPowerSpectrum):
    """CAMB linear matter power spectrum.

    Compute the linear matter power spectrum on a two dimensional grid of redshift and
    wavenumber using CAMB.
    """

    def __init__(self, kmax, redshift, cosmology, A_s, n_s, **kwargs):
        try:
            from camb import CAMBparams, model, get_results
        except ImportError:
            raise Exception("camb is required to use skypy.power_spectrum.camb")

        h2 = cosmology.h * cosmology.h

        pars = CAMBparams()
        pars.set_cosmology(
            H0=cosmology.H0.value,
            ombh2=cosmology.Ob0 * h2,
            omch2=cosmology.Odm0 * h2,
            omk=cosmology.Ok0,
            TCMB=cosmology.Tcmb0.value,
            mnu=np.sum(cosmology.m_nu.to_value(units.eV)),
            standard_neutrino_neff=cosmology.Neff,
        )

        pars.InitPower.ns = n_s
        pars.InitPower.As = A_s
        pars.NonLinear = model.NonLinear_none
        k_per_logint, var1, var2, hubble_units, nonlinear = (
            None,
            None,
            None,
            False,
            False,
        )
        pars.set_matter_power(
            redshifts=redshift, kmax=kmax, k_per_logint=k_per_logint, silent=True
        )
        results = get_results(pars)
        k, z, p = results.get_linear_matter_power_spectrum(
            var1, var2, hubble_units, nonlinear=nonlinear
        )

        super().__init__(k * cosmology.h, z, p)
