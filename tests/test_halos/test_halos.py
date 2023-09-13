from astropy.cosmology import FlatLambdaCDM
import numpy as np
from sim_pipeline.Halos.halos import (
    number_density_at_redshift,
    growth_factor_at_redshift,
    halo_mass_at_z,
    set_defaults,
    redshift_halos_array_from_comoving_density,
)
from astropy.cosmology import default_cosmology
from astropy import units


def test_halo_mass_at_z():
    mass = halo_mass_at_z(z=0.5, resolution=100)
    assert mass[0] > 10**10
    assert mass[0] < 10**16


def test_number_density_at_redshift():
    z = 0.5
    CDF = number_density_at_redshift(z=z)
    assert CDF is not None


def test_growth_factor_at_redshift():
    growth_factor = growth_factor_at_redshift(z=0.5)
    assert isinstance(growth_factor, float)
    assert np.isfinite(growth_factor)


def test_defaults_set():
    (
        m_min,
        m_max,
        wavenumber,
        resolution,
        power_spectrum,
        cosmology,
        collapse_function,
        params,
    ) = set_defaults()
    assert m_min == 1e10
    assert m_max == 1e14
    assert len(wavenumber) == 1000
    assert resolution == 1000
    assert cosmology.name == "Planck18"
    assert collapse_function.__name__ == "ellipsoidal_collapse_function"
    assert params == (0.3, 0.7, 0.3, 1.686)


def test_redshift_halos_array_from_comoving_density():
    redshift_list = np.linspace(0, 5.00, 1000)
    sky_area = 0.00005 * units.deg**2
    cosmo = default_cosmology.get()
    result = redshift_halos_array_from_comoving_density(
        redshift_list=redshift_list,
        cosmology=cosmo,
        sky_area=sky_area,
        m_min=1e12,
        m_max=1e16,
    )
    mass = halo_mass_at_z(z=result, m_min=1e12, m_max=1e16)
    assert len(mass) <= 50
    assert len(mass) > 0
    assert mass[0] > 10**12
    assert isinstance(mass[0], np.ndarray)
