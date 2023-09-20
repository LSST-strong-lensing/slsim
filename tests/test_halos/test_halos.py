import numpy as np
from sim_pipeline.Halos.halos import number_density_at_redshift, growth_factor_at_redshift, halo_mass_at_z, \
    set_defaults, redshift_halos_array_from_comoving_density, mass_first_moment_at_redshift, \
    redshift_mass_sheet_correction_array_from_comoving_density, kappa_ext_for_each_sheet

from astropy.cosmology import default_cosmology
from astropy import units
import pytest


cosmo = default_cosmology.get()


def test_halo_mass_at_z():
    mass = halo_mass_at_z(z=0.5, resolution=100)
    assert mass[0] > 10 ** 10
    assert mass[0] < 10 ** 16


def test_number_density_at_redshift():
    z = 0.5
    CDF = number_density_at_redshift(z=z)
    assert CDF is not None


def test_growth_factor_at_redshift():
    growth_factor = growth_factor_at_redshift(z=0.5)
    assert isinstance(growth_factor, float)
    assert np.isfinite(growth_factor)


def test_defaults_set():
    m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params = set_defaults()
    assert m_min == 1E+10
    assert m_max == 1E+14
    assert len(wavenumber) == 1000
    assert resolution == 1000
    assert cosmology.name == 'Planck18'
    assert collapse_function.__name__ == 'ellipsoidal_collapse_function'
    assert params == (0.3, 0.7, 0.3, 1.686)


def test_redshift_halos_array_from_comoving_density():
    redshift_list = np.linspace(0, 5.00, 1000)
    sky_area = 0.00005 * units.deg ** 2
    result = redshift_halos_array_from_comoving_density(redshift_list=redshift_list,
                                                        cosmology=cosmo,
                                                        sky_area=sky_area,
                                                        m_min=1E+12,
                                                        m_max=1E+16)
    mass = halo_mass_at_z(z=result, m_min=1E+12, m_max=1E+16)
    assert len(mass) <= 50
    assert len(mass) > 0
    assert mass[0] > 10 ** 12
    assert isinstance(mass[0], np.ndarray)


def test_valid_input_values():
    # Arrange
    z = [0.5, 1.0, 1.5]
    m_min = 1E+10
    m_max = 1E+14
    resolution = 1000
    wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
    cosmology = default_cosmology.get()
    params = (0.3, 0.7, 0.3, 1.686)

    result = mass_first_moment_at_redshift(z, m_min=m_min, m_max=m_max, resolution=resolution,
                                           wavenumber=wavenumber,
                                           cosmology=cosmology,
                                           params=params)

    assert len(result) == len(z)
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert isinstance(result[2], float)


def test_returns_array_of_redshift_values():
    redshift_list = np.linspace(0, 3, 1000)
    sky_area = 10 * units.deg ** 2
    cosmology = cosmo

    result = redshift_mass_sheet_correction_array_from_comoving_density(redshift_list, sky_area, cosmology)

    assert len(result) == 67781753
    assert isinstance(result[0], float)


def test_standard_input_for_all_parameters():
    redshift_list = [0.5, 0.7, 0.9]
    first_moment = [1, 2, 3]
    sky_area = 0.1 * units.deg ** 2
    cosmology = cosmo

    result = kappa_ext_for_each_sheet(redshift_list, first_moment, sky_area, cosmology)

    assert result == pytest.approx([-1.21439496e-17, -1.90636260e-17, -2.41697932e-17], rel=1e-10)
