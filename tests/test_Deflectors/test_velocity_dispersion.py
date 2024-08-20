from slsim.Deflectors.velocity_dispersion import (
    schechter_vel_disp,
    schechter_velocity_dispersion_function,
    vel_disp_composite_model,
    vel_disp_nfw_aperture,
    redshifts_from_comoving_density,
)
import numpy as np
import numpy.testing as npt
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
import pytest


def test_vel_disp_composite_model():
    """"""
    m_star = 10**11  # M_sun
    rs_star = 0.005  # 5kpc
    m_halo = 10**13.5  # M_sun
    c_halo = 10
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    r = 1
    z_lens = 0.5
    vel_disp = vel_disp_composite_model(
        r, m_star, rs_star, m_halo, c_halo, cosmo, z_lens
    )
    npt.assert_almost_equal(vel_disp, 200, decimal=-1)


def test_vel_disp_nfw_aperture():
    m_halo = 10**15  # M_sun
    c_halo = 5
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    r = 10
    z_lens = 0.5
    vel_disp = vel_disp_nfw_aperture(
        r=r,
        m_halo=m_halo,
        c_halo=c_halo,
        cosmo=cosmo,
        z_lens=z_lens,
    )
    npt.assert_almost_equal(vel_disp, 1000, decimal=-1)


def test_schechter_vdf():
    """SDSS velocity dispersion function for galaxies brighter than Mr >= -16.8."""

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    phi_star = 8.0 * 10 ** (-3) / cosmo.h**3
    vd_star = 161
    alpha = 2.32
    beta = 2.67

    vel_disp_list = schechter_velocity_dispersion_function(
        alpha,
        beta,
        phi_star,
        vd_star,
        vd_min=50,
        vd_max=500,
        size=10000,
        resolution=100,
    )

    # plt.hist(np.log10(vel_disp_list))
    # plt.show()

    redshift = np.linspace(start=0, stop=0.2, num=10)
    from astropy.units import Quantity

    sky_area = Quantity(value=0.1, unit="deg2")
    vd_min, vd_max = 50, 500

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    cosmology = cosmo
    np.random.seed(42)

    z_list, vel_disp_list = schechter_vel_disp(
        redshift,
        phi_star,
        alpha,
        beta,
        vd_star,
        vd_min,
        vd_max,
        sky_area,
        cosmology,
        noise=True,
    )
    assert len(z_list) == 130


def test_redshifts_from_comoving_density():
    # Define input parameters
    redshift = np.linspace(0.1, 2.0, 50)
    density = np.ones_like(redshift) * 1e-3  # constant density
    sky_area = Quantity(value=1, unit="deg2")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Test with noise = True
    redshifts = redshifts_from_comoving_density(
        redshift, density, sky_area, cosmo, noise=True
    )

    # Check the output is an array and is not empty
    assert isinstance(redshifts, np.ndarray)
    assert len(redshifts) > 0

    # Test with noise = False
    redshifts_no_noise = redshifts_from_comoving_density(
        redshift, density, sky_area, cosmo, noise=False
    )

    # Check the output is an array and is not empty
    assert isinstance(redshifts_no_noise, np.ndarray)
    assert len(redshifts_no_noise) > 0

    # Check that the number of galaxies is approximately equal to the expected number
    expected_number = np.sum(
        (cosmo.differential_comoving_volume(redshift) * sky_area).to_value("Mpc3")
        * density
    )
    assert np.isclose(len(redshifts_no_noise), int(expected_number), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
