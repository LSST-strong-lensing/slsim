from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    schechter_vel_disp,
    schechter_velocity_dispersion_function,
    vel_disp_composite_model,
    vel_disp_nfw_aperture,
    redshifts_from_comoving_density,
    vel_disp_power_law,
    theta_E_from_vel_disp_epl,
)
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
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


def test_vel_disp_power_law():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z_lens = 0.5
    z_source = 1.5
    theta_E = 1.5
    gamma = 2
    r_half = 1
    kwargs_light = [
        {
            "magnitude": 1,
            "R_sersic": 2,
            "n_sersic": 1,
            "e1": 0,
            "e2": 0,
            "center_x": 0,
            "center_y": 0,
        }
    ]
    light_model_list = ["SERSIC_ELLIPSE"]

    # kwargs_light = [{"amp": 1, "gamma": 2, "e1": 0, "e2": 0, "center_x": 0, "center_y": 0}]
    # light_model_list = ["POWER_LAW"]
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

    vel_disp_pl = vel_disp_power_law(
        theta_E, gamma, r_half, kwargs_light, light_model_list, lens_cosmo
    )

    vel_disp_sis = lens_cosmo.sis_theta_E2sigma_v(theta_E)
    npt.assert_almost_equal(vel_disp_pl / vel_disp_sis, 1, decimal=1)


def test_theta_E_from_vel_disp_epl():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z_lens = 0.5
    z_source = 1.5
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    theta_E = 1.5
    gamma = 2.5
    r_half = 1
    kwargs_light = [
        {
            "magnitude": 1,
            "R_sersic": 2,
            "n_sersic": 1,
            "e1": 0,
            "e2": 0,
            "center_x": 0,
            "center_y": 0,
        }
    ]
    light_model_list = ["SERSIC_ELLIPSE"]

    vel_disp_pl = vel_disp_power_law(
        theta_E, gamma, r_half, kwargs_light, light_model_list, lens_cosmo
    )

    theta_E_out = theta_E_from_vel_disp_epl(
        vel_disp_pl,
        gamma,
        r_half,
        kwargs_light,
        light_model_list,
        lens_cosmo,
        kappa_ext=0,
        sis_convention=False,
    )
    npt.assert_almost_equal(theta_E_out / theta_E, 1, decimal=3)


def test_schechter_vdf():
    """SDSS velocity dispersion function for galaxies brighter than Mr >=
    -16.8."""

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    phi_star = 8.0 * 10 ** (-3) / cosmo.h**3
    vd_star = 161
    alpha = 2.32
    beta = 2.67

    vel_disp_list1 = schechter_velocity_dispersion_function(
        alpha,
        beta,
        phi_star,
        vd_star,
        vd_min=50,
        vd_max=500,
        size=10000,
        resolution=100,
    )
    vel_disp_list2 = schechter_velocity_dispersion_function(
        alpha,
        beta,
        phi_star,
        vd_star,
        vd_min=50,
        vd_max=450,
        size=None,
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
    assert len(z_list) == 117
    assert len(vel_disp_list1) == 10000
    assert len(np.array([vel_disp_list2])) == 1


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
    dN_dz = (cosmo.differential_comoving_volume(redshift) * sky_area).to_value("Mpc3")
    dN_dz *= density
    N = np.trapz(dN_dz, redshift)
    assert np.isclose(len(redshifts_no_noise), int(N), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
