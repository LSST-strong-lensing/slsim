from sim_pipeline.Deflectors.velocity_dispersion import (
    schechter_vel_disp,
    schechter_velocity_dispersion_function,
)
import numpy as np


def test_schechter_vdf():
    # SDSS velocity dispersion function for galaxies brighter than Mr >= -16.8
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    phi_star = 8.0 * 10 ** (-3) / cosmo.h**3
    vd_star = 161
    alpha = 2.32
    beta = 2.67

    vel_disp_list = schechter_velocity_dispersion_function(
        alpha, beta, vd_star, vd_min=50, vd_max=500, size=10000, resolution=100
    )

    # plt.hist(np.log10(vel_disp_list))
    # plt.show()

    redshift = np.linspace(start=0, stop=0.2, num=10)
    from astropy.units import Quantity

    sky_area = Quantity(value=0.1, unit="deg2")
    vd_min, vd_max = 50, 500
    from astropy.cosmology import FlatLambdaCDM

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
    assert len(z_list) == 373

    # plt.hist(np.log10(vel_disp_list))
    # plt.show()
