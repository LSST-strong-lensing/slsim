from astropy.cosmology import FlatLambdaCDM
import numpy as np
from sim_pipeline.halos.halos import number_density_at_redshift, growth_factor_at_redshift, halo_mass_at_z


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
