from slsim.Deflectors.DeflectorPopulation.generate_cluster_deflectors import (
    GeneratedDeflector,
    GeneratedDeflectorPopulation,
)
from slsim.Deflectors.deflector_group import DeflectorGroup
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity

import numpy as np
import numpy.testing as npt
import slsim.Pipelines as pipelines


class TestGeneratedDeflector(object):

    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.M200h = 10 ** 14
        self.z = 0.35
        self.deflector = GeneratedDeflector(self.M200h, self.z, self.cosmo)

        galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
            sky_area=Quantity(value=0.001, unit="deg2"),
        )

        self.red = galaxy_simulation_pipeline.red_galaxies
        self.blue = galaxy_simulation_pipeline.blue_galaxies

    def test_init(self):
        assert self.deflector.M200h == self.M200h
        assert self.deflector.z == self.z
        assert self.deflector.c > 0
        assert self.deflector.R200h > 0
        assert self.deflector.Rvir > 0

    def test__accreted_subhalo_density_pdf(self):
        m = np.logspace(10, 13, 5)
        out = self.deflector._accreted_subhalo_density_pdf(m)
        assert out.shape == m.shape
        assert np.all(out > 0)

    def test__unevolved_spatial_distribution_pdf(self):
        r = np.array([10.0, 100.0, 500.0])
        out = self.deflector._unevolved_spatial_distribution_pdf(r)
        assert out.shape == r.shape
        assert np.all(out > 0)

    def test__evolved_subhalo_mass_fraction_of_accretion(self):
        r = np.linspace(10, self.deflector.R200h, 20)
        out = self.deflector._evolved_subhalo_mass_fraction_of_accretion(r)
        assert out.shape == r.shape
        assert np.all(out >= 0)

    def test__generate_subhalos(self):
        out = self.deflector._generate_subhalos(1e12, self.M200h)
        assert out.ndim == 2
        assert out.shape[1] == 3

    def test__generate_deflector_population(self):
        halos = self.deflector._generate_deflector_population(
            min_subhalo_accretion_mass=1e12
        )
        assert isinstance(halos, list)
        # host halo is always first
        assert halos[0]["dist_to_center"] == 0
        assert halos[0]["subhalo_m_evolved"] == self.M200h
        assert len(halos[0]["pos_2d"]) == 2

    def test_get_deflector_data(self):
        data = self.deflector.get_deflector_data(
            self.red, self.blue, min_subhalo_accretion_mass=1e12
        )
        keys = [
            "kwargs_mass_list",
            "kwargs_light_list",
            "center_x_deflector_list",
            "center_y_deflector_list",
        ]
        n = len(data["kwargs_mass_list"])
        for key in keys:
            assert isinstance(data[key], list)
            assert len(data[key]) == n
        assert data["kwargs_mass_list"][0]["mass_type"] == "NFW_HERNQUIST"

        data = self.deflector.get_deflector_data(
            self.red,
            self.blue,
            min_subhalo_accretion_mass=1e12,
            kwargs_host_mass={"gamma_in": 1.2},
        )
        assert data["kwargs_mass_list"][0]["gamma_in"] == 1.2

    def test_get_deflector(self):
        deflector = self.deflector.get_deflector(
            self.red, self.blue, min_subhalo_accretion_mass=1e12
        )
        assert isinstance(deflector, DeflectorGroup)
        assert deflector.redshift == self.z

    def test__fraction_red_galaxies(self):
        out = self.deflector._fraction_red_galaxies(0.5 * self.deflector.R200, 1e10)
        assert 0 <= out <= 1

    def test__fraction_central_red(self):
        out = self.deflector._fraction_central_red(1e12)
        assert 0 <= out <= 1


class TestGeneratedDeflectorPopulation(object):

    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
            sky_area=Quantity(value=0.001, unit="deg2"),
        )

        self.red = galaxy_simulation_pipeline.red_galaxies
        self.blue = galaxy_simulation_pipeline.blue_galaxies

        self.population = GeneratedDeflectorPopulation(
            Mmin=1e13,
            Mmax=1e14,
            zmin=0.4,
            zmax=0.5,
            red_galaxies=self.red,
            blue_galaxies=self.blue,
            sky_area=Quantity(value=100000, unit="deg2"),
            cosmo=self.cosmo,
            crop_subhalo_dist=50,
            min_subhalo_accretion_mass=1e11,
        )

    def test_deflector_number(self):
        n = self.population.deflector_number()
        assert isinstance(n, int)
        assert n > 0

    def test_draw_deflector(self):
        deflector = self.population.draw_deflector()
        assert isinstance(deflector, DeflectorGroup)
        assert deflector.deflector_type == "group"
        assert 0.4 <= deflector.redshift <= 0.5

        # verbose shouldn't crash either
        self.population.verbose = True
        self.population.draw_deflector()