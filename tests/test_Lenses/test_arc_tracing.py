import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity

import slsim.Pipelines as pipelines
from slsim.Deflectors.DeflectorPopulation.generate_cluster_deflectors import (
    GeneratedDeflector,
)
from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source
from slsim.Sources.SourcePopulation.galaxies import convert_catalog_to_source

from slsim.Lenses.arc_tracing import (
    compute_LW_phi,
    dist_hlr,
    solve_max_rad_re,
    trace_finite_source_images,
)


class TestArcTracing(object):

    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # deflector: a generated cluster (DeflectorGroup), same recipe as the cluster tests
        galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
            sky_area=Quantity(value=0.001, unit="deg2"),
        )
        red = galaxy_simulation_pipeline.red_galaxies
        blue = galaxy_simulation_pipeline.blue_galaxies
        self.deflector = GeneratedDeflector(10**14, 0.35, self.cosmo).get_deflector(
            red, blue, min_subhalo_accretion_mass=1e12
        )

        # source: a simple sersic galaxy behind the cluster
        source_dict = {
            "z": 1.5,
            "angular_size": 0.3,
            "mag_r": 22.0,
            "e1": 0.05,
            "e2": 0.0,
            "n_sersic": 1,
            "center_x": 0.3,
            "center_y": 0.2,
        }
        kwargs_source = convert_catalog_to_source(
            source_dict,
            extended_source_type="single_sersic",
            catalog_type=None,
            size_model=None,
            cosmo=self.cosmo,
            include_all_keywords=False,
        )
        self.source = Source(cosmo=self.cosmo, **kwargs_source)

        self.lens = Lens(
            deflector_class=self.deflector,
            source_class=self.source,
            cosmo=self.cosmo,
        )

    def test_solve_max_rad_re(self):
        result = solve_max_rad_re(np.array([1.0, 4.0]), np.array([10.0, 10.0]), 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_dist_hlr(self):
        result = dist_hlr(self.source, np.array([0.5, 1.0]), np.array([0.0, 0.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_compute_LW_phi(self):
        idx = np.arange(20)
        gx, gy = np.meshgrid((idx + 0.5) * 0.1, (idx + 0.5) * 0.1, indexing="ij")
        keep = (np.hypot(gx, gy) >= 1.0) & (np.hypot(gx, gy) <= 1.6)
        length, width, phi = compute_LW_phi((gx[keep], gy[keep]), pix_width=0.1)
        assert isinstance(length, (float, np.floating))
        assert isinstance(width, (float, np.floating))
        assert isinstance(phi, (float, np.floating))

    def test_trace_finite_source_images(self):
        result = trace_finite_source_images(self.lens, search_window=20, mag_band="r")
        assert isinstance(result, list)
        assert len(result) == 1  # one source
        assert isinstance(result[0], list)
        for X, Y in result[0]:
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)

        result = trace_finite_source_images(self.lens, search_window=20, mag_band="r", return_interior=True)
        assert isinstance(result, list)
        assert len(result) == 1  # one source
        assert isinstance(result[0], list)
        for X, Y in result[0]:
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)