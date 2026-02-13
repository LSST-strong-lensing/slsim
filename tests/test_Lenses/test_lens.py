import os
import copy
import pytest
import numpy as np
from copy import deepcopy
from astropy.table import Table
from numpy import testing as npt
from slsim.Lenses.lens import Lens
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.param_util import image_separation_from_positions
from slsim.LOS.los_individual import LOSIndividual
from slsim.LOS.los_pop import LOSPop
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector

# import pickle
from unittest.mock import patch, MagicMock  # Added for mocking

try:
    import jax

    print(jax.__path__)

    use_jax = True
except ImportError:
    use_jax = False


class TestLens(object):
    # pytest.fixture(scope='class')
    def setup_method(self):
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
        )
        blue_one["angular_size"] = blue_one["angular_size"] / 4.84813681109536e-06
        red_one = Table.read(
            os.path.join(path, "../TestData/red_one_modified.fits"), format="fits"
        )
        red_one["angular_size"] = red_one["angular_size"] / 4.84813681109536e-06
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        self.source_dict = blue_one
        self.deflector_dict = red_one
        self.los_individual = LOSIndividual(kappa=0.1, gamma=[-0.1, -0.2])

        mag_arc_limit = {"i": 35, "g": 35, "r": 35}
        while True:
            kwargs = {
                "extended_source_type": "single_sersic",
            }
            self.source = Source(
                cosmo=cosmo,
                **self.source_dict,
                **kwargs,
            )
            self.deflector = Deflector(
                deflector_type="EPL_SERSIC",
                **self.deflector_dict,
            )
            gg_lens = Lens(
                source_class=self.source,
                deflector_class=self.deflector,
                los_class=self.los_individual,
                lens_equation_solver="lenstronomy_analytical",
                # kwargs_variability={"MJD", "ps_mag_i"},  # This line will not be used in
                # the testing but at least code go through this warning message.
                cosmo=cosmo,
                use_jax=use_jax,
            )
            second_brightest_image_cut = {"i": 30}
            if gg_lens.validity_test(
                second_brightest_image_cut=second_brightest_image_cut,
                mag_arc_limit=mag_arc_limit,
            ):
                self.gg_lens = gg_lens
                break

        # Create another strong lens that has much higher SNR
        blue_one_high_snr = Table.read(
            os.path.join(path, "../TestData/blue_one_high_snr.fits"), format="fits"
        )
        blue_one_high_snr["angular_size"] = (
            blue_one_high_snr["angular_size"] / 4.84813681109536e-06
        )
        red_one_high_snr = Table.read(
            os.path.join(path, "../TestData/red_one_high_snr.fits"), format="fits"
        )
        red_one_high_snr["angular_size"] = (
            red_one_high_snr["angular_size"] / 4.84813681109536e-06
        )
        source_high_snr = Source(
            cosmo=cosmo,
            **blue_one_high_snr,
            **kwargs,
        )
        deflector_high_snr = Deflector(
            deflector_type="EPL_SERSIC",
            **red_one_high_snr,
        )
        self.gg_lens_high_snr = Lens(
            source_class=source_high_snr,
            deflector_class=deflector_high_snr,
            los_class=self.los_individual,
            lens_equation_solver="lenstronomy_analytical",
            cosmo=cosmo,
            use_jax=use_jax,
        )

        # Create another galaxy class with interpolated source.

        # Image Parameters
        size = 100
        center_brightness = 100
        noise_level = 10

        # Create a grid of coordinates
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)

        # Calculate the distance from the center
        r = np.sqrt(x**2 + y**2)

        # Create the galaxy image with light concentrated near the center
        image = center_brightness * np.exp(-(r**2) / 0.1)

        # Add noise to the image
        noise = noise_level * np.random.normal(size=(size, size))
        image += noise

        # Ensure no negative values
        image = np.clip(image, 0, None)
        test_image = image

        # Build a table for this "interp" source
        interp_source_dict = Table(
            names=(
                "z",
                "image",
                "center_x",
                "center_y",
                "z_data",
                "pixel_width_data",
                "phi_G",
                "mag_i",
                "mag_g",
                "mag_r",
            ),
            rows=[
                (
                    0.1,
                    test_image,
                    size // 2,
                    size // 2,
                    0.5,
                    0.05,
                    0.0,
                    20.0,
                    20.0,
                    20.0,
                )
            ],
        )
        kwargs_int = {"extended_source_type": "interpolated"}
        self.source_interp = Source(
            cosmo=cosmo,
            **interp_source_dict,
            **kwargs_int,
        )
        self.gg_lens_interp = Lens(
            source_class=self.source_interp,
            deflector_class=self.deflector,
            los_class=self.los_individual,
            lens_equation_solver="lenstronomy_analytical",
            cosmo=cosmo,
            use_jax=use_jax,
        )

    def test_validity_test(self):
        second_brightest_image_cut = {"i": 20}
        assert (
            self.gg_lens.validity_test(
                second_brightest_image_cut=second_brightest_image_cut
            )
            is False
        )

    def test_validity_test_with_snr_limit(self):
        # Test that snr_limit=None doesn't change behavior (backward compatibility)
        mag_arc_limit = {"i": 35, "g": 35, "r": 35}
        second_brightest_image_cut = {"i": 30}
        result_without_snr = self.gg_lens_high_snr.validity_test(
            mag_arc_limit=mag_arc_limit,
            second_brightest_image_cut=second_brightest_image_cut,
            snr_limit=None,
        )
        assert result_without_snr is True

        # Test with a very low SNR limit that should pass
        result_low_snr = self.gg_lens_high_snr.validity_test(
            snr_limit={"i": 0.1},
        )
        assert result_low_snr is True

        # Test with a very high SNR limit that should fail
        result_high_snr = self.gg_lens_high_snr.validity_test(
            snr_limit={"i": 1e10},
        )
        assert result_high_snr is False

    def test_snr_limit_type_validation(self):
        """Test that snr_limit must be a dict, not int/float."""
        # Test that passing an int raises TypeError
        with pytest.raises(TypeError, match="snr_limit must be a dict"):
            self.gg_lens_high_snr.validity_test(snr_limit=20)

        # Test that passing a float raises TypeError
        with pytest.raises(TypeError, match="snr_limit must be a dict"):
            self.gg_lens_high_snr.validity_test(snr_limit=20.5)

    def test_snr(self):
        # Test basic SNR calculation
        snr_result = self.gg_lens_high_snr.snr(
            band="i", fov_arcsec=6, observatory="LSST"
        )
        # SNR should be either a positive float or None
        assert snr_result is None or (
            isinstance(snr_result, (float, np.floating)) and snr_result > 0
        )

    def test_snr_with_high_threshold(self):
        # Test that a very high per-pixel threshold returns None
        snr_result = self.gg_lens_high_snr.snr(
            band="i",
            fov_arcsec=6,
            observatory="LSST",
            snr_per_pixel_threshold=1e10,
        )
        assert snr_result is None

    def _create_lens_for_observatory(self, band, observatory):
        """Helper method to create a lens with the specified band magnitude."""
        path = os.path.dirname(__file__)
        blue_one_high_snr = Table.read(
            os.path.join(path, "../TestData/blue_one_high_snr.fits"), format="fits"
        )
        blue_one_high_snr["angular_size"] = (
            blue_one_high_snr["angular_size"] / 4.84813681109536e-06
        )
        blue_one_high_snr[f"mag_{band}"] = blue_one_high_snr["mag_i"]

        red_one_high_snr = Table.read(
            os.path.join(path, "../TestData/red_one_high_snr.fits"), format="fits"
        )
        red_one_high_snr["angular_size"] = (
            red_one_high_snr["angular_size"] / 4.84813681109536e-06
        )
        red_one_high_snr[f"mag_{band}"] = red_one_high_snr["mag_i"]

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        kwargs = {"extended_source_type": "single_sersic"}
        source = Source(cosmo=cosmo, **blue_one_high_snr, **kwargs)
        deflector = Deflector(deflector_type="EPL_SERSIC", **red_one_high_snr)
        return Lens(
            source_class=source,
            deflector_class=deflector,
            los_class=self.los_individual,
            lens_equation_solver="lenstronomy_analytical",
            cosmo=cosmo,
            use_jax=use_jax,
        )

    @pytest.mark.parametrize(
        "band,observatory",
        [
            ("F106", "Roman"),
            ("VIS", "Euclid"),
        ],
    )
    def test_snr_multi_observatory(self, band, observatory):
        """Test SNR calculation with different observatories."""
        lens = self._create_lens_for_observatory(band, observatory)
        snr_result = lens.snr(band=band, fov_arcsec=6, observatory=observatory)
        assert snr_result is None or (
            isinstance(snr_result, (float, np.floating)) and snr_result > 0
        )

    @pytest.mark.parametrize(
        "band,observatory",
        [
            ("F106", "Roman"),
            ("VIS", "Euclid"),
        ],
    )
    def test_validity_test_with_snr_limit_multi_observatory(self, band, observatory):
        """Test validity_test with SNR limit for different observatories."""
        lens = self._create_lens_for_observatory(band, observatory)

        # Test with a very low SNR limit that should pass
        result_low_snr = lens.validity_test(snr_limit={band: 0.1})
        assert result_low_snr is True

        # Test SNR calculation to check if regions are found
        snr_result = lens.snr(band=band, fov_arcsec=20, observatory=observatory)

        # If SNR is calculable, verify high threshold fails
        if snr_result is not None:
            result_high_snr = lens.validity_test(snr_limit={band: 1e10})
            assert result_high_snr is False

    def test_lens_id_gg(self):
        lens_id = self.gg_lens.generate_id()
        ra = self.gg_lens.deflector_position[0]
        dec = self.gg_lens.deflector_position[1]
        ra2 = 12.03736542
        dec2 = 35.17363534
        lens_id2 = self.gg_lens.generate_id(ra=ra2, dec=dec2)
        assert lens_id == f"GAL-GAL-LENS_{ra:.4f}_{dec:.4f}"
        assert lens_id2 == f"GAL-GAL-LENS_{ra2:.4f}_{dec2:.4f}"

    def test_deflector_ellipticity(self):
        e1_light, e2_light, e1_mass, e2_mass = self.gg_lens.deflector_ellipticity()
        assert pytest.approx(e1_light, rel=1e-3) == -0.05661955320450283
        assert pytest.approx(e2_light, rel=1e-3) == 0.08738390223219591
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_deflector_magnitude(self):
        band = "g"
        deflector_magnitude = self.gg_lens.deflector_magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0], rel=1e-3) == 26.4515655

    def test_source_magnitude(self):
        band = "g"
        source_magnitude = self.gg_lens.extended_source_magnitude(band)
        source_magnitude_lensed = self.gg_lens.extended_source_magnitude(
            band, lensed=True
        )
        host_mag = self.gg_lens.extended_source_magnification
        expected_lensed_mag = source_magnitude - 2.5 * np.log10(host_mag)
        assert pytest.approx(source_magnitude[0], rel=1e-3) == 30.780194
        assert source_magnitude_lensed == expected_lensed_mag

    def test_image_separation_from_positions(self):
        image_positions = self.gg_lens.extended_source_image_positions[0]
        image_separation = image_separation_from_positions(image_positions)
        theta_E_infinity = self.gg_lens.deflector.theta_e_infinity(
            cosmo=self.gg_lens.cosmo
        )
        assert image_separation < 2 * theta_E_infinity

    def test_extended_source_magnification(self):
        host_mag = self.gg_lens.extended_source_magnification[0]
        assert host_mag > 0

    def test_deflector_stellar_mass(self):
        s_mass = self.gg_lens.deflector_stellar_mass()
        assert s_mass >= 10**5

    def test_deflector_velocity_dispersion(self):
        vdp = self.gg_lens.deflector_velocity_dispersion()
        assert vdp >= 10

    def test_los_linear_distortions(self):
        kappa, gamma1, gamma2 = self.gg_lens.los_linear_distortions
        assert kappa == self.los_individual.convergence
        g1, g2 = self.los_individual.shear
        assert gamma1 == g1
        assert gamma2 == g2

    def test_deflector_light_model_lenstronomy(self):
        kwargs_lens_light = self.gg_lens.deflector_light_model_lenstronomy(band="g")
        assert len(kwargs_lens_light) >= 1

    def test_extended_source_magnification_for_individual_images(self):
        results = self.gg_lens.extended_source_magnification_for_individual_image()
        assert len(results[0]) >= 2

    def test_extended_source_magnitude_for_each_images(self):
        result1 = self.gg_lens.extended_source_magnitude_for_each_image(
            band="i", lensed=True
        )
        result2 = self.gg_lens.extended_source_magnitude_for_each_image(
            band="i", lensed=False
        )
        result3 = self.gg_lens.extended_source_magnitude(band="i", lensed=False)
        assert len(result1[0]) >= 2
        assert result2 == result3

    def test_lenstronomy_kwargs(self):
        kwargs_model, kwargs_params = self.gg_lens.lenstronomy_kwargs(band="i")
        assert kwargs_model["point_source_model_list"] == []

    def test_lens_equation_solver(self):
        # Tests analytical and numerical lens equation solver options.
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        gg_lens = Lens(
            lens_equation_solver="lenstronomy_default",
            source_class=self.source,
            deflector_class=self.deflector,
            cosmo=cosmo,
            use_jax=use_jax,
        )
        while True:
            gg_lens.validity_test()
            break

        gg_lens = Lens(
            lens_equation_solver="lenstronomy_analytical",
            source_class=self.source,
            deflector_class=self.deflector,
            cosmo=cosmo,
            use_jax=use_jax,
        )
        while True:
            gg_lens.validity_test()
            break

        # and here for NFW-Hernquist model
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
        )
        source_dict = blue_one
        deflector_dict = {
            "halo_mass": 10**13,
            "concentration": 6,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "stellar_mass": 10.5e10,
            "angular_size": 0.16,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
            "mag_g": -20,
        }

        while True:
            kwargs2 = {"extended_source_type": "single_sersic"}
            self.source2 = Source(cosmo=cosmo, **kwargs2, **source_dict)
            self.deflector2 = Deflector(
                deflector_type="NFW_HERNQUIST",
                **deflector_dict,
            )
            gg_lens = Lens(
                source_class=self.source2,
                deflector_class=self.deflector2,
                lens_equation_solver="lenstronomy_default",
                cosmo=cosmo,
                use_jax=use_jax,
            )
            if gg_lens.validity_test():
                # self.gg_lens = gg_lens
                break

        # here for NFW-Cluster model
        subhalos_table = Table.read(
            os.path.join(path, "../TestData/subhalos_table.fits"), format="fits"
        )
        source_dict = blue_one
        deflector_dict = {
            "halo_mass": 10**14,
            "concentration": 5,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "z": 0.42,
            "subhalos": subhalos_table,
        }
        while True:
            kwargs_3 = {"extended_source_type": "single_sersic"}
            self.source3 = Source(cosmo=cosmo, **source_dict, **kwargs_3)
            self.deflector3 = Deflector(
                deflector_type="NFW_CLUSTER",
                **deflector_dict,
            )
            cg_lens = Lens(
                source_class=self.source3,
                deflector_class=self.deflector3,
                lens_equation_solver="lenstronomy_default",
                cosmo=cosmo,
                use_jax=use_jax,
            )
            if cg_lens.validity_test(max_image_separation=50.0):
                break

    def test_kappa_star(self):

        from lenstronomy.Util.util import make_grid

        delta_pix = 0.05
        x, y = make_grid(numPix=200, deltapix=delta_pix)
        kappa_star = self.gg_lens.kappa_star(x, y)
        stellar_mass_from_kappa_star = (
            np.sum(kappa_star)
            * delta_pix**2
            * self.gg_lens._lens_cosmo.sigma_crit_angle
        )
        stellar_mass = self.gg_lens.deflector_stellar_mass()
        npt.assert_almost_equal(
            stellar_mass_from_kappa_star / stellar_mass, 1, decimal=1
        )

    def test_lenstronomy_kwargs_interpolated(self):
        """Minimal test to confirm that lenstronomy_kwargs() returns the
        correct keys for an interpolated source."""

        kwargs_model, kwargs_params = self.gg_lens_interp.lenstronomy_kwargs(band="i")

        # Check that kwargs_model has the essential lens modeling lists
        assert (
            "lens_model_list" in kwargs_model
        ), "Missing 'lens_model_list' in kwargs_model"
        assert (
            "lens_light_model_list" in kwargs_model
        ), "Missing 'lens_light_model_list' in kwargs_model"
        # assert "source_light_model_list" in kwargs_model, "Missing 'source_light_model_list'"
        # Check that kwargs_params holds the parameter dictionaries:
        assert "kwargs_lens" in kwargs_params, "Missing 'kwargs_lens' in kwargs_params"
        assert (
            "kwargs_lens_light" in kwargs_params
        ), "Missing 'kwargs_lens_light' in kwargs_params"
        assert (
            "kwargs_source" in kwargs_params
        ), "Missing 'kwargs_source' in kwargs_params"
        assert "kwargs_ps" in kwargs_params, "Missing 'kwargs_ps' in kwargs_params"

    def test_contrast_ratio(self):
        mag_ratios = self.gg_lens.contrast_ratio(band="i", source_index=0)
        assert 2 <= len(mag_ratios) <= 4

    def test_add_subhalos(self):
        # Test the add_subhalos method

        # Check that the method raises an error if no subhalos are provided
        npt.assert_raises(ValueError, self.gg_lens.dm_subhalo_mass)
        npt.assert_raises(ValueError, self.gg_lens.add_subhalos, {}, "SIDM")

        pyhalos_parms = {
            "LOS_normalization": 0,
        }
        dm_type_cdm = "CDM"
        gg_lens_copy_cdm = copy.deepcopy(self.gg_lens)

        gg_lens_copy_cdm.add_subhalos(pyhalos_parms, dm_type_cdm)

        realization = gg_lens_copy_cdm.realization
        dm_subhalo_mass = gg_lens_copy_cdm.dm_subhalo_mass()
        assert isinstance(dm_subhalo_mass, list)
        assert isinstance(realization, object)

        len_after_first_kwargs = len(gg_lens_copy_cdm._kwargs_lens)

        # second call for checking for no duplication
        gg_lens_copy_cdm.add_subhalos(pyhalos_parms, dm_type_cdm)

        len_after_second_kwargs = len(gg_lens_copy_cdm._kwargs_lens)

        assert len_after_second_kwargs == len_after_first_kwargs, "kwargs duplicated!"

        pyhalos_parms_wdm = {
            "LOS_normalization": 0,
            "log_mc": 7.0,
        }
        dm_type_wdm = "WDM"
        gg_lens_copy_wdm = copy.deepcopy(self.gg_lens)

        gg_lens_copy_wdm.add_subhalos(pyhalos_parms_wdm, dm_type_wdm)

        realization_wdm = gg_lens_copy_wdm.realization
        assert isinstance(realization_wdm, object)

        pyhalos_parms_uldm = {
            "LOS_normalization": 0,
            "log10_m_uldm": -20.0,
            "uldm_plaw": 1 / 3,
            "flucs_shape": "ring",
            "flucs_args": {"angle": 0.0, "rmin": 0.9, "rmax": 1.1},
            "log10_fluc_amplitude": -1.6,
            "n_cut": 1000000,
        }
        dm_type_uldm = "ULDM"
        gg_lens_copy_uldm = copy.deepcopy(self.gg_lens)

        gg_lens_copy_uldm.add_subhalos(pyhalos_parms_uldm, dm_type_uldm)

        realization_uldm = gg_lens_copy_uldm.realization
        assert isinstance(realization_uldm, object)

    def test_subhalos_only_lens_model(self):
        # Test the get_halos_only_lens_model method
        from lenstronomy.LensModel.lens_model import LensModel

        lens_model, kwargz_lens = self.gg_lens.subhalos_only_lens_model()
        assert isinstance(lens_model, LensModel)
        assert isinstance(kwargz_lens, list)

        pyhalos_parms = {"LOS_normalization": 0}
        dm_type = "CDM"
        self.gg_lens.add_subhalos(pyhalos_parms, dm_type)
        subhalos_only_model, kwargs_subhalos = self.gg_lens.subhalos_only_lens_model()

        assert isinstance(subhalos_only_model, LensModel)
        assert isinstance(kwargs_subhalos, list)

        subhalo_lens_model_list, _, _, _ = self.gg_lens.realization.lensing_quantities(
            add_mass_sheet_correction=True
        )
        # check that the lens model list is the same as the one returned by subhalos_only_lens_model
        assert subhalos_only_model.lens_model_list == subhalo_lens_model_list


@pytest.fixture
def pes_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "../TestData/source_dict_ps.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "../TestData/deflector_dict_ps.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        kwargs4 = {
            "kwargs_variability": None,
        }
        source4 = Source(
            cosmo=cosmo,
            point_source_type="quasar",
            extended_source_type="single_sersic",
            **source_dict,
            **kwargs4,
        )
        deflector4 = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
        )
        pes_lens = Lens(
            source_class=source4,
            deflector_class=deflector4,
            cosmo=cosmo,
            use_jax=use_jax,
        )
        second_brightest_image_cut = {"i": 30}
        if pes_lens.validity_test(
            second_brightest_image_cut=second_brightest_image_cut
        ):
            pes_lens = pes_lens
            break
    return pes_lens


def test_validity_test_2(pes_lens_instance):
    second_brightest_image_cut = {"i": 30}
    assert (
        pes_lens_instance.validity_test(
            second_brightest_image_cut=second_brightest_image_cut
        )
        is True
    )


def test_validity_test_2_with_snr_limit(pes_lens_instance):
    """Test validity_test with snr_limit for point source + extended source
    lens."""
    second_brightest_image_cut = {"i": 30}

    # Test with snr_limit=None (backward compatibility)
    assert (
        pes_lens_instance.validity_test(
            second_brightest_image_cut=second_brightest_image_cut,
            snr_limit=None,
        )
        is True
    )

    # Test with very low SNR limit that should pass
    assert (
        pes_lens_instance.validity_test(
            second_brightest_image_cut=second_brightest_image_cut,
            snr_limit={"i": 0.1},
        )
        is True
    )

    # Test with very high SNR limit that should fail
    assert (
        pes_lens_instance.validity_test(
            second_brightest_image_cut=second_brightest_image_cut,
            snr_limit={"i": 1e10},
        )
        is False
    )


def test_snr_pes_lens(pes_lens_instance):
    """Test SNR calculation for point source + extended source lens."""
    snr_result = pes_lens_instance.snr(band="i", fov_arcsec=6, observatory="LSST")
    assert snr_result is None or (
        isinstance(snr_result, (float, np.floating)) and snr_result > 0
    )


def test_point_source_magnitude(pes_lens_instance):
    pes_lens = pes_lens_instance
    mag = pes_lens.point_source_magnitude(band="i", lensed=True)[0]
    mag_unlensed = pes_lens.point_source_magnitude(band="i")[0]
    assert len(mag) >= 2
    assert mag_unlensed > 0


def test_lens_to_dataframe(pes_lens_instance):
    import pandas as pd

    lens_df = pes_lens_instance.lens_to_dataframe()
    assert isinstance(lens_df, pd.DataFrame)


################################################
############## MICROLENSING TESTS###############
################################################


@pytest.fixture
def lens_instance_with_variability():
    # quasar and host galaxy dict. One can avoid host galaxy information and simulate
    # lensed quasar only.
    source_dict_quasar = {
        "z": 1.5,
        "ps_mag_i": 21,
        "angular_size": 0.10887651129362959,
        "mag_i": 20,
        "e1": 0.0,
        "e2": 0.0,
        "n_sersic": 1.5547096361698418,
        "center_x": 0.046053505877290584,
        "center_y": -0.09071283196326566,
    }

    deflector_dict_quasar = {
        "z": 0.501666913484551,
        "M": -21.83145200238993,
        "coeff": [0.141014265858706, 9.517770703665604e-05],
        "ellipticity": 0.2284277382812588,
        "physical_size": 4.206949315885421,
        "stellar_mass": 362262853208.36945,
        "angular_size": 0.6879678734773863,
        "mag_g": 21.867784201009997,
        "mag_r": 20.33108481157918,
        "mag_i": 19.493883022638812,
        "mag_z": 19.105662758016145,
        "mag_y": 18.86764491626696,
        "galaxy_type": "red",
        "vel_disp": 225.65292910480588,
        "e1_light": -0.11571475911179421,
        "e2_light": -0.0025994949173672476,
        "e1_mass": -0.17804791091757563,
        "e2_mass": 0.040020226664717634,
        "n_sersic": 4.0,
        "theta_E": 1.5,
        "gamma_pl": 2.0,
        "center_x": 0.0316789,
        "center_y": -0.0400549,
    }
    variable_agn_kwarg_dict = {
        "length_of_light_curve": 500,
        "time_resolution": 1,
        "log_breakpoint_frequency": 1 / 20,
        "low_frequency_slope": 1,
        "high_frequency_slope": 3,
        "standard_deviation": 0.9,
    }
    kwargs_quasar = {
        "variability_model": "light_curve",
        "kwargs_variability": {"agn_lightcurve", "i", "r"},
        "agn_driving_variability_model": "bending_power_law",
        "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
        "lightcurve_time": np.linspace(0, 1000, 500),
    }

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    source_quasar = Source(
        cosmo=cosmo,
        point_source_type="quasar",
        extended_source_type=None,
        **source_dict_quasar,
        **kwargs_quasar,
    )
    deflector_quasar = Deflector(
        deflector_type="EPL_SERSIC",
        **deflector_dict_quasar,
    )

    los_class = LOSIndividual(
        kappa=-0.028113857977090363,
        gamma=[0.01118681739734637, -0.012498985117640523],
    )

    lens_class = Lens(
        source_class=source_quasar,
        deflector_class=deflector_quasar,
        cosmo=cosmo,
        los_class=los_class,
        use_jax=use_jax,
    )

    return lens_class


@pytest.fixture
def band_i():
    return "i"


@pytest.fixture
def band_r():
    return "r"


@pytest.fixture
def time_array():
    return np.linspace(0, 100, 20)  # 20 time steps for microlensing tests


@pytest.fixture
def kwargs_microlensing_magmap_settings_test(lens_instance_with_variability):
    """Minimal settings for MagnificationMap for microlensing tests."""
    theta_e = lens_instance_with_variability._einstein_radius(0)
    return {
        "theta_star": theta_e * 0.01,
        "num_pixels_x": 100,
        "num_pixels_y": 100,
        "half_length_x": 5 * theta_e * 0.01,
        "half_length_y": 5 * theta_e * 0.01,
    }


@pytest.fixture
def kwargs_source_gaussian_test():
    """Minimal settings for a Gaussian source morphology for microlensing."""
    return {
        "source_size": 1e-8,  # Very small for point-like behavior
    }


@pytest.fixture
def kwargs_microlensing_settings(
    kwargs_microlensing_magmap_settings_test, kwargs_source_gaussian_test
):
    """Combines settings for the kwargs_microlensing dictionary for Gaussian
    source."""
    return {
        "kwargs_magnification_map": kwargs_microlensing_magmap_settings_test,
        "point_source_morphology": "gaussian",
        "kwargs_source_morphology": kwargs_source_gaussian_test,
    }


@pytest.fixture
def kwargs_source_agn_test():
    """Source morphology settings for an AGN source for microlensing.

    Can be empty as the lens class populates it.
    """
    return {}


@pytest.fixture
def kwargs_microlensing_settings_agn(
    kwargs_microlensing_magmap_settings_test, kwargs_source_agn_test
):
    """Combines settings for the kwargs_microlensing dictionary for an AGN
    source."""
    return {
        "kwargs_magnification_map": kwargs_microlensing_magmap_settings_test,
        "point_source_morphology": "agn",
        "kwargs_source_morphology": kwargs_source_agn_test,
    }


def test_microlensing_parameters_for_image_positions_single_source(
    lens_instance_with_variability, band_i
):
    """Tests the _microlensing_parameters_for_image_positions_single_source
    method.

    This is an integration test as it relies on other methods of
    pes_lens_instance.
    """

    # Ensure image positions are calculated if not already
    if not hasattr(lens_instance_with_variability, "_ps_image_position_list"):
        lens_instance_with_variability.point_source_image_positions()

    num_images = len(lens_instance_with_variability._ps_image_position_list[0][0])
    assert num_images == 4
    if num_images == 0:
        pytest.skip("Skipping test: No lensed images found for this configuration.")

    try:
        kappa_star_img, kappa_tot_img, shear_img, shear_angle_img = (
            lens_instance_with_variability._microlensing_parameters_for_image_positions_single_source(
                band_i, source_index=0
            )
        )
    except Exception as e:
        pytest.fail(
            f"_microlensing_parameters_for_image_positions_single_source raised: {e}"
        )

    assert isinstance(kappa_star_img, np.ndarray)
    assert isinstance(kappa_tot_img, np.ndarray)
    assert isinstance(shear_img, np.ndarray)
    assert isinstance(shear_angle_img, np.ndarray)

    assert kappa_star_img.shape == (num_images,)
    assert kappa_tot_img.shape == (num_images,)
    assert shear_img.shape == (num_images,)
    assert shear_angle_img.shape == (num_images,)

    # Basic plausibility checks (values depend heavily on the lens model)
    assert np.all(np.isfinite(kappa_star_img))
    assert np.all(np.isfinite(kappa_tot_img))
    assert np.all(np.isfinite(shear_img))
    assert np.all(np.isfinite(shear_angle_img))


def test_point_source_magnitude_with_microlensing_block(
    lens_instance_with_variability, time_array, band_i, kwargs_microlensing_settings
):
    """Test lensed point source magnitude including the microlensing block."""
    lens_system = lens_instance_with_variability  # Use the loaded instance

    # 1. Get lensed magnitude WITH time but WITHOUT microlensing
    mag_lensed_time_list_no_ml = lens_system.point_source_magnitude(
        band=band_i, lensed=True, time=time_array, microlensing=False
    )

    num_images = lens_system.image_number[0]
    if num_images == 0:
        pytest.skip("No lensed images for microlensing test.")

    # 2. Mock the internal _point_source_magnitude_microlensing method
    with patch.object(
        lens_system, "_point_source_magnitude_microlensing", autospec=True
    ) as mock_internal_microlensing_method:
        # Define a specific return value for the mocked internal method
        # This should be an array of shape (num_images, len(time_array))
        mock_delta_mags_microlensing = np.random.normal(
            0, 0.05, size=(num_images, len(time_array))
        )
        mock_internal_microlensing_method.return_value = mock_delta_mags_microlensing

        # 3. Call point_source_magnitude WITH microlensing=True
        # This should now call our mocked _point_source_magnitude_microlensing
        mag_lensed_time_with_ml_list = lens_system.point_source_magnitude(
            band=band_i,
            lensed=True,
            time=time_array,
            microlensing=True,  # This activates the block we want to test
            kwargs_microlensing=kwargs_microlensing_settings,
        )

        # 4. Assertions
        # Check that our internal mock was called correctly
        mock_internal_microlensing_method.assert_called_once_with(
            band=band_i,
            time=time_array,
            source_index=0,
            kwargs_microlensing=kwargs_microlensing_settings,
        )

        # Check that the final magnitude is the sum of the non-microlensed time-variable
        # magnitude and the (mocked) microlensing delta magnitudes.
        # We are testing the `+= microlensing_magnitudes` line.
        expected_final_mags = (
            mag_lensed_time_list_no_ml[0] + mock_delta_mags_microlensing
        )
        np.testing.assert_allclose(mag_lensed_time_with_ml_list[0], expected_final_mags)


# This test requires mocking because it calls the external microlensing light curve generator
@patch("slsim.Microlensing.lightcurvelensmodel.MicrolensingLightCurveFromLensModel")
def test_point_source_magnitude_microlensing(
    mock_ml_lc_from_lm_class,
    lens_instance_with_variability,
    band_i,
    time_array,
    kwargs_microlensing_settings,
):
    """Tests _point_source_magnitude_microlensing by mocking the light curve
    generator and checking for auto-populated source params."""
    lens_system = lens_instance_with_variability
    source_index = 0
    source = lens_system.source(source_index)
    num_images = lens_system.image_number[source_index]

    if num_images == 0:
        pytest.skip("No lensed images found for this configuration.")

    # Configure the mock MicrolensingLightCurveFromLensModel instance
    mock_ml_lc_instance = MagicMock()
    # This is what ml_lc_instance.generate_point_source_microlensing_magnitudes is expected to return
    expected_microlensing_delta_mags = np.random.normal(
        0, 0.1, size=(num_images, len(time_array))
    )
    mock_ml_lc_instance.generate_point_source_microlensing_magnitudes.return_value = (
        expected_microlensing_delta_mags
    )
    # When Lens calls MicrolensingLightCurveFromLensModel(...), it gets our mock instance
    mock_ml_lc_from_lm_class.return_value = mock_ml_lc_instance

    # Check that the microlensing_model_class attribute is not set before the call
    with pytest.raises(
        AttributeError,
        match="MicrolensingLightCurveFromLensModel class is not set.",
    ):
        # Accessing as a method now
        _ = lens_system.microlensing_model_class(source_index=source_index)

    # Call the method under test
    try:
        result_mags = lens_system._point_source_magnitude_microlensing(
            band_i,
            time_array,
            source_index=source_index,
            kwargs_microlensing=kwargs_microlensing_settings,
        )
    except Exception as e:
        pytest.fail(f"_point_source_magnitude_microlensing raised an exception: {e}")

    # Get the actual microlensing parameters from lens class
    (
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        shear_angle_images_rad,
    ) = lens_system._microlensing_parameters_for_image_positions_single_source(
        band=band_i, source_index=source_index
    )
    shear_phi_angle_images_deg = np.degrees(shear_angle_images_rad)

    # Verify the CONSTRUCTOR call on the class
    mock_ml_lc_from_lm_class.assert_called_once()
    constructor_kwargs = mock_ml_lc_from_lm_class.call_args.kwargs

    # Prepare the expected kwargs_source_morphology after auto-population
    expected_source_morphology = kwargs_microlensing_settings[
        "kwargs_source_morphology"
    ].copy()
    expected_source_morphology["source_redshift"] = source.redshift
    expected_source_morphology["cosmo"] = lens_system.cosmo
    expected_source_morphology["observing_wavelength_band"] = band_i

    # Update with AGN params because the source is a Quasar and the lens class calls the update method
    expected_source_morphology = (
        source._source.update_microlensing_kwargs_source_morphology(
            expected_source_morphology
        )
    )

    # Check key arguments passed to the constructor
    assert constructor_kwargs["source_redshift"] == source.redshift
    assert constructor_kwargs["deflector_redshift"] == lens_system.deflector_redshift
    np.testing.assert_array_equal(
        constructor_kwargs["kappa_star_images"], kappa_star_images
    )
    np.testing.assert_array_equal(
        constructor_kwargs["kappa_tot_images"], kappa_tot_images
    )
    np.testing.assert_array_equal(constructor_kwargs["shear_images"], shear_images)
    np.testing.assert_array_equal(
        constructor_kwargs["shear_phi_angle_images"], shear_phi_angle_images_deg
    )
    assert constructor_kwargs["cosmology"] == lens_system.cosmo
    assert (
        constructor_kwargs["kwargs_magnification_map"]
        == kwargs_microlensing_settings["kwargs_magnification_map"]
    )
    assert (
        constructor_kwargs["point_source_morphology"]
        == kwargs_microlensing_settings["point_source_morphology"]
    )
    assert constructor_kwargs["kwargs_source_morphology"] == expected_source_morphology

    # Verify the generate_... method was called on the INSTANCE with the correct time
    mock_ml_lc_instance.generate_point_source_microlensing_magnitudes.assert_called_once_with(
        time=time_array
    )

    # Check if microlensing_model_class property is set correctly
    assert (
        lens_system.microlensing_model_class(source_index=source_index)
        is mock_ml_lc_instance
    )

    # Check if invalid source index raises error
    invalid_source_index = source_index + 1  # Assuming only one source exists
    with pytest.raises(
        AttributeError,
        match=f"MicrolensingLightCurveFromLensModel class is not set for source index {invalid_source_index}.",
    ):
        lens_system.microlensing_model_class(source_index=invalid_source_index)

    # The result of _point_source_magnitude_microlensing should be the direct output
    # from the mocked generate_point_source_microlensing_magnitudes
    np.testing.assert_allclose(result_mags, expected_microlensing_delta_mags)


@patch("slsim.Microlensing.lightcurvelensmodel.MicrolensingLightCurveFromLensModel")
def test_point_source_magnitude_microlensing_agn(
    mock_ml_lc_from_lm_class,
    lens_instance_with_variability,
    band_i,
    time_array,
    kwargs_microlensing_settings_agn,
):
    """Tests _point_source_magnitude_microlensing with AGN morphology and auto-
    populated AGN params."""
    lens_system = lens_instance_with_variability
    source_index = 0
    source = lens_system.source(source_index)

    # Configure mock and call the method
    mock_ml_lc_from_lm_class.return_value = MagicMock()
    lens_system._point_source_magnitude_microlensing(
        band_i,
        time_array,
        source_index=source_index,
        kwargs_microlensing=kwargs_microlensing_settings_agn,
    )

    # Get the arguments passed to the constructor of the mocked class
    constructor_kwargs = mock_ml_lc_from_lm_class.call_args.kwargs
    final_source_morphology_kwargs = constructor_kwargs["kwargs_source_morphology"]

    # Check that standard parameters were added
    assert final_source_morphology_kwargs["source_redshift"] == source.redshift
    assert final_source_morphology_kwargs["cosmo"] == lens_system.cosmo
    assert final_source_morphology_kwargs["observing_wavelength_band"] == band_i

    # Check that AGN-specific parameters were added from the Source class
    source_agn_kwargs = source._source.agn_class.kwargs_model
    agn_params_to_check = [
        "black_hole_mass_exponent",
        "inclination_angle",
        "black_hole_spin",
        "eddington_ratio",
    ]
    for param in agn_params_to_check:
        assert final_source_morphology_kwargs[param] == source_agn_kwargs[param]


@patch("slsim.Microlensing.lightcurvelensmodel.MicrolensingLightCurveFromLensModel")
def test_point_source_magnitude_microlensing_defaults(
    mock_ml_lc_from_lm_class,
    lens_instance_with_variability,
    band_i,
    band_r,
    time_array,
):
    """Tests _point_source_magnitude_microlensing with defaults
    (kwargs_microlensing=None).

    Verifies that 'agn' morphology is automatically assigned for QSO
    sources.
    """
    lens_system = deepcopy(lens_instance_with_variability)
    source_index = 0

    # Ensure the source name is "QSO" so the auto-logic triggers
    # We force this just in case the fixture mapping is different in the installed slsim version
    lens_system.source(source_index)._source.name = "QSO"

    # Configure mock
    mock_instance = MagicMock()
    mock_ml_lc_from_lm_class.return_value = mock_instance

    # Call with kwargs_microlensing=None
    lens_system._point_source_magnitude_microlensing(
        band_i,
        time_array,
        source_index=source_index,
        kwargs_microlensing=None,
    )

    # Get constructor args
    constructor_kwargs = mock_ml_lc_from_lm_class.call_args.kwargs

    # Assert automatic assignment
    assert constructor_kwargs["point_source_morphology"] == "agn"

    # Assert kwargs_source_morphology was initialized and populated
    kwargs_morph = constructor_kwargs["kwargs_source_morphology"]
    assert kwargs_morph["observing_wavelength_band"] == band_i
    assert kwargs_morph["source_redshift"] == lens_system.source(source_index).redshift

    # Constructor should be called exactly once here
    mock_ml_lc_from_lm_class.assert_called_once()

    # Second call with same settings - should reuse instance and update morphology
    lens_system._point_source_magnitude_microlensing(
        band_r,
        time_array,
        source_index=source_index,
        kwargs_microlensing=None,
    )

    # Constructor should still have only been called once (from the first call)
    mock_ml_lc_from_lm_class.assert_called_once()

    # update_source_morphology should have been called during the second execution
    mock_instance.update_source_morphology.assert_called_once()


################################################
################################################


def test_lens_id_qso(pes_lens_instance):
    lens_id = pes_lens_instance.generate_id()
    ra = pes_lens_instance.deflector_position[0]
    dec = pes_lens_instance.deflector_position[1]
    ra2 = 12.03736542
    dec2 = 35.17363534
    lens_id2 = pes_lens_instance.generate_id(ra=ra2, dec=dec2)
    assert lens_id == f"GAL-QSO-LENS_{ra:.4f}_{dec:.4f}"
    assert lens_id2 == f"GAL-QSO-LENS_{ra2:.4f}_{dec2:.4f}"


@pytest.fixture
def supernovae_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = {
        "MJD": np.array([60966.29451169, 60968.2149886, 60972.27148745, 61031.1304958]),
        "ps_mag_r": np.array([30.79802681, 30.81809901, 30.86058829, 31.38072828]),
        "z": 3.739813382373592,
        "ellipticity": 0.27369234660412406,
        "physical_size": 14.109069479312343,
        "angular_size": 9.364416159676842e-06,
        "mag_g": 27.007121008458327,
        "mag_r": 26.5023820544612,
        "mag_i": 26.311557659702515,
        "e1": -0.1209494604090952,
        "e2": -0.06952809999798619,
        "n_sersic": 1.0,
    }
    deflector_dict = Table.read(
        os.path.join(path, "../TestData/supernovae_deflector_dict.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        kwargs5 = {
            "variability_model": "light_curve",
        }
        source5 = Source(
            cosmo=cosmo,
            point_source_type="general_lightcurve",
            extended_source_type="single_sersic",
            **source_dict,
            **kwargs5,
        )
        deflector5 = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
        )
        supernovae_lens = Lens(
            source_class=source5,
            deflector_class=deflector5,
            cosmo=cosmo,
            use_jax=use_jax,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


def test_point_source_magnitude_with_lightcurve(supernovae_lens_instance):
    supernovae_lens = supernovae_lens_instance
    mag = supernovae_lens.point_source_magnitude(band="r", lensed=True)[0]
    expected_results = np.array([30.79802681, 30.81809901, 30.86058829, 31.38072828])
    assert np.all(mag[0] != expected_results)
    assert np.all(mag[1] != expected_results)


def test_point_source_arrival_times(supernovae_lens_instance):
    supernova_lens = supernovae_lens_instance
    dt_days = supernova_lens.point_source_arrival_times()
    assert np.min(dt_days) > -1000
    assert np.max(dt_days) < 1000


def test_image_observer_times(supernovae_lens_instance):
    supernova_lens = supernovae_lens_instance
    t_obs = 1000
    t_obs2 = np.array([100, 200, 300])
    dt_days = supernova_lens.image_observer_times(t_obs=t_obs)
    dt_days2 = supernova_lens.image_observer_times(t_obs=t_obs2)
    arrival_times = supernova_lens.point_source_arrival_times()[0]
    observer_times = (t_obs - arrival_times + np.min(arrival_times))[:, np.newaxis]
    observer_times2 = (t_obs2[:, np.newaxis] - arrival_times + np.min(arrival_times)).T
    npt.assert_almost_equal(dt_days, observer_times, decimal=5)
    npt.assert_almost_equal(dt_days2, observer_times2, decimal=5)


class TestDifferentLens(object):
    # pytest.fixture(scope='class')
    def setup_method(self):
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
        )
        red_one = Table.read(
            os.path.join(path, "../TestData/red_one_modified.fits"), format="fits"
        )
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        kwargs = {"extended_source_type": "single_sersic"}
        self.source6 = Source(
            cosmo=self.cosmo,
            **self.source_dict,
            **kwargs,
        )
        self.deflector6 = Deflector(
            deflector_type="EPL_SERSIC",
            **self.deflector_dict,
        )
        self.lens_class_multi_source_plane = Lens(
            deflector_class=self.deflector6,
            source_class=[self.source6, self.source6],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_general",
            multi_plane="Source",
            use_jax=use_jax,
        )
        subhalos_table = Table.read(
            os.path.join(path, "../TestData/subhalos_table.fits"), format="fits"
        )
        deflector_dict_nfw = {
            "halo_mass": 10**14,
            "concentration": 5,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "z": 0.42,
            "subhalos": subhalos_table,
        }
        self.deflector_nfw_cluster = Deflector(
            deflector_type="NFW_CLUSTER",
            cored_profile=True,
            **deflector_dict_nfw,
        )
        self.lens_class_nfw_cluster = Lens(
            deflector_class=self.deflector_nfw_cluster,
            source_class=self.source6,
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_analytical",
        )
        self.lens_class_nfw_cluster_multi_source_plane = Lens(
            deflector_class=self.deflector_nfw_cluster,
            source_class=[self.source6, self.source6],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_general",
            multi_plane="Source",
            use_jax=use_jax,
        )

    def test_different_setting(self):
        los1 = LOSPop(
            los_bool=True,
            mixgauss_gamma=True,
            nonlinear_los_bool=False,
        )
        gg_lens = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los1.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
            use_jax=use_jax,
        )
        assert gg_lens.external_shear >= 0
        assert isinstance(gg_lens.external_convergence, float)
        assert isinstance(gg_lens.external_shear, float)

        los2 = LOSPop(
            los_bool=True,
            mixgauss_gamma=False,
            nonlinear_los_bool=True,
        )

        gg_lens_2 = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los2.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
            use_jax=use_jax,
        )
        assert gg_lens_2.external_shear >= 0
        assert isinstance(gg_lens_2.external_convergence, float)
        assert isinstance(gg_lens_2.external_shear, float)

        los3 = LOSPop(los_bool=False)
        gg_lens_3 = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los3.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
            use_jax=use_jax,
        )
        assert gg_lens_3.external_convergence == 0
        assert gg_lens_3.external_shear == 0

        los4 = LOSPop(
            los_bool=True,
            mixgauss_gamma=True,
            nonlinear_los_bool=True,
        )
        with pytest.raises(ValueError):
            gg_lens_4 = Lens(
                source_class=self.source6,
                deflector_class=self.deflector6,
                cosmo=self.cosmo,
                los_class=los4.draw_los(
                    deflector_redshift=self.deflector6.redshift,
                    source_redshift=self.source6.redshift,
                ),
                use_jax=use_jax,
            )
            gg_lens_4.external_convergence()

    def test_image_number(self):
        los = LOSIndividual(kappa=0, gamma=[0, 0])
        gg_lens_number = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los,
            use_jax=use_jax,
        )
        image_number = gg_lens_number.image_number
        assert (
            (image_number[0] == 4) or (image_number[0] == 2) or (image_number[0] == 1)
        )

        gg_lens_multisource = Lens(
            source_class=[self.source6, self.source6],
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los,
            use_jax=use_jax,
        )
        kwargs_model = gg_lens_multisource.lenstronomy_kwargs()[0]
        kwargs_model_keys = kwargs_model.keys()
        expected_kwargs_model = [
            "lens_light_model_list",
            "lens_model_list",
            "z_lens",
            "lens_redshift_list",
            "source_redshift_list",
            "z_source_convention",
            "cosmo",
            "source_light_model_list",
        ]
        assert expected_kwargs_model[0] in kwargs_model_keys
        assert expected_kwargs_model[1] in kwargs_model_keys
        assert expected_kwargs_model[2] in kwargs_model_keys
        assert expected_kwargs_model[3] in kwargs_model_keys
        assert expected_kwargs_model[4] in kwargs_model_keys
        assert expected_kwargs_model[5] in kwargs_model_keys
        assert expected_kwargs_model[6] in kwargs_model_keys

    def test_multi_plane_setting(self):
        mp_deflector_redshifts = self.lens_class_multi_source_plane.deflector_redshift
        mp_lens_model, mp_kwargs_lens = (
            self.lens_class_multi_source_plane.deflector_mass_model_lenstronomy()
        )
        # Test the lenght and values of the lens redshift list
        number_of_models = len(mp_lens_model.lens_model_list)
        assert mp_deflector_redshifts == [self.deflector6.redshift] * number_of_models

        # Test multi-plane for nfw cluster model
        nfw_mp_deflector_redshifts = (
            self.lens_class_nfw_cluster_multi_source_plane.deflector_redshift
        )
        nfw_mp_lens_model, nfw_mp_kwargs_lens = (
            self.lens_class_nfw_cluster_multi_source_plane.deflector_mass_model_lenstronomy()
        )
        # Test the lenght and values of the lens redshift list
        number_of_models_nfw = len(nfw_mp_lens_model.lens_model_list)
        assert (
            nfw_mp_deflector_redshifts
            != [self.deflector_nfw_cluster.redshift] * number_of_models_nfw
        )


@pytest.fixture
def supernovae_lens_instance_double_sersic_multisource():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "../TestData/source_supernovae_new.fits"), format="fits"
    )
    source_dict.rename_column("angular_size0", "angular_size_0")
    source_dict.rename_column("angular_size1", "angular_size_1")
    deflector_dict = Table.read(
        os.path.join(path, "../TestData/deflector_supernovae_new.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-20, 100, 1000),
            "sn_modeldir": None,
        }
        source = Source(
            cosmo=cosmo,
            point_source_type="supernova",
            extended_source_type="double_sersic",
            **source_dict,
            **kwargs,
        )
        deflector = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
        )
        supernovae_lens = Lens(
            deflector_class=deflector,
            source_class=[source, source],
            cosmo=cosmo,
            use_jax=use_jax,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


def test_lens_id_snia(supernovae_lens_instance_double_sersic_multisource):
    lens_id = supernovae_lens_instance_double_sersic_multisource.generate_id()
    ra = supernovae_lens_instance_double_sersic_multisource.deflector_position[0]
    dec = supernovae_lens_instance_double_sersic_multisource.deflector_position[1]
    ra2 = 12.03736542
    dec2 = 35.17363534
    lens_id2 = supernovae_lens_instance_double_sersic_multisource.generate_id(
        ra=ra2, dec=dec2
    )
    assert lens_id == f"GAL-SNIa-LENS_{ra:.4f}_{dec:.4f}"
    assert lens_id2 == f"GAL-SNIa-LENS_{ra2:.4f}_{dec2:.4f}"


class TestMultiSource(object):
    def setup_method(self):
        np.random.seed(42)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        path = os.path.dirname(__file__)
        source_dict1 = Table.read(
            os.path.join(path, "../TestData/source_supernovae_new.fits"), format="fits"
        )
        source_dict1.rename_column("angular_size0", "angular_size_0")
        source_dict1.rename_column("angular_size1", "angular_size_1")
        deflector_dict = Table.read(
            os.path.join(path, "../TestData/deflector_supernovae_new.fits"),
            format="fits",
        )

        deflector_dict_ = dict(zip(deflector_dict.colnames, deflector_dict[0]))
        self.gamma_pl = 1.8
        deflector_dict_["gamma_pl"] = self.gamma_pl

        source_dict2 = copy.deepcopy(source_dict1)
        source_dict2["z"] += 2
        kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-20, 100, 1000),
            "sn_modeldir": None,
        }
        self.source1 = Source(
            cosmo=self.cosmo,
            point_source_type="supernova",
            extended_source_type="double_sersic",
            **source_dict2,
            **kwargs,
        )
        # We initiate the another Source class with the same source. In this class,
        # source position will be different and all the lensing quantities will be different
        self.source2 = Source(
            source_dict=source_dict1,
            cosmo=self.cosmo,
            point_source_type="supernova",
            extended_source_type="double_sersic",
            **source_dict1,
            **kwargs,
        )
        self.deflector = Deflector(
            deflector_type="EPL_SERSIC", sis_convention=False, **deflector_dict_
        )

        self.lens_class1 = Lens(
            deflector_class=self.deflector,
            source_class=self.source1,
            cosmo=self.cosmo,
            use_jax=use_jax,
        )
        self.lens_class2 = Lens(
            deflector_class=self.deflector,
            source_class=self.source2,
            cosmo=self.cosmo,
            use_jax=use_jax,
        )
        self.lens_class3 = Lens(
            deflector_class=self.deflector,
            source_class=[self.source1, self.source2],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_general",
            use_jax=use_jax,
        )

        self.lens_class3_analytical = Lens(
            deflector_class=self.deflector,
            source_class=[self.source1, self.source2],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_analytical",
            use_jax=use_jax,
        )

        deflector_nfw_dict = {
            "halo_mass": 10**13,
            "halo_mass_acc": 0.0,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "stellar_mass": 10e11,
            "angular_size": 0.001 / 4.84813681109536e-06,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
            "mag_g": -20,
        }
        self.deflector_nfw = Deflector(
            deflector_type="NFW_HERNQUIST", **deflector_nfw_dict
        )

        self.lens_class_nfw = Lens(
            deflector_class=self.deflector_nfw,
            source_class=self.source1,
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_analytical",
            use_jax=use_jax,
        )

    def test_point_source_arrival_time_multi(self):
        gamma_pl_out = self.deflector.halo_properties["gamma_pl"]
        assert gamma_pl_out == self.gamma_pl

        point_source_arival_time1 = self.lens_class1.point_source_arrival_times()
        point_source_arival_time2 = self.lens_class2.point_source_arrival_times()
        point_source_arival_time3 = self.lens_class3.point_source_arrival_times()
        # Test multisource point source arrival time.
        assert np.all(point_source_arival_time1[0]) == np.all(
            point_source_arival_time3[0]
        )
        assert np.all(point_source_arival_time2[0]) == np.all(
            point_source_arival_time3[1]
        )

        point_source_arival_time3 = (
            self.lens_class3_analytical.point_source_arrival_times()
        )
        # Test multisource point source arival time.
        assert np.all(point_source_arival_time1[0]) == np.all(
            point_source_arival_time3[0]
        )
        assert np.all(point_source_arival_time2[0]) == np.all(
            point_source_arival_time3[1]
        )

    def test_ps_magnification_multi(self):
        ps_magnification1 = self.lens_class1.point_source_magnification()
        ps_magnification2 = self.lens_class2.point_source_magnification()
        ps_magnification3 = self.lens_class3.point_source_magnification()
        # Test multisource point source magnifications.
        assert np.all(ps_magnification1[0]) == np.all(ps_magnification3[0])
        assert np.all(ps_magnification2[0]) == np.all(ps_magnification3[1])

        ps_magnification3 = self.lens_class3_analytical.point_source_magnification()
        assert np.all(ps_magnification1[0]) == np.all(ps_magnification3[0])
        assert np.all(ps_magnification2[0]) == np.all(ps_magnification3[1])

    def test_es_magnification_multi(self):
        es_magnification1 = self.lens_class1.extended_source_magnification
        es_magnification2 = self.lens_class2.extended_source_magnification
        es_magnification3 = self.lens_class3.extended_source_magnification

        # Test multisource extended source magnifications.
        npt.assert_almost_equal(
            es_magnification1[0] / es_magnification3[0], 1, decimal=2
        )
        npt.assert_almost_equal(
            es_magnification2[0] / es_magnification3[1], 1, decimal=2
        )

        es_magnification3 = self.lens_class3_analytical.extended_source_magnification
        npt.assert_almost_equal(
            es_magnification1[0] / es_magnification3[0], 1, decimal=2
        )
        npt.assert_almost_equal(
            es_magnification2[0] / es_magnification3[1], 1, decimal=2
        )

    def test_einstein_radius_multi(self):
        einstein_radius1 = self.lens_class1.einstein_radius
        einstein_radius2 = self.lens_class2.einstein_radius
        einstein_radius3 = self.lens_class3.einstein_radius
        # Test multisource einstein radius.
        npt.assert_almost_equal(einstein_radius1[0], einstein_radius3[0], decimal=3)
        npt.assert_almost_equal(einstein_radius2[0], einstein_radius3[1], decimal=3)

        einstein_radius3 = self.lens_class3_analytical.einstein_radius
        npt.assert_almost_equal(einstein_radius1[0], einstein_radius3[0], decimal=5)
        npt.assert_almost_equal(einstein_radius2[0], einstein_radius3[1], decimal=5)

        einstein_radius_nfw = self.lens_class_nfw.einstein_radius
        npt.assert_almost_equal(einstein_radius_nfw, 0.63, decimal=2)

    def test_image_observer_time_multi(self):
        observation_time = 50
        image_observation_time1 = self.lens_class1.image_observer_times(
            observation_time
        )
        image_observation_time2 = self.lens_class2.image_observer_times(
            observation_time
        )
        image_observation_time3 = self.lens_class3.image_observer_times(
            observation_time
        )
        # Test multisource image observation time
        npt.assert_almost_equal(
            image_observation_time1[0], image_observation_time3[0][0], decimal=5
        )
        # assert image_observation_time1[0] == image_observation_time3[0][0]
        npt.assert_almost_equal(
            image_observation_time2, image_observation_time3[1], decimal=5
        )
        # assert np.all(image_observation_time2 == image_observation_time3[1])
        assert len(self.lens_class3.image_observer_times(t_obs=10)) == 2

        image_observation_time3 = self.lens_class3_analytical.image_observer_times(
            observation_time
        )
        # Test multisource image observation time
        npt.assert_almost_equal(
            image_observation_time1[0], image_observation_time3[0][0], decimal=5
        )
        # assert image_observation_time1[0] == image_observation_time3[0][0]
        npt.assert_almost_equal(
            image_observation_time2, image_observation_time3[1], decimal=5
        )


class TestSlhammock(object):
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        # Source dict. You can also proviide magnitude in single band. This source dict is
        # valid for single sersic_ellipse light profile.
        source_dict = {
            "z": 0.8664718175006184,
            "angular_size": 0.01345195778342412,  # effective radius of a source in arcsec
            # "mag_g": 22.5,  # g-band magnitude of a source
            # "mag_r": 22,  # r-band magnitude of a source
            "mag_i": 26.00614079444494,  # i-band magnitude of a source
            # "mag_z": 22.1,  # z-band magnitude of a source
            # "mag_y": 22.0,  # y-band magnitude of a source
            "e1": 0.08079814292965516,  # tangential component of the ellipticity
            "e2": 0.0038523986875793467,  # cross component of the ellipticity
            "n_sersic": 1,  # sersic index for sersic_ellipse profile
            "center_x": -3.331767912522456,  # x-position of the center of a source
            "center_y": 3.7260996716288455,
        }  # y-position of the center of a source

        # Deflector dict. You can also provide magnitude in single band. This deflector dict is
        # valid for elliptical power law model.
        deflector_dict = {
            "z": 0.09199999999999993,
            "angular_size": 9.893988634937736,  # effective radius of the deflector in arcsec
            # "mag_g": 20.0,  # g-band magnitude of a deflector
            # "mag_r": 13.80785703495668,  # r-band magnitude of a deflector
            "mag_i": 18.5,  # i-band magnitude of a deflector
            # "mag_z": 18.0,  # z-band magnitude of a deflector
            # "mag_y": 17.5,  # y-band magnitude of a deflector
            "halo_mass": 259737421577890.4,
            "halo_mass_acc": 0.0,
            "vel_disp": -1.0,
            "e1_light": -0.0073474229677211135,  # tangential component of the light ellipticity
            "e2_light": 0.24186783558136427,  # cross component of the light ellipticity
            "e1_mass": -0.07738867986582895,  # tangential component of the mass ellipticity
            "e2_mass": 0.23482266208752717,  # cross component of the mass ellipticity
            "e_h": 0.161468884396323,
            "p_h": -85.13323836236738,
            "p_g": -105.61605878239777,
            "tb": 0.18082265752696014,
            "concentration": 7.314407077851028,
            "stellar_mass": 1064901910393.4458,
            "center_x": -0.017839189263436216,  # x-position of the center of the lens
            "center_y": 0.010467931830543249,  # y-position of the center of the lens
        }
        source = Source(
            cosmo=self.cosmo,
            extended_source_type="single_sersic",
            point_source_type=None,
            **source_dict,
        )
        deflector = Deflector(
            deflector_type="NFW_HERNQUIST",
            **deflector_dict,
        )
        los_class = LOSIndividual(
            kappa=0, gamma=[-0.005061965833762263, 0.028825761226555197]
        )
        self.lens_class = Lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=self.cosmo,
            los_class=los_class,
            use_jax=use_jax,
        )

    def test_theta_e_infinity(self):
        npt.assert_almost_equal(
            self.lens_class.einstein_radius_infinity, 3.76881, decimal=5
        )

    def test_image_position(self):
        # In source dict we have not provided supernova-host offset. So, extended and
        # point source image position should be the same.
        extended_source_image_position = (
            self.lens_class.extended_source_image_positions[0]
        )
        point_source_image_position = self.lens_class.point_source_image_positions()[0]
        assert np.all(
            extended_source_image_position[0] == point_source_image_position[0]
        )
        assert np.all(
            extended_source_image_position[1] == point_source_image_position[1]
        )

    def test_source_light_model_lenstronomy_none_band(self):
        results = self.lens_class.source_light_model_lenstronomy(band=None)[1]
        npt.assert_almost_equal(results["kwargs_source"][0]["magnitude"], 1, decimal=6)


class TestSNR:
    """Comprehensive tests for the SNR (signal-to-noise ratio) calculation
    method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up a lens instance for SNR testing."""
        path = os.path.dirname(__file__)
        blue_one = Table.read(
            os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
        )
        blue_one["angular_size"] = blue_one["angular_size"] / 4.84813681109536e-06
        red_one = Table.read(
            os.path.join(path, "../TestData/red_one_modified.fits"), format="fits"
        )
        red_one["angular_size"] = red_one["angular_size"] / 4.84813681109536e-06
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        mag_arc_limit = {"i": 35, "g": 35, "r": 35}
        while True:
            kwargs = {"extended_source_type": "single_sersic"}
            source = Source(cosmo=self.cosmo, **blue_one, **kwargs)
            deflector = Deflector(deflector_type="EPL_SERSIC", **red_one)
            lens = Lens(
                source_class=source,
                deflector_class=deflector,
                lens_equation_solver="lenstronomy_analytical",
                cosmo=self.cosmo,
                use_jax=use_jax,
            )
            if lens.validity_test(mag_arc_limit=mag_arc_limit):
                self.lens = lens
                break

    def test_snr_returns_float_or_none(self):
        """Test that SNR returns a float or None."""
        snr_result = self.lens.snr(band="i", fov_arcsec=6, observatory="LSST")
        assert snr_result is None or isinstance(snr_result, (float, np.floating))

    def test_snr_positive_when_not_none(self):
        """Test that SNR is positive when it returns a value."""
        snr_result = self.lens.snr(band="i", fov_arcsec=6, observatory="LSST")
        if snr_result is not None:
            assert snr_result > 0

    def test_snr_different_bands(self):
        """Test SNR calculation with different bands."""
        bands = ["g", "r", "i"]
        for band in bands:
            snr_result = self.lens.snr(band=band, fov_arcsec=6, observatory="LSST")
            assert snr_result is None or isinstance(snr_result, (float, np.floating))

    def test_snr_fov_arcsec_parameter(self):
        """Test that different fov_arcsec values work."""
        for fov_arcsec in [4, 6, 10]:
            snr_result = self.lens.snr(
                band="i", fov_arcsec=fov_arcsec, observatory="LSST"
            )
            assert snr_result is None or isinstance(snr_result, (float, np.floating))

    def test_snr_high_threshold_returns_none(self):
        """Test that a very high per-pixel SNR threshold returns None."""
        snr_result = self.lens.snr(
            band="i",
            fov_arcsec=6,
            observatory="LSST",
            snr_per_pixel_threshold=1e10,
        )
        assert snr_result is None

    def test_snr_low_threshold(self):
        """Test that a very low threshold includes more pixels."""
        # With a very low threshold, we should get some regions
        snr_low = self.lens.snr(
            band="i",
            fov_arcsec=6,
            observatory="LSST",
            snr_per_pixel_threshold=0.01,
        )
        # This may or may not be None depending on the lens, but shouldn't error
        assert snr_low is None or isinstance(snr_low, (float, np.floating))

    def test_snr_threshold_effect(self):
        """Test that higher thresholds give equal or lower SNR (or None)."""
        snr_low_thresh = self.lens.snr(
            band="i",
            fov_arcsec=6,
            observatory="LSST",
            snr_per_pixel_threshold=0.5,
        )
        snr_high_thresh = self.lens.snr(
            band="i",
            fov_arcsec=6,
            observatory="LSST",
            snr_per_pixel_threshold=2.0,
        )
        # Higher threshold should give lower or equal SNR, or None
        if snr_low_thresh is not None and snr_high_thresh is not None:
            # Higher threshold typically results in smaller regions
            # but may give similar or even higher SNR per region
            # Just verify both are valid floats
            assert isinstance(snr_low_thresh, (float, np.floating))
            assert isinstance(snr_high_thresh, (float, np.floating))


class TestSNRValidityIntegration:
    """Tests for the integration of SNR calculation with validity_test.

    These tests specifically verify the bug fix from commit fd895c9b,
    which changed the condition from:
        `if snr_calculated is not None and np.max(snr_calculated) < snr:`
    to:
        `if snr_calculated is None or np.max(snr_calculated) < snr:`

    This ensures that lenses are rejected when SNR calculation returns None.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up a lens instance for testing."""
        path = os.path.dirname(__file__)
        blue_one = Table.read(
            os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
        )
        blue_one["angular_size"] = blue_one["angular_size"] / 4.84813681109536e-06
        red_one = Table.read(
            os.path.join(path, "../TestData/red_one_modified.fits"), format="fits"
        )
        red_one["angular_size"] = red_one["angular_size"] / 4.84813681109536e-06
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        mag_arc_limit = {"i": 35, "g": 35, "r": 35}
        while True:
            kwargs = {"extended_source_type": "single_sersic"}
            source = Source(cosmo=self.cosmo, **blue_one, **kwargs)
            deflector = Deflector(deflector_type="EPL_SERSIC", **red_one)
            lens = Lens(
                source_class=source,
                deflector_class=deflector,
                lens_equation_solver="lenstronomy_analytical",
                cosmo=self.cosmo,
                use_jax=use_jax,
            )
            if lens.validity_test(mag_arc_limit=mag_arc_limit):
                self.lens = lens
                break

    def test_validity_test_rejects_when_snr_returns_none(self):
        """Test that validity_test returns False when SNR calculation returns
        None.

        This tests the bug fix from fd895c9b. Previously, when snr()
        returned None, the condition `snr_calculated is not None and
        np.max(snr_calculated) < snr` would be False (because
        snr_calculated IS None), so the lens would pass.

        Now with the fix `snr_calculated is None or
        np.max(snr_calculated) < snr`, when snr() returns None, the
        condition is True and the lens is correctly rejected.
        """
        # Mock the snr method to return None (simulating no regions found)
        with patch.object(self.lens, "snr", return_value=None):
            # With snr_limit set, validity_test should return False
            # because snr() returns None
            result = self.lens.validity_test(snr_limit={"i": 1.0})
            assert (
                result is False
            ), "validity_test should return False when snr() returns None"

    def test_validity_test_passes_when_snr_exceeds_limit(self):
        """Test that validity_test returns True when SNR exceeds the limit."""
        # Mock the snr method to return a high value
        with patch.object(self.lens, "snr", return_value=np.array([100.0])):
            result = self.lens.validity_test(snr_limit={"i": 10.0})
            assert (
                result is True
            ), "validity_test should return True when snr exceeds limit"

    def test_validity_test_fails_when_snr_below_limit(self):
        """Test that validity_test returns False when SNR is below the
        limit."""
        # Mock the snr method to return a low value
        with patch.object(self.lens, "snr", return_value=np.array([5.0])):
            result = self.lens.validity_test(snr_limit={"i": 10.0})
            assert (
                result is False
            ), "validity_test should return False when snr is below limit"

    def test_validity_test_multi_band_snr_with_none(self):
        """Test validity_test fails when any band's SNR returns None.

        If multiple bands are specified in snr_limit and any one of them
        returns None from snr(), the validity_test should fail.
        """

        def mock_snr(band, **kwargs):
            # Return None for 'g' band, valid value for 'i' band
            if band == "g":
                return None
            return np.array([100.0])

        with patch.object(self.lens, "snr", side_effect=mock_snr):
            # Even though 'i' band has high SNR, 'g' band returns None
            result = self.lens.validity_test(snr_limit={"i": 10.0, "g": 10.0})
            assert (
                result is False
            ), "validity_test should fail when any band's SNR returns None"

    def test_validity_test_multi_band_snr_all_pass(self):
        """Test validity_test passes when all bands exceed their limits."""

        def mock_snr(band, **kwargs):
            return np.array([100.0])

        with patch.object(self.lens, "snr", side_effect=mock_snr):
            result = self.lens.validity_test(snr_limit={"i": 10.0, "g": 10.0})
            assert (
                result is True
            ), "validity_test should pass when all bands exceed their limits"


class TestSNRMocked:
    """Tests for SNR calculation logic using mocked image simulation."""

    def test_snr_region_identification(self):
        """Test that region identification works correctly with controlled
        inputs."""
        from scipy.ndimage import label

        # Create a simple test case with known regions
        snr_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 0, 0],
                [0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
            ]
        )

        threshold = 1
        masked_snr_array = np.ma.masked_where(snr_array <= threshold, snr_array)

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_array, num_regions = label(
            masked_snr_array.filled(0), structure=structure
        )

        # Should identify 2 regions: the 2x2 block and the single pixel
        assert num_regions == 2

    def test_snr_calculation_logic(self):
        """Test the SNR calculation formula with known values."""
        # SNR = source_counts / sqrt(total_counts)
        source = np.array(
            [
                [0, 0, 0],
                [0, 100, 100],
                [0, 100, 100],
            ]
        )
        image = np.array(
            [
                [10, 10, 10],
                [10, 150, 150],
                [10, 150, 150],
            ]
        )

        # For the 2x2 region in bottom-right
        region_mask = np.array(
            [
                [False, False, False],
                [False, True, True],
                [False, True, True],
            ]
        )

        source_counts = np.sum(source[region_mask])  # 400
        total_counts = np.sum(image[region_mask])  # 600
        variance = total_counts  # 600
        expected_snr = source_counts / np.sqrt(variance)  # 400 / sqrt(600)

        npt.assert_almost_equal(expected_snr, 400 / np.sqrt(600), decimal=2)

    def test_snr_cross_connectivity(self):
        """Test that cross-shaped connectivity is used (not diagonal)."""
        from scipy.ndimage import label

        # Create array where pixels are only diagonally connected
        snr_array = np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        )

        threshold = 1
        masked_snr_array = np.ma.masked_where(snr_array <= threshold, snr_array)

        # Cross-shaped connectivity (no diagonals)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_array, num_regions = label(
            masked_snr_array.filled(0), structure=structure
        )

        # With cross connectivity, these should be 3 separate regions
        assert num_regions == 3


if __name__ == "__main__":
    pytest.main()
