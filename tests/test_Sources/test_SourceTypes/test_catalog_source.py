import numpy as np
import os
import pathlib
import pytest

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from slsim.Pipelines import SkyPyPipeline
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.catalog_source import CatalogSource
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.Lenses.lens import Lens
from slsim.ImageSimulation.image_simulation import lens_image
from slsim.Util.param_util import gaussian_psf

hst_cosmos_path = os.path.join(
    str(pathlib.Path(__file__).parent.parent.parent),
    "TestData",
    "test_COSMOS_23.5_training_sample",
)
cosmos_web_path = os.path.join(
    str(pathlib.Path(__file__).parent.parent.parent),
    "TestData",
    "test_COSMOS_WEB",
)


class TestCatalogSource:
    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {
            "z": 3.5,
            "mag_i": 20.3,
            "n_sersic": 0.8,
            "angular_size": 0.3,  # arcseconds
            "e1": 0.19697001616620306,
            "e2": 0.040998265256000574,
            "center_x": 0.0,
            "center_y": 0.0,
            "phi_G": 0,
        }
        self.source1 = CatalogSource(
            cosmo=cosmo,
            catalog_path=hst_cosmos_path,
            catalog_type="HST_COSMOS",
            **source_dict,
        )

        self.source2 = CatalogSource(
            cosmo=cosmo,
            catalog_path=cosmos_web_path,
            catalog_type="COSMOS_WEB",
            **source_dict,
        )

    def test_init_catalog(self):
        assert id(self.source1.final_catalog) == id(
            CatalogSource.processed_hst_cosmos_catalog
        )
        assert id(self.source2.final_catalog) == id(
            CatalogSource.processed_cosmos_web_catalog
        )

    def test_kwargs_extended_source_light(self):
        light_model_list1, results = self.source1.kwargs_extended_light(band="i")
        _, results2 = self.source1.kwargs_extended_light(band=None)

        with fits.open(hst_cosmos_path + "/test_galaxy_images_23.5.fits") as file:
            image_ref = file[2].data

        np.testing.assert_allclose(results[0]["image"], image_ref)
        assert results[0]["magnitude"] == 20.3
        assert results2[0]["magnitude"] == 1

        light_model_list2, results3 = self.source2.kwargs_extended_light(band="i")
        assert light_model_list1 == light_model_list2
        assert results3[0]["magnitude"] == 20.3

    def test_redshift(self):
        assert self.source1.redshift == 3.5

    def test_angular_size(self):
        assert self.source1.angular_size == 0.3

    def test_ellipticity(self):
        e1, e2 = self.source1.ellipticity
        assert e1 == 0.19697001616620306
        assert e2 == 0.040998265256000574

    def test_extended_source_magnitude(self):
        assert self.source1.extended_source_magnitude("i") == 20.3
        with pytest.raises(ValueError):
            self.source1.extended_source_magnitude("g")

    def test_extended_source_light_model(self):
        source_model, kwargs_light = self.source1.kwargs_extended_light()
        assert source_model[0] == "INTERPOL"

    def test_raises(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {
            "z": 0.5,
            "mag_i": 20.3,
            "n_sersic": 0.8,
            "angular_size": 999,  # arcseconds
            "e1": 0.09697001616620306,
            "e2": 0.040998265256000574,
            "center_x": 0.0,
            "center_y": 0.0,
            "phi_G": 0,
        }
        np.testing.assert_raises(
            ValueError,
            CatalogSource,
            cosmo=cosmo,
            catalog_path=hst_cosmos_path,
            catalog_type="incorrect",
            **source_dict,
        )

        source = CatalogSource(
            cosmo=cosmo,
            catalog_path=hst_cosmos_path,
            catalog_type="HST_COSMOS",
            **source_dict,
        )
        np.testing.assert_raises(
            ValueError,
            source.kwargs_extended_light,
        )

    def test_sersic_fallback(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {
            "z": 0.5,
            "mag_i": 20.3,
            "n_sersic": 0.8,
            "angular_size": 1.3,  # arcseconds
            "e1": 0.09697001616620306,
            "e2": 0.040998265256000574,
            "center_x": 0.0,
            "center_y": 0.0,
            "phi_G": 0,
        }

        source1 = CatalogSource(
            cosmo=cosmo,
            catalog_path=hst_cosmos_path,
            catalog_type="HST_COSMOS",
            max_scale=0.1,
            sersic_fallback=True,
            **source_dict,
        )
        source2 = SingleSersic(**source_dict)
        assert source1.kwargs_extended_light() == source2.kwargs_extended_light()

        source1 = CatalogSource(
            cosmo=cosmo,
            catalog_path=cosmos_web_path,
            catalog_type="COSMOS_WEB",
            max_scale=0.1,
            sersic_fallback=True,
            **source_dict,
        )
        assert source1.kwargs_extended_light() == source2.kwargs_extended_light()


def test_source1():
    # Set match_n_sersic to False

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # approximate HST/ACS F814W zero‑point
    m_zp = 25.5

    # build source dict for SLSIM
    source_dict = {
        "z": 0.5,
        "mag_i": 20.3,
        "n_sersic": 0.8,
        "angular_size": 0.3,  # arcseconds
        "e1": 0.09697001616620306,
        "e2": 0.040998265256000574,
        "center_x": 0.0,
        "center_y": 0.0,
        "phi_G": 0,
    }
    source1 = Source(
        extended_source_type="catalog_source",
        cosmo=cosmo,
        catalog_path=hst_cosmos_path,
        catalog_type="HST_COSMOS",
        match_n_sersic=False,
        **source_dict,
    )
    source2 = Source(extended_source_type="single_sersic", cosmo=cosmo, **source_dict)

    # dummy, zero‑mass deflector
    deflector = Deflector(
        deflector_type="EPL_SERSIC",
        **{
            "z": 0.5,
            "theta_E": 0.0,
            "e1_light": 0.0,
            "e2_light": 0.0,
            "e1_mass": 0.0,
            "e2_mass": 0.0,
            "gamma_pl": 2.0,
            "angular_size": 0.05,
            "n_sersic": 1.0,
            "mag_g": 99.0,
            "mag_r": 99.0,
            "mag_i": 99.0,
            "mag_z": 99.0,
            "mag_y": 99.0,
        },
    )

    lens_class1 = Lens(
        source_class=source1,
        deflector_class=deflector,
        cosmo=cosmo,
    )
    lens_class2 = Lens(
        source_class=source2,
        deflector_class=deflector,
        cosmo=cosmo,
    )

    # build transform matrix = pixel_scale arcsec/pix
    pixscale = 0.03
    transform_pix2angle = np.array([[pixscale, 0], [0, pixscale]])
    psf_kernel = gaussian_psf(fwhm=0.1, delta_pix=pixscale, num_pix=21)
    num_pix = int(source_dict["angular_size"] / pixscale * 4)
    # simulate source‑only image
    cosmos_image = lens_image(
        lens_class=lens_class1,
        band="i",
        mag_zero_point=m_zp,
        num_pix=num_pix,
        psf_kernel=psf_kernel,
        transform_pix2angle=transform_pix2angle,
        exposure_time=None,
        t_obs=None,
        std_gaussian_noise=None,
        with_source=True,
        with_deflector=False,
    )
    sersic_image = lens_image(
        lens_class=lens_class2,
        band="i",
        mag_zero_point=m_zp,
        num_pix=num_pix,
        psf_kernel=psf_kernel,
        transform_pix2angle=transform_pix2angle,
        exposure_time=None,
        t_obs=None,
        std_gaussian_noise=None,
        with_source=True,
        with_deflector=False,
    )
    np.testing.assert_allclose(
        np.sum(sersic_image), np.sum(cosmos_image), atol=20, rtol=0.1
    )
    assert sersic_image.shape == cosmos_image.shape


def test_source2():
    # Set match_n_sersic to True

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # approximate HST/ACS F814W zero‑point
    m_zp = 25.5

    # build source dict for SLSIM
    source_dict = {
        "z": 0.5,
        "mag_i": 20.3,
        "n_sersic": 0.8,
        "angular_size": 0.3,  # arcseconds
        "e1": 0.09697001616620306,
        "e2": 0.040998265256000574,
        "center_x": 0.0,
        "center_y": 0.0,
        "phi_G": 0,
    }
    source1 = Source(
        extended_source_type="catalog_source",
        cosmo=cosmo,
        catalog_path=hst_cosmos_path,
        catalog_type="HST_COSMOS",
        match_n_sersic=True,
        **source_dict,
    )
    source2 = Source(extended_source_type="single_sersic", cosmo=cosmo, **source_dict)

    # dummy, zero‑mass deflector
    deflector = Deflector(
        deflector_type="EPL_SERSIC",
        **{
            "z": 0.5,
            "theta_E": 0.0,
            "e1_light": 0.0,
            "e2_light": 0.0,
            "e1_mass": 0.0,
            "e2_mass": 0.0,
            "gamma_pl": 2.0,
            "angular_size": 0.05,
            "n_sersic": 1.0,
            "mag_g": 99.0,
            "mag_r": 99.0,
            "mag_i": 99.0,
            "mag_z": 99.0,
            "mag_y": 99.0,
        },
    )

    lens_class1 = Lens(
        source_class=source1,
        deflector_class=deflector,
        cosmo=cosmo,
    )
    lens_class2 = Lens(
        source_class=source2,
        deflector_class=deflector,
        cosmo=cosmo,
    )

    # build transform matrix = pixel_scale arcsec/pix
    pixscale = 0.03
    transform_pix2angle = np.array([[pixscale, 0], [0, pixscale]])
    psf_kernel = gaussian_psf(fwhm=0.1, delta_pix=pixscale, num_pix=21)
    num_pix = int(source_dict["angular_size"] / pixscale * 4)
    # simulate source‑only image
    cosmos_image = lens_image(
        lens_class=lens_class1,
        band="i",
        mag_zero_point=m_zp,
        num_pix=num_pix,
        psf_kernel=psf_kernel,
        transform_pix2angle=transform_pix2angle,
        exposure_time=None,
        t_obs=None,
        std_gaussian_noise=None,
        with_source=True,
        with_deflector=False,
    )
    sersic_image = lens_image(
        lens_class=lens_class2,
        band="i",
        mag_zero_point=m_zp,
        num_pix=num_pix,
        psf_kernel=psf_kernel,
        transform_pix2angle=transform_pix2angle,
        exposure_time=None,
        t_obs=None,
        std_gaussian_noise=None,
        with_source=True,
        with_deflector=False,
    )
    np.testing.assert_allclose(
        np.sum(sersic_image), np.sum(cosmos_image), atol=20, rtol=0.1
    )
    assert sersic_image.shape == cosmos_image.shape


def test_galaxies():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 0.01 * u.deg**2

    pipeline = SkyPyPipeline(
        skypy_config=None,
        sky_area=sky_area,
        filters=None,
        cosmo=cosmo,
    )
    galaxy_list = pipeline.blue_galaxies
    kwargs_cut = {
        "band": "i",
        "band_max": 20,
        "z_min": 0.1,
        "z_max": 1.5,
    }
    source_simulation = Galaxies(
        galaxy_list=galaxy_list,
        kwargs_cut=kwargs_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        source_size=None,
        extended_source_type="catalog_source",
        extended_source_kwargs={
            "catalog_path": hst_cosmos_path,
            "catalog_type": "HST_COSMOS",
        },
    )
    source = source_simulation.draw_source()
    assert isinstance(source._source, CatalogSource)


if __name__ == "__main__":
    pytest.main()
