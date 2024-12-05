import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.lens import Lens
from slsim.LsstSciencePipeline.opsim_pipeline import opsim_time_series_images_data
from slsim.LsstSciencePipeline.util_lsst import (
    opsim_variable_lens_injection,
    optimized_transient_event_time_mjd,
    transient_data_with_cadence,
)
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
import astropy.coordinates as coord
import astropy.units as u
import pytest


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
        source = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="single_sersic",
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
            lightcurve_time=np.linspace(0, np.pi, 100),
        )
        deflector = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict,
        )
        pes_lens = Lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=cosmo,
        )
        if pes_lens.validity_test():
            pes_lens = pes_lens
            break
    return pes_lens


def test_opsim_time_series_images_data():
    """Only run this test function if user has an OpSim database downloaded in
    the folder data/OpSim_database."""

    path = os.path.dirname(__file__)
    opsim_path = os.path.join(path, "../../data/OpSim_database/")
    if os.path.exists(opsim_path):
        files_in_folder = os.listdir(opsim_path)

        if files_in_folder:
            opsim_path_db = os.path.join(opsim_path, files_in_folder[0])

            # Test coordinates
            dec_points = np.array([-9.3, -36.2, -70.9, 19.9, -9.5, 6.7, -45.9, -37.1])
            ra_points = np.array([150.6, 4.1, 52.8, 33.2, 67.0, 124.4, -14.5, -166.9])

            # Create opsim_data instance
            opsim_data = opsim_time_series_images_data(
                ra_points,
                dec_points,
                obs_strategy="baseline_v3.0_10yrs",
                MJD_min=60000,
                MJD_max=60500,
                print_warning=False,
                opsim_path=opsim_path_db,
            )

            assert isinstance(opsim_data, list)  # is opsim_data a list?
            assert len(opsim_data) == len(
                dec_points
            )  # does it have the same length as number of points given?
            assert opsim_data[0].keys() == [
                "bkg_noise",  # does it contain the right data columns?
                "psf_kernel",
                "obs_time",
                "expo_time",
                "zero_point",
                "calexp_center",
                "band",
            ]
            assert isinstance(
                opsim_data[0]["bkg_noise"][0], float
            )  # are entries from bkg_noise floats?
            assert (
                opsim_data[0]["psf_kernel"][0].ndim == 2
            )  # is psf_kernel a 2 dimensional array?
            assert isinstance(
                opsim_data[0]["obs_time"][0], float
            )  # are entries from obs_time floats?
            assert isinstance(
                opsim_data[0]["expo_time"][0], float
            )  # are entries from expo_time floats?
            assert isinstance(
                opsim_data[0]["zero_point"][0], float
            )  # are entries from zero_point floats?
            assert isinstance(
                opsim_data[0]["calexp_center"][0], np.ndarray
            )  # is calexp_center an array?
            assert opsim_data[0]["calexp_center"][0].shape == (
                2,
            )  # is calexp_center an array of length 2?
            assert all(
                isinstance(item, float) for item in opsim_data[0]["calexp_center"][0]
            )  # are entries floats?
            assert isinstance(
                opsim_data[0]["band"][0], str
            )  # are entries from band strings?


def test_opsim_variable_lens_injection(pes_lens_instance):
    lens_class = pes_lens_instance

    # Load example opsim data format
    path = os.path.dirname(__file__)
    module_path, _ = os.path.split(path)
    expo_data = Table.read(os.path.join(path, "../TestData/expo_data_opsim.hdf5"))

    transform_pix2angle = np.array([[0.2, 0], [0, 0.2]])
    bands = ["g", "r", "i"]

    # Create example images
    results = opsim_variable_lens_injection(
        lens_class,
        bands=bands,
        num_pix=301,
        transform_pix2angle=transform_pix2angle,
        exposure_data=expo_data,
    )

    expo_bands = np.array([b for b in expo_data["band"]])
    mask = np.isin(expo_bands, bands)
    assert len(results) == len(expo_data[mask])


def test_transient_event_time_basic():
    mjd_times = np.linspace(58000, 58100, 200)
    lightcurve_range = (-50, 100)
    min_points = 100

    result = optimized_transient_event_time_mjd(mjd_times, lightcurve_range, min_points)

    assert result is not None
    start, end = result + lightcurve_range[0], result + lightcurve_range[1]
    points_in_range = np.sum((mjd_times >= start) & (mjd_times <= end))
    assert points_in_range >= min_points


def test_transient_event_time_no_valid_start():
    mjd_times = np.linspace(58000, 58100, 50)
    lightcurve_range = (-50, 100)
    min_points = 100

    result = optimized_transient_event_time_mjd(mjd_times, lightcurve_range, min_points)

    assert result is None


@pytest.fixture
def lens_class_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    path = os.path.dirname(__file__)
    source_dict1 = Table.read(
        os.path.join(path, "../TestData/source_supernovae_new.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "../TestData/deflector_supernovae_new.fits"), format="fits"
    )

    deflector_dict_ = dict(zip(deflector_dict.colnames, deflector_dict[0]))
    gamma_pl = 1.8
    deflector_dict_["gamma_pl"] = gamma_pl
    while True:
        source1 = Source(
            source_dict=source_dict1,
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="double_sersic",
            lightcurve_time=np.linspace(-50, 50, 50),
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i", "r", "z", "g", "y"},
            sn_type="Ia",
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
        )
        deflector = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict_,
            sis_convention=False,
        )

        lens_class1 = Lens(
            deflector_class=deflector,
            source_class=source1,
            cosmo=cosmo,
        )
        if lens_class1.validity_test():
            lens_class = lens_class1
            break
    return lens_class


@pytest.fixture
def exposure_data():
    num_obs = 20
    obs_times = np.linspace(59000, 59050, num_obs)
    bkg_noise = np.random.uniform(0.1, 0.5, num_obs)
    psf_fwhm = np.random.uniform(0.6, 1.2, num_obs)
    zero_points = np.random.uniform(25, 30, num_obs)
    expo_times = np.random.uniform(30, 60, num_obs)
    bands = ["g", "r", "i", "z"] * (num_obs // 4 + 1)

    # Create ra and dec values as Angle objects with units
    ra_points = coord.Angle(np.random.uniform(low=0, high=360, size=num_obs) * u.degree)
    ra_points = ra_points.wrap_at(180 * u.degree)

    dec_uniform = np.random.uniform(
        low=np.sin(np.radians(-75)), high=np.sin(np.radians(5)), size=num_obs
    )
    dec_points = coord.Angle(np.degrees(np.arcsin(dec_uniform)) * u.degree)

    # Combine ra_points and dec_points as a list of tuples, preserving units
    calexp_centers = list(zip(ra_points, dec_points))
    path = os.path.dirname(__file__)
    psf_kernel = np.load(
        os.path.join(path, "../TestData/psf_kernels_for_deflector.npy")
    )
    psf_kernel_list = [psf_kernel] * num_obs
    return Table(
        {
            "obs_time": obs_times,
            "bkg_noise": bkg_noise,
            "psf_fwhm": psf_fwhm,
            "zero_point": zero_points,
            "expo_time": expo_times,
            "calexp_center": calexp_centers,
            "band": bands[:num_obs],
            "psf_kernel": psf_kernel_list,
        }
    )


def test_transient_data_with_cadence(lens_class_instance, exposure_data):
    result = transient_data_with_cadence(
        lens_class=lens_class_instance,
        exposure_data=exposure_data,
        transform_pix2angle=np.array([[0.2, 0], [0, 0.2]]),
        num_pix=61,
        min_points=5,
    )
    colname = result.colnames
    assert isinstance(result, Table)
    assert len(result) >= 5
    assert len(colname) == 23  # 8 already existing col and 15 newly added.
    assert "obs_time_in_days" in colname
    assert "lens_id" in colname
    assert "mag_image_1" in colname
    assert "mag_image_2" in colname
    assert "mag_image_3" in colname
    assert "mag_image_4" in colname
    assert "mag_error_image_1_low" in colname
    assert "mag_error_image_1_high" in colname
    assert "mag_error_image_2_low" in colname
    assert "mag_error_image_2_high" in colname
    assert "mag_error_image_3_low" in colname
    assert "mag_error_image_3_high" in colname
    assert "mag_error_image_4_low" in colname
    assert "mag_error_image_4_high" in colname
    assert "lens_image" in colname


if __name__ == "__main__":
    pytest.main()
