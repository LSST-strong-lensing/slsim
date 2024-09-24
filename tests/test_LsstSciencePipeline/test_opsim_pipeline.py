import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.lens import Lens
from slsim.LsstSciencePipeline.opsim_pipeline import (
    opsim_variable_lens_injection,
    opsim_time_series_images_data,
)
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
        pes_lens = Lens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            source_type="point_plus_extended",
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
            cosmo=cosmo,
            lightcurve_time=np.linspace(0, np.pi, 100),
        )
        if pes_lens.validity_test():
            pes_lens = pes_lens
            break
    return pes_lens


def test_opsim_time_series_images_data():

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
        opsim_path="../../data/OpSim_database/baseline_v3.0_10yrs.db"
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
    assert isinstance(opsim_data[0]["band"][0], str)  # are entries from band strings?


def test_opsim_variable_lens_injection(pes_lens_instance):
    lens_class = pes_lens_instance

    # Load example opsim data format
    expo_data = Table.read("../TestData/expo_data_opsim.hdf5", path="data")

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
