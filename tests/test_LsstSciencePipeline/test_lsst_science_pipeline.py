import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.Lenses.lens import Lens
from slsim.LsstSciencePipeline.util_lsst import (
    variable_lens_injection,
    multiple_variable_lens_injection,
)
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
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
            "lightcurve_time": np.linspace(0, 1000, 1000),
        }
        source = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            point_source_type="quasar",
            extended_source_type="single_sersic",
            **kwargs_quasar,
            **source_dict,
        )
        deflector = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
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


def test_variable_lens_injection(pes_lens_instance):
    lens_class = pes_lens_instance
    path = os.path.dirname(__file__)
    expo_data = Table.read(
        os.path.join(path, "../TestData/expo_data_1.fits"), format="fits"
    )
    transf_matrix_single = np.array([[0.2, 0], [0, 0.2]])
    transform_matrices = [transf_matrix_single.copy() for _ in range(len(expo_data))]
    results = variable_lens_injection(
        lens_class,
        band="i",
        num_pix=301,
        transform_pix2angle=transform_matrices,
        exposure_data=expo_data,
    )
    assert len(results) == len(expo_data)


def test_multiple_variable_lens_injection(pes_lens_instance):
    lens_class = [pes_lens_instance, pes_lens_instance]
    path = os.path.dirname(__file__)
    expo_data_1 = Table.read(
        os.path.join(path, "../TestData/expo_data_1.fits"), format="fits"
    )
    expo_data_2 = Table.read(
        os.path.join(path, "../TestData/expo_data_2.fits"), format="fits"
    )
    expo_data = [expo_data_1, expo_data_2]
    transf_matrix_single = np.array([[0.2, 0], [0, 0.2]])
    transform_matrices = []
    for data in expo_data:
        transform_matrices.append(
            [transf_matrix_single.copy() for _ in range(len(data))]
        )
    results = multiple_variable_lens_injection(
        lens_class,
        band="i",
        num_pix=301,
        transform_matrices_list=transform_matrices,
        exposure_data_list=expo_data,
    )
    assert len(results) == len(expo_data)
