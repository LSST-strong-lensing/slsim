import numpy as np
from slsim.Util.coolest_slsim_interface import (
    update_coolest_from_slsim,
    create_slsim_from_coolest,
)
from slsim.lens import Lens
import os
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import pytest
from numpy import testing as npt


@pytest.fixture
def supernovae_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "TestData/source_supernovae_new.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "TestData/deflector_supernovae_new.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        supernovae_lens = Lens(
            deflector_dict=deflector_dict,
            source_dict=source_dict,
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_type="Ia",
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="double_sersic",
            lightcurve_time=np.linspace(-20, 100, 1000),
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


def test_update_coolest_from_slsim_and_create_slsim_from_coolest(
    supernovae_lens_instance,
):
    # Define test data
    path = os.path.dirname(__file__)
    lens_class = supernovae_lens_instance
    test_path = path + "/TestData/"
    test_file_name = "coolest_template"
    test_file_name_updated = "coolest_template_update"
    test_band = "i"
    test_mag_zero_point = 27
    expected_result = lens_class.lenstronomy_kwargs(band="i")
    # Call the function (this updates coolest file and saves in TestData)
    updated_coolest = update_coolest_from_slsim(
        lens_class, test_path, test_file_name, test_band, test_mag_zero_point
    )
    slsim_from_updated_coolest = create_slsim_from_coolest(
        test_path, test_file_name_updated, test_mag_zero_point
    )

    assert updated_coolest is None
    assert len(slsim_from_updated_coolest) == len(expected_result)
    assert (
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["theta_E"]
        == expected_result[1]["kwargs_lens"][0]["theta_E"]
    )
    assert np.all(slsim_from_updated_coolest[0] == expected_result[0])
    assert (
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["theta_E"]
        == expected_result[1]["kwargs_lens"][0]["theta_E"]
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["gamma"],
        expected_result[1]["kwargs_lens"][0]["gamma"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["e1"],
        expected_result[1]["kwargs_lens"][0]["e1"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["e2"],
        expected_result[1]["kwargs_lens"][0]["e2"],
    )
    assert (
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["center_x"]
        == expected_result[1]["kwargs_lens"][0]["center_x"]
    )
    assert (
        slsim_from_updated_coolest[1]["kwargs_lens"][0]["center_y"]
        == expected_result[1]["kwargs_lens"][0]["center_y"]
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_lens"][1]["gamma1"],
        expected_result[1]["kwargs_lens"][1]["gamma1"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_lens"][2]["kappa"],
        expected_result[1]["kwargs_lens"][2]["kappa"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_source"][0]["magnitude"],
        expected_result[1]["kwargs_source"][0]["magnitude"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_source"][1]["magnitude"],
        expected_result[1]["kwargs_source"][1]["magnitude"],
    )
    npt.assert_almost_equal(
        slsim_from_updated_coolest[1]["kwargs_ps"][0]["magnitude"],
        expected_result[1]["kwargs_ps"][0]["magnitude"],
    )

    os.remove(test_path + "coolest_template_update.json")
