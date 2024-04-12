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
                kwargs_variability= {"supernovae_lightcurve", "i"},
                sn_type="Ia",
                sn_absolute_mag_band="bessellb",
                sn_absolute_zpsys="ab",
                cosmo=cosmo,
                source_type="point_plus_extended",
                light_profile="double_sersic",
                lightcurve_time=np.linspace(
        -20, 100, 1000
    ),
            )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens

def test_update_coolest_from_slsim_and_create_slsim_from_coolest(
        supernovae_lens_instance):
    # Define test data
    path = os.path.dirname(__file__)
    lens_class = supernovae_lens_instance
    test_path = path + "/TestData/"
    test_file_name = "coolest_template"
    test_band = "i"
    test_mag_zero_point = 27
    # Call the function
    result = update_coolest_from_slsim(
        lens_class, test_path, test_file_name, test_band, test_mag_zero_point
    )
    assert result is None


"""def test_create_slsim_from_coolest(supernovae_lens_instance):
    # Define test data
    path = os.path.dirname(__file__)
    lens_class = supernovae_lens_instance
    test_path = path + "/TestData/"
    test_file_name = "coolest_template_update"
    test_mag_zero_point = 27
    expected_result = lens_class.lenstronomy_kwargs(band="i")

    # Call the function
    result = create_slsim_from_coolest(
        test_path, test_file_name, test_mag_zero_point
    )

    # Assert the result
    assert len(result) == len(expected_result)
    assert (
        result[1]["kwargs_lens"][0]["theta_E"]
        == expected_result[1]["kwargs_lens"][0]["theta_E"]
    )"""
