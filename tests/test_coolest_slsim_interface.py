from slsim.Util.coolest_slsim_interface import (update_coolest_from_lenstronomy_slsim,
                                                 create_lenstronomy_slsim_from_coolest)
import pickle
import os

def test_update_coolest_from_lenstronomy_slsim():
        # Define test data
        path = os.path.dirname(__file__)
        with open(path+"/TestData/lens_class.pickle", "rb") as f:
            # Load the object from the pickle file
            lens_class = pickle.load(f)
        test_path = path + "/TestData/"
        test_file_name = "coolest_template"
        test_band = "i"
        test_mag_zero_point = 27
        # Call the function
        result = update_coolest_from_lenstronomy_slsim(
            lens_class, test_path, test_file_name, test_band, test_mag_zero_point
        )
        assert result is None

def test_create_lenstronomy_slsim_from_coolest():
    # Define test data
    path = os.path.dirname(__file__)
    with open(path+"/TestData/lens_class.pickle", "rb") as f:
            # Load the object from the pickle file
            lens_class = pickle.load(f)
    test_path = path + "/TestData/"
    test_file_name = "coolest_template_update"
    test_mag_zero_point = 27
    expected_result = lens_class.lenstronomy_kwargs(band="i")

    # Call the function
    result = create_lenstronomy_slsim_from_coolest(
        test_path, test_file_name, test_mag_zero_point
    )

    # Assert the result
    assert len(result) == len(expected_result)
    assert result[1]["kwargs_lens"][0]["theta_E"] == \
          expected_result[1]["kwargs_lens"][0]["theta_E"]