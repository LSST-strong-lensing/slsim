import astropy.cosmology
from slsim.lens import Lens
from slsim.roman_image_simulation import simulate_roman_image

import os
import pytest


# NOTE: Galsim is required which is not supported on Windows
def test_simulate_roman_image():

    cosmo = astropy.cosmology.default_cosmology.get()

    z_lens = 0.2902115249535011
    z_source = 0.5876899931818929

    lens_stellar_mass = 286796906929.3925  # solar masses
    lens_velocity_dispersion = 295.2347999078027  # km/s

    deflector_dict = {
        "center_x": -0.007876281728887604,
        "center_y": 0.010633393703246008,
        "e1_mass": -0.004858808997848661,
        "e2_mass": 0.0075210751726143355,
        "stellar_mass": lens_stellar_mass,
        "e1_light": -0.023377277902774978,
        "e2_light": 0.05349948216860632,
        "vel_disp": lens_velocity_dispersion,
        "angular_size": 0.5300707454127908,
        "n_sersic": 4.0,
        "z": z_lens,
        "mag_F106": 17.5664222662219,
        "mag_F129": 17.269983557132853,
        "mag_F184": 17.00761457389914,
    }

    los_dict = {
        "gamma": [-0.03648819840013156, -0.06511863424492038],
        "kappa": 0.06020941823541971,
    }

    source_dict = {
        "angular_size": 0.1651633078964498,
        "center_x": 0.30298310338567075,
        "center_y": -0.3505004565139597,
        "e1": 0.06350855238708408,
        "e2": -0.08420760408362458,
        "mag_F106": 21.434711611915137,
        "mag_F129": 21.121205893763328,
        "mag_F184": 20.542431041034558,
        "n_sersic": 1.0,
        "z": z_source,
    }

    band = "F106"

    lens = Lens(
        source_dict=source_dict,
        deflector_dict=deflector_dict,
        los_dict=los_dict,
        cosmo=cosmo,
    )

    psf_directory = os.path.join(os.path.dirname(__file__), "..", "data", "webbpsf")
    final_array = simulate_roman_image(
        lens,
        band,
        num_pix=45,
        oversample=5,
        add_noise=True,
        psf_directory=psf_directory,
    )
    assert final_array.shape == (45, 45)
