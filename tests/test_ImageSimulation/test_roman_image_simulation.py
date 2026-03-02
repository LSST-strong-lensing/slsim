import astropy.cosmology
import numpy as np
import numpy.testing as npt
from slsim.Lenses.lens import Lens
from slsim.ImageSimulation.roman_image_simulation import (
    simulate_roman_image,
)
from slsim.ImageSimulation import image_quality_lenstronomy
from slsim.ImageSimulation.image_simulation import simulate_image
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.LOS.los_individual import LOSIndividual

import os
import pathlib
import pickle
import pytest

COSMO = astropy.cosmology.default_cosmology.get()

DEFLECTOR_DICT = {
    "center_x": -0.007876281728887604,
    "center_y": 0.010633393703246008,
    "e1_mass": -0.004858808997848661,
    "e2_mass": 0.0075210751726143355,
    "stellar_mass": 286796906929.3925,
    "e1_light": -0.023377277902774978,
    "e2_light": 0.05349948216860632,
    "vel_disp": 295.2347999078027,
    "angular_size": 0.5300707454127908,
    "n_sersic": 4.0,
    "z": 0.2902115249535011,
    "mag_F106": 17.5664222662219,
    "mag_F129": 17.269983557132853,
    "mag_F184": 17.00761457389914,
}

LOS_DICT = {
    "gamma": [-0.03648819840013156, -0.06511863424492038],
    "kappa": 0.06020941823541971,
}

SOURCE_DICT = {
    "angular_size": 0.1651633078964498,
    "center_x": 0.30298310338567075,
    "center_y": -0.3505004565139597,
    "e1": 0.06350855238708408,
    "e2": -0.08420760408362458,
    "mag_F106": 21.434711611915137,
    "mag_F129": 21.121205893763328,
    "mag_F184": 20.542431041034558,
    "n_sersic": 1.0,
    "z": 0.5876899931818929,
    "x_off": -0.053568932950377096,
    "y_off": 0.04383056304876015,
}

BAND = "F106"
kwargs_extended = {"extended_source_type": "single_sersic"}
source = Source(cosmo=COSMO, **kwargs_extended, **SOURCE_DICT)
pointsource_kwargs = {
    "variability_model": "light_curve",
    "kwargs_variability": ["supernovae_lightcurve", "F184", "F129", "F106"],
    "sn_type": "Ia",
    "sn_absolute_mag_band": "bessellb",
    "sn_absolute_zpsys": "ab",
    "lightcurve_time": np.linspace(-50, 100, 100),
    "sn_modeldir": None,
}
supernova_source = Source(
    cosmo=COSMO,
    point_source_type="supernova",
    extended_source_type="single_sersic",
    **pointsource_kwargs,
    **SOURCE_DICT,
)

deflector = Deflector(
    deflector_type="EPL_SERSIC",
    **DEFLECTOR_DICT,
)
LENS = Lens(
    source_class=source,
    deflector_class=deflector,
    los_class=LOSIndividual(**LOS_DICT),
    cosmo=COSMO,
)
SNIa_Lens = Lens(
    source_class=supernova_source,
    deflector_class=deflector,
    los_class=LOSIndividual(**LOS_DICT),
    cosmo=COSMO,
)

PSF_DIRECTORY = os.path.join(str(pathlib.Path(__file__).parent.parent), "TestData")

DETECTOR_KWARGS = {
    "detector": 3,
    "detector_pos": (1300, 3100),
    "ra": 29,
    "dec": -38,
}


# NOTE: Galsim is required which is not supported on Windows
def test_simulate_roman_image_with_psf_and_noise():
    final_image_galsim = simulate_roman_image(
        lens_class=LENS,
        band=BAND,
        num_pix=45,
        oversample=3,
        add_noise=True,
        subtract_mean_background=True,
        psf_directory=PSF_DIRECTORY,
        galsim_convolve=True,
        **DETECTOR_KWARGS,
    )

    final_image_lenstronomy = simulate_roman_image(
        lens_class=LENS,
        band=BAND,
        num_pix=45,
        oversample=3,
        add_noise=True,
        subtract_mean_background=True,
        psf_directory=PSF_DIRECTORY,
        galsim_convolve=False,
        **DETECTOR_KWARGS,
    )

    assert final_image_galsim.shape == (45, 45)
    assert final_image_lenstronomy.shape == (45, 45)
    npt.assert_almost_equal(
        (final_image_galsim - final_image_lenstronomy)
        / (final_image_galsim + final_image_lenstronomy + 1)
        / 2,
        0,
        decimal=1,
    )

    final_image_galsim = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=45,
        oversample=3,
        add_noise=True,
        psf_directory=PSF_DIRECTORY,
        galsim_convolve=True,
        **DETECTOR_KWARGS,
    )

    final_image_lenstronomy = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=45,
        oversample=3,
        add_noise=True,
        psf_directory=PSF_DIRECTORY,
        galsim_convolve=False,
        **DETECTOR_KWARGS,
    )

    assert final_image_galsim.shape == (45, 45)
    assert final_image_lenstronomy.shape == (45, 45)
    npt.assert_almost_equal(
        (final_image_galsim - final_image_lenstronomy)
        / (final_image_galsim + final_image_lenstronomy)
        / 2,
        0,
        decimal=1,
    )

    # with randomized detector, detector position
    final_image_lenstronomy2 = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=45,
        oversample=3,
        add_noise=True,
        psf_directory=PSF_DIRECTORY,
        galsim_convolve=False,
    )
    assert not np.allclose(final_image_lenstronomy, final_image_lenstronomy2)


def test_simulate_roman_image_with_psf_without_noise():
    with open(
        os.path.join(PSF_DIRECTORY, "F106_SCA01_2000_2000_3.pkl"), "rb"
    ) as psf_file:
        psf = pickle.load(psf_file)

    kwargs_psf = {
        "point_source_supersampling_factor": 3,
        "psf_type": "PIXEL",
        "kernel_point_source": psf[0].data,
        "kernel_point_source_normalisation": False,
    }
    kwargs_numerics = {
        "point_source_supersampling_factor": 3,
        "supersampling_factor": 3,
        "supersampling_convolution": True,
    }
    # Manually convolves psf through lenstronomy, no roman detector effects or background
    array = simulate_image(
        lens_class=LENS,
        band=BAND,
        num_pix=51,
        observatory="Roman",
        kwargs_psf=kwargs_psf,
        kwargs_numerics=kwargs_numerics,
        add_noise=False,
    )
    image_ref = array[3:-3, 3:-3]

    # Convolves psf through galsim, also no roman detector effects or background
    kwargs_single_band = image_quality_lenstronomy.kwargs_single_band(
        observatory="Roman", band=BAND
    )
    exposure_time = (
        kwargs_single_band["exposure_time"] * kwargs_single_band["num_exposures"]
    )
    galsim_image = simulate_roman_image(
        lens_class=LENS,
        band=BAND,
        num_pix=45,
        oversample=3,
        seed=42,
        add_noise=False,
        psf_directory=PSF_DIRECTORY,
        detector=1,
        detector_pos=(2000, 2000),
        exposure_time=exposure_time,
    )

    # Makes sure that each pixel matches in flux by 2%, and the total flux matches by up to 0.1
    # Most pixels should match by 0.005% but there may be a couple ones that only match by 2%
    np.testing.assert_allclose(galsim_image, image_ref, rtol=0.02, atol=0)
    np.testing.assert_allclose(
        np.sum(galsim_image), np.sum(image_ref), rtol=0, atol=0.1
    )


def test_simulate_roman_image_with_time_variable_source():
    detector_kwargs = {
        "detector": 1,
        "detector_pos": (2000, 2000),
        "ra": 24,
        "dec": -24,
    }

    # without noise; checking for time variability
    lens_image = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=71,
        oversample=3,
        add_noise=False,
        t_obs=0,
        with_source=True,
        with_deflector=True,
        psf_directory=PSF_DIRECTORY,
        **detector_kwargs,
    )

    # this is literally the same thing, checks that all randomness is gone
    lens_image2 = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=71,
        oversample=3,
        add_noise=False,
        t_obs=0,
        with_source=True,
        with_deflector=True,
        psf_directory=PSF_DIRECTORY,
        **detector_kwargs,
    )

    npt.assert_allclose(lens_image, lens_image2, atol=1e-16, rtol=1e-16)

    # now change the observation time, check we get a different result
    lens_image3 = simulate_roman_image(
        lens_class=SNIa_Lens,
        band=BAND,
        num_pix=71,
        oversample=3,
        add_noise=False,
        t_obs=99,
        with_source=True,
        with_deflector=True,
        psf_directory=PSF_DIRECTORY,
        **detector_kwargs,
    )

    assert not np.allclose(lens_image, lens_image3, atol=1e-16, rtol=1e-16)


if __name__ == "__main__":
    pytest.main()
