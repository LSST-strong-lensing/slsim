from astropy.cosmology import FlatLambdaCDM
import numpy as np
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.Lenses.lens import Lens
from slsim.LOS.los_individual import LOSIndividual
from slsim.ImageSimulation.image_simulation import lens_image, simulate_image
from slsim.Util.param_util import gaussian_psf
import pytest

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

DEFLECTOR_DICT = {
    "z": 0.5,
    "angular_size": 0.09604418906529916,  # effective radius of the deflector in arcsec
    "mag_VIS": 19.5,  # VIS-band magnitude of a deflector
    "theta_E": 2,  # Einstein radius of the deflector
    "e1_light": 0.09096489106609575,  # tangential component of the light ellipticity
    "e2_light": 0.1489400739802363,  # cross component of the light ellipticity
    "e1_mass": 0.1082427319496781,  # tangential component of the mass ellipticity
    "e2_mass": 0.10051583213026649,  # cross component of the mass ellipticity
    "gamma_pl": 2.0,  # power law slope in elliptical power law mass model
    "n_sersic": 2.4362388918558664,  # sersic index of a sersic_ellipse profile of a deflector
    "center_x": 0.10039720005025651,  # x-position of the center of the lens
    "center_y": -0.0002092046265491892,  # y-position of the center of the lens
}

LOS_DICT = {
    "gamma": [-0.03648819840013156, -0.06511863424492038],
    "kappa": 0.06020941823541971,
}

SOURCE_DICT = {
    "z": 1.0,
    "angular_size": 0.10887651129362959,  # effective radius of a source in arcsec
    "mag_VIS": 22.1,  # VIS-band magnitude of a source
    "e1": 0.0,  # tangential component of the ellipticity
    "e2": 0.0,  # cross component of the ellipticity
    "n_sersic": 1.5547096361698418,  # sersic index for sersic_ellipse profile
    "center_x": 0.056053505877290584,  # x-position of the center of a source
    "center_y": -0.08071283196326566,
}

psf_kernel = gaussian_psf(fwhm=0.16, delta_pix=0.101, num_pix=41)

transform_matrix = np.array([[0.101, 0], [0, 0.101]])

BAND = "VIS"
source = Source(cosmo=COSMO, extended_source_type="single_sersic", **SOURCE_DICT)
deflector = Deflector(
    deflector_type="EPL_SERSIC",
    **DEFLECTOR_DICT,
)
los_class = LOSIndividual(kappa=0, gamma=[-0.005061965833762263, 0.028825761226555197])

lens_class = Lens(
    source_class=source, deflector_class=deflector, cosmo=COSMO, los_class=los_class
)


# NOTE: Galsim is required which is not supported on Windows
def test_simulate_euclid_image_with_noise():
    euclid_vis_image = simulate_image(
        lens_class=lens_class,
        observatory="Euclid",
        band=BAND,
        num_pix=61,
        add_noise=True,
    )
    assert euclid_vis_image.shape == (61, 61)


def test_lens_image_euclid():
    euclid_vis_lens_image = lens_image(
        lens_class=lens_class,
        band=BAND,
        mag_zero_point=24,  # lsst coadd images have zero point magnitude of 27.
        num_pix=61,
        psf_kernel=psf_kernel,
        transform_pix2angle=transform_matrix,
        exposure_time=565,
        t_obs=None,
        std_gaussian_noise=None,
        with_source=True,
        with_deflector=True,
        gain=3.1,
        single_visit_mag_zero_points={"VIS": 24.0},
    )

    assert np.shape(euclid_vis_lens_image)[0] == 61


if __name__ == "__main__":
    pytest.main()
