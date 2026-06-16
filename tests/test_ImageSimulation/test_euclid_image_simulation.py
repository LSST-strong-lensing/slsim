from astropy.cosmology import FlatLambdaCDM
import numpy as np
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.Lenses.lens import Lens
from slsim.LOS.los_individual import LOSIndividual
from slsim.ImageSimulation.image_simulation import lens_image, simulate_image
from slsim.ImageSimulation import euclid_image_simulation as euclid_rgb
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


def test_euclid_nisp_num_pix_from_vis():
    assert euclid_rgb.euclid_nisp_num_pix_from_vis(61) == 21


def test_euclid_rgb_from_image_list_colour_modes_and_stretches():
    vis, y, j, h = _rgb_h_test_images()

    for stretch in ["linear", "mtf", "arcsinh"]:
        for colour in [
            "VIS",
            "VIS_Y",
            "VIS_J",
            "VIS_Y_J",
            "VIS_WEIGHTED_Y_J",
            "VIS_WEIGHTED_Y_J_H",
        ]:
            image = euclid_rgb.euclid_rgb_from_image_list(
                [vis, y, j, h],
                colour=colour,
                stretch=stretch,
                black_percentile=0,
                white_percentile=100,
            )
            assert image.shape == (5, 5, 3)
            assert np.all(np.isfinite(image))
            assert np.all((image >= 0) & (image <= 1))


def test_euclid_rgb_from_image_list_vis_weighted_colour_mode():
    vis, y, j = _rgb_test_images()

    image = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_WEIGHTED_Y_J",
        stretch="linear",
        black_percentile=0,
        white_percentile=100,
        vis_red_fraction=0.25,
        vis_green_fraction=0.5,
        use_luminance=False,
    )

    vis_prepared = euclid_rgb._prepare_channel(vis, 0, 100)
    y_prepared = euclid_rgb._prepare_channel(
        euclid_rgb._resample_to_shape(y, vis.shape), 0, 100
    )
    j_prepared = euclid_rgb._prepare_channel(
        euclid_rgb._resample_to_shape(j, vis.shape), 0, 100
    )

    expected_red = 0.75 * j_prepared + 0.25 * vis_prepared
    expected_green = 0.5 * y_prepared + 0.5 * vis_prepared

    np.testing.assert_allclose(image[:, :, 0], expected_red)
    np.testing.assert_allclose(image[:, :, 1], expected_green)
    np.testing.assert_allclose(image[:, :, 2], vis_prepared)


def test_euclid_rgb_from_image_list_vis_weighted_y_j_h_colour_mode():
    vis, y, j, h = _rgb_h_test_images()

    image = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j, h],
        colour="VIS_WEIGHTED_Y_J_H",
        stretch="linear",
        black_percentile=0,
        white_percentile=100,
        vis_red_fraction=0.2,
        vis_green_fraction=0.4,
        use_luminance=False,
    )

    vis_prepared = euclid_rgb._prepare_channel(vis, 0, 100)
    y_prepared = euclid_rgb._prepare_channel(
        euclid_rgb._resample_to_shape(y, vis.shape), 0, 100
    )
    j_prepared = euclid_rgb._prepare_channel(
        euclid_rgb._resample_to_shape(j, vis.shape), 0, 100
    )
    h_prepared = euclid_rgb._prepare_channel(
        euclid_rgb._resample_to_shape(h, vis.shape), 0, 100
    )

    expected_red = 0.8 * h_prepared + 0.2 * vis_prepared
    expected_green = 0.6 * (0.5 * (y_prepared + j_prepared)) + 0.4 * vis_prepared

    np.testing.assert_allclose(image[:, :, 0], expected_red)
    np.testing.assert_allclose(image[:, :, 1], expected_green)
    np.testing.assert_allclose(image[:, :, 2], vis_prepared)


def test_euclid_rgb_from_image_list_optional_display_settings():
    vis, y, j = _rgb_test_images()

    no_luminance = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_Y_J",
        stretch="mtf",
        use_luminance=False,
        black_percentile=0,
        white_percentile=100,
    )
    rec709 = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_Y_J",
        stretch="mtf",
        luminance_method="rec709",
        black_percentile=0,
        white_percentile=100,
    )
    auto_mtf = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_Y_J",
        stretch="mtf",
        mtf_midtone="auto",
        mtf_target_mean=0.2,
        mtf_region_size=3,
        black_percentile=0,
        white_percentile=100,
    )
    q1_arcsinh = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_Y_J",
        stretch="arcsinh",
        arcsinh_scale="euclid_q1",
        black_percentile=0,
        white_percentile=100,
    )
    dict_arcsinh = euclid_rgb.euclid_rgb_from_image_list(
        [vis, y, j],
        colour="VIS_Y",
        stretch="arcsinh",
        arcsinh_scale={"VIS": 10, "Y": 2, "median": 3},
        black_percentile=0,
        white_percentile=100,
    )

    for image in [no_luminance, rec709, auto_mtf, q1_arcsinh, dict_arcsinh]:
        assert image.shape == (5, 5, 3)
        assert np.all(np.isfinite(image))


def test_euclid_rgb_from_image_list_errors():
    vis, y, j = _rgb_test_images()

    with pytest.raises(ValueError, match="at least VIS"):
        euclid_rgb.euclid_rgb_from_image_list([])

    with pytest.raises(ValueError, match="requires Y"):
        euclid_rgb.euclid_rgb_from_image_list([vis], colour="VIS_Y")

    with pytest.raises(ValueError, match="requires J"):
        euclid_rgb.euclid_rgb_from_image_list([vis, y], colour="VIS_J")

    with pytest.raises(ValueError, match="requires H"):
        euclid_rgb.euclid_rgb_from_image_list([vis, y, j], colour="VIS_WEIGHTED_Y_J_H")

    with pytest.raises(ValueError, match="colour must be"):
        euclid_rgb.euclid_rgb_from_image_list([vis, y, j], colour="BAD")

    with pytest.raises(ValueError, match="stretch must be"):
        euclid_rgb.euclid_rgb_from_image_list([vis, y, j], stretch="bad")

    with pytest.raises(ValueError, match="luminance_method"):
        euclid_rgb.euclid_rgb_from_image_list(
            [vis, y, j], colour="VIS_Y_J", luminance_method="bad"
        )

    with pytest.raises(ValueError, match="vis_red_fraction"):
        euclid_rgb.euclid_rgb_from_image_list(
            [vis, y, j],
            colour="VIS_WEIGHTED_Y_J",
            vis_red_fraction=-0.1,
        )

    with pytest.raises(ValueError, match="vis_green_fraction"):
        euclid_rgb.euclid_rgb_from_image_list(
            [vis, y, j],
            colour="VIS_WEIGHTED_Y_J",
            vis_green_fraction=1.1,
        )


def test_resampling_crop_pad_and_channel_preparation_helpers():
    small = np.ones((2, 2))
    padded = euclid_rgb._center_crop_or_pad(small, (4, 4))
    assert padded.shape == (4, 4)
    assert np.sum(padded) == 4

    large = np.arange(25).reshape(5, 5)
    cropped = euclid_rgb._center_crop_or_pad(large, (3, 3))
    assert cropped.shape == (3, 3)
    npt_expected = large[1:4, 1:4]
    np.testing.assert_array_equal(cropped, npt_expected)

    resampled = euclid_rgb._resample_to_shape(np.ones((2, 3)), (5, 4))
    assert resampled.shape == (5, 4)

    degenerate = euclid_rgb._prepare_channel(np.ones((3, 3)), 1, 99)
    assert np.all(degenerate == 0)


def test_stretch_and_scale_helpers():
    channel = np.linspace(0, 1, 9).reshape(3, 3)

    assert euclid_rgb._arcsinh_scale_for_band(None, "VIS") == 500.0
    assert euclid_rgb._arcsinh_scale_for_band("euclid_q1", "Y") == 1.0
    assert euclid_rgb._arcsinh_scale_for_band("euclid_q1", "H") == 0.25
    assert euclid_rgb._arcsinh_scale_for_band({"J": 2.0}, "J") == 2.0
    assert euclid_rgb._arcsinh_scale_for_band({"VIS": 2.0}, "Y") == 1.0
    assert euclid_rgb._arcsinh_scale_for_band(4.0, "VIS") == 4.0
    assert euclid_rgb._arcsinh_scale_for_mixed_channel(None, "Y") == 250.5
    assert euclid_rgb._arcsinh_scale_for_mixed_channel({"median": 7.0}, "Y") == 7.0
    assert euclid_rgb._arcsinh_scale_for_mixed_channel({"VIS": 4.0}, "J") == 2.25

    stretched = euclid_rgb._arcsinh_stretch(channel, 4.0)
    assert stretched.shape == channel.shape
    assert np.all((stretched >= 0) & (stretched <= 1))

    mtf = euclid_rgb._midtone_transfer_function(channel, 0.2)
    assert mtf.shape == channel.shape
    assert np.all((mtf >= 0) & (mtf <= 1))

    with pytest.raises(ValueError, match="arcsinh_scale"):
        euclid_rgb._arcsinh_stretch(channel, 0)

    with pytest.raises(ValueError, match="mtf_midtone"):
        euclid_rgb._midtone_transfer_function(channel, 1.5)


def test_auto_mtf_and_region_helpers():
    channel = np.linspace(0, 1, 25).reshape(5, 5)

    region = euclid_rgb._central_region(channel, 3)
    assert region.shape == (3, 3)

    full_region = euclid_rgb._central_region(channel, 10)
    assert full_region.shape == channel.shape

    auto_midtone = euclid_rgb._auto_mtf_midtone(channel, target_mean=0.2, region_size=3)
    assert 0 < auto_midtone < 1

    resolved_auto = euclid_rgb._resolve_mtf_midtone("auto", channel, 0.2, 3)
    assert 0 < resolved_auto < 1
    assert euclid_rgb._resolve_mtf_midtone(0.3, channel, 0.2, 3) == 0.3
    assert euclid_rgb._auto_mtf_midtone(np.zeros((3, 3)), 0.2, 3) == 0.5

    with pytest.raises(ValueError, match="mtf_target_mean"):
        euclid_rgb._auto_mtf_midtone(channel, target_mean=1.2, region_size=3)

    with pytest.raises(ValueError, match="mtf_region_size"):
        euclid_rgb._central_region(channel, 0)


def test_mixed_channel_helper_and_luminance_helper():
    vis, y, _ = _rgb_test_images()
    vis = euclid_rgb._prepare_channel(vis, 0, 100)
    y = euclid_rgb._prepare_channel(y, 0, 100)
    y = euclid_rgb._resample_to_shape(y, vis.shape)

    mixed_mtf = euclid_rgb._mixed_channel(
        vis,
        y,
        stretch="mtf",
        arcsinh_scale=4.0,
        mtf_midtone=0.2,
        mtf_target_mean=0.2,
        mtf_region_size=3,
        band="Y",
    )
    mixed_arcsinh = euclid_rgb._mixed_channel(
        vis,
        y,
        stretch="arcsinh",
        arcsinh_scale={"VIS": 10.0, "Y": 2.0},
        mtf_midtone=0.2,
        mtf_target_mean=0.2,
        mtf_region_size=3,
        band="Y",
    )
    mixed_linear = euclid_rgb._mixed_channel(
        vis,
        y,
        stretch="linear",
        arcsinh_scale=4.0,
        mtf_midtone=0.2,
        mtf_target_mean=0.2,
        mtf_region_size=3,
        band="Y",
    )
    assert mixed_mtf.shape == vis.shape
    assert mixed_arcsinh.shape == vis.shape
    assert mixed_linear.shape == vis.shape

    with pytest.raises(ValueError, match="stretch must be"):
        euclid_rgb._mixed_channel(
            vis,
            y,
            stretch="bad",
            arcsinh_scale=4.0,
            mtf_midtone=0.2,
            mtf_target_mean=0.2,
            mtf_region_size=3,
            band="Y",
        )

    rgb = np.dstack([vis, y, vis])
    lum = vis
    assert euclid_rgb._apply_luminance(rgb, lum, method="mean").shape == rgb.shape
    assert euclid_rgb._apply_luminance(rgb, lum, method="rec709").shape == rgb.shape

    with pytest.raises(ValueError, match="luminance_method"):
        euclid_rgb._apply_luminance(rgb, lum, method="bad")


def test_vis_weighted_channel_helper():
    vis = np.ones((3, 3))
    nisp = np.zeros((3, 3))

    weighted = euclid_rgb._vis_weighted_channel(
        vis_channel=vis,
        colour_channel=nisp,
        vis_fraction=0.4,
        channel_name="red",
    )
    np.testing.assert_allclose(weighted, 0.4 * np.ones((3, 3)))

    with pytest.raises(ValueError, match="vis_red_fraction"):
        euclid_rgb._vis_weighted_channel(
            vis_channel=vis,
            colour_channel=nisp,
            vis_fraction=1.2,
            channel_name="red",
        )


def test_display_colour_balance_helper():
    rgb = np.array(
        [
            [[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]],
            [[1.2, -0.1, 0.5], [0.1, 0.1, 0.1]],
        ]
    )

    balanced = euclid_rgb._apply_display_colour_balance(
        rgb,
        channel_gains=(1.0, 0.5, 0.25),
        saturation=0.5,
    )

    clipped = np.clip(rgb, 0, 1)
    gained = clipped * np.array([1.0, 0.5, 0.25])[None, None, :]
    grey = np.mean(gained, axis=-1, keepdims=True)
    expected = np.clip(grey + 0.5 * (gained - grey), 0, 1)

    np.testing.assert_allclose(balanced, expected)


def test_display_colour_balance_helper_errors():
    rgb = np.ones((2, 2, 3))

    with pytest.raises(ValueError, match="channel_gains"):
        euclid_rgb._apply_display_colour_balance(
            rgb,
            channel_gains=(1.0, 0.5),
        )

    with pytest.raises(ValueError, match="saturation"):
        euclid_rgb._apply_display_colour_balance(
            rgb,
            saturation=-0.1,
        )


def _rgb_test_images():
    vis = np.linspace(0, 2, 25).reshape(5, 5)
    y = np.linspace(0.1, 1.1, 9).reshape(3, 3)
    j = np.linspace(0.2, 1.2, 16).reshape(4, 4)
    return vis, y, j


def _rgb_h_test_images():
    vis, y, j = _rgb_test_images()
    h = np.linspace(0.3, 1.3, 4).reshape(2, 2)
    return vis, y, j, h


if __name__ == "__main__":
    pytest.main()
