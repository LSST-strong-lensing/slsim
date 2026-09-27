import numpy as np
import pytest

from slsim.Util.color_gradient import (
    attach_foreground_deflector_color_gradient,
    component_weights_for_band,
    default_reference_band,
    radial_color_gradient_image,
)
from slsim.ImageSimulation.image_quality_lenstronomy import register_observatory
from astropy.table import Table


class DummyObservatory:
    def __init__(self, band, **kwargs):
        self.band = band


def test_component_weights_for_band_uses_power_law_sed_slopes():
    color_gradient = {
        "component_spectral_slopes": [2.0, -1.0],
        "reference_band": "i",
    }

    w_g = component_weights_for_band((0.4, 0.6), "g", color_gradient)
    w_i = component_weights_for_band((0.4, 0.6), "i", color_gradient)
    w_y = component_weights_for_band((0.4, 0.6), "y", color_gradient)

    assert w_i == pytest.approx((0.4, 0.6))
    assert w_y[0] > w_g[0]
    assert np.sum(w_y) == pytest.approx(1.0)


def test_attach_foreground_deflector_color_gradient_adds_table_columns():
    table = Table({"mag_i": [20.0, 21.0]})
    color_gradient = {
        "component_spectral_slopes": [2.0, -1.0],
        "reference_band": "i",
    }

    result = attach_foreground_deflector_color_gradient(
        table, color_gradient, component_weights=(2, 3)
    )

    assert result is table
    assert table["color_gradient"][0] == color_gradient
    assert table["w0"][0] == pytest.approx(0.4)
    assert table["w1"][0] == pytest.approx(0.6)

    unchanged = Table({"mag_i": [20.0]})
    assert attach_foreground_deflector_color_gradient(unchanged, None) is unchanged

    with pytest.raises(ValueError, match="must be a dictionary"):
        attach_foreground_deflector_color_gradient(table, "bad")
    with pytest.raises(ValueError, match="two values"):
        attach_foreground_deflector_color_gradient(table, color_gradient, (1, 2, 3))
    with pytest.raises(ValueError, match="positive sum"):
        attach_foreground_deflector_color_gradient(table, color_gradient, (0, 0))


def test_component_weights_validation_and_default_reference():
    source_dict = {"mag_g": 22, "mag_i": 22, "mag_y": 22}
    assert default_reference_band(source_dict) == "i"

    assert component_weights_for_band((2, 3), "i") == pytest.approx((0.4, 0.6))
    assert component_weights_for_band((0.4, 0.6), "i", {}) == pytest.approx((0.4, 0.6))

    with pytest.raises(ValueError, match="must be a dictionary"):
        component_weights_for_band((0.4, 0.6), "i", "bad")

    with pytest.raises(ValueError, match="must match the number of components"):
        component_weights_for_band(
            (0.4, 0.6),
            "i",
            {"component_spectral_slopes": [1.0]},
        )

    with pytest.raises(ValueError, match="must be in"):
        component_weights_for_band(
            (0.4, 0.6),
            "i",
            {"component_spectral_slopes": [1.0, 0.0], "min_weight": 0.5},
        )

    register_observatory(
        name="ZeroWavelengthReferenceObs",
        observatory_class=DummyObservatory,
        bands=["ZW1", "ZW2"],
    )
    with pytest.raises(ValueError, match="neither an explicitly configured"):
        component_weights_for_band(
            (0.4, 0.6),
            "i",
            {"component_spectral_slopes": [1.0, 0.0], "reference_band": "ZW1"},
        )

    register_observatory(
        name="NonPositiveReferenceWavelengthObs",
        observatory_class=DummyObservatory,
        bands=["NPW0", "NPW1"],
        speclite_fmt=None,
        effective_wavelengths={"NPW0": 0.0, "NPW1": 1.0},
    )
    with pytest.raises(ValueError, match="reference band wavelength must be positive"):
        component_weights_for_band(
            (0.4, 0.6),
            "NPW1",
            {
                "component_spectral_slopes": [1.0, 0.0],
                "reference_band": "NPW0",
            },
        )


def test_radial_color_gradient_image_preserves_flux_and_reference_band():
    image = np.ones((9, 9))
    color_gradient = {"grad_color": -0.4, "reference_band": "F814W"}

    assert (
        radial_color_gradient_image(
            image=image,
            band=None,
            color_gradient=color_gradient,
            angular_size=0.3,
            pixel_scale=0.03,
        )
        is image
    )
    with pytest.raises(ValueError, match="must be a dictionary"):
        radial_color_gradient_image(
            image=image,
            band="i",
            color_gradient="bad",
            angular_size=0.3,
            pixel_scale=0.03,
        )

    reference = radial_color_gradient_image(
        image=image,
        band="F814W",
        color_gradient=color_gradient,
        angular_size=0.3,
        pixel_scale=0.03,
    )
    np.testing.assert_allclose(reference, image)

    chromatic = radial_color_gradient_image(
        image=image,
        band="i",
        color_gradient=color_gradient,
        angular_size=0.3,
        pixel_scale=0.03,
    )
    assert not np.allclose(chromatic, image)
    assert np.sum(chromatic) == pytest.approx(np.sum(image))

    no_gradient = radial_color_gradient_image(
        image=image,
        band="i",
        color_gradient={"grad_color": 0.0},
        angular_size=0.3,
        pixel_scale=0.03,
    )
    np.testing.assert_allclose(no_gradient, image)
