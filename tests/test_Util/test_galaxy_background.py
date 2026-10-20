"""Regression checks for simple sky estimation and structural edge
rejection."""

import pytest
import unittest
import tempfile
from unittest import mock
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

import numpy as np
import slsim.Util.galaxy_background as galaxy_background
from slsim.Util.galaxy_background import (
    check_galaxy_edge,
    subtract_estimated_background,
    subtract_catalog_background,
    filter_edge_catalog,
)


class TestGalaxyBackground(unittest.TestCase):
    def test_catalog_background_validation(self):
        image = np.ones((10, 10))
        invalid_metadata = (
            {},
            {"NOISE_MEAN": np.ma.masked},
            {"NOISE_MEAN": "not-a-number"},
            {"NOISE_MEAN": np.nan},
            {"NOISE_MEAN": np.inf},
            {"NOISE_MEAN": [1, 2]},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    subtract_catalog_background(image, metadata)
        with self.assertRaisesRegex(ValueError, "2D"):
            subtract_catalog_background(np.ones(10), {"NOISE_MEAN": 0})

    def test_estimated_background_validation(self):
        image = np.ones((40, 40))
        bad_arguments = (
            ({"image": np.ones((9, 40))}, "at least 10"),
            ({"image": image, "border_fraction": 0}, "border_fraction"),
            ({"image": image, "border_fraction": 0.5}, "border_fraction"),
            ({"image": image, "sigma": np.nan}, "sigma or dilation"),
            ({"image": image, "sigma": 0}, "sigma or dilation"),
            ({"image": image, "dilation": 1.5}, "sigma or dilation"),
            ({"image": image, "dilation": -1}, "sigma or dilation"),
            ({"image": image, "source_mask": np.ones((2, 2))}, "Mask shape"),
            ({"image": image, "coverage_mask": np.ones((2, 2))}, "Mask shape"),
        )
        for kwargs, message in bad_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    subtract_estimated_background(**kwargs)

        mask = np.ones(image.shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "Insufficient sky pixels"):
            subtract_estimated_background(image, coverage_mask=mask)

    def test_insufficient_sky_after_bright_mask(self):
        image = np.ones((40, 40))
        original_sigma_clip = galaxy_background.sigma_clip
        calls = 0

        def controlled_sigma_clip(values, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.ma.array(np.zeros_like(values))
            return original_sigma_clip(values, **kwargs)

        with mock.patch.object(
            galaxy_background, "sigma_clip", side_effect=controlled_sigma_clip
        ):
            with self.assertRaisesRegex(ValueError, "Insufficient unmasked sky"):
                subtract_estimated_background(image, dilation=0)

    def test_catalog_background_preserves_input_and_nonfinite_pixels(self):
        image = np.array([[-2, 2, 5], [np.nan, np.inf, -np.inf]])
        original = image.copy()
        corrected, _, _ = subtract_catalog_background(
            image, {"NOISE_MEAN": 2}
        )
        np.testing.assert_array_equal(corrected, [[0, 0, 3], [np.nan, np.inf, -np.inf]])
        np.testing.assert_array_equal(image, original)
        for sky in (0, -3):
            result, _, _ = subtract_catalog_background(
                original, {"NOISE_MEAN": sky}
            )
            finite = np.isfinite(original)
            np.testing.assert_array_equal(
                result[finite], np.maximum(original[finite] - sky, 0)
            )

    def test_source_mask_floor_and_coverage_preservation(self):
        image = np.full((40, 40), 7.0)
        image[20, 20] = 2
        image[20, 21] = 10
        image[0, 0] = np.nan
        image[0, 1] = np.inf
        image[0, 2] = -np.inf
        coverage = np.zeros(image.shape, bool)
        coverage[0, 3] = True
        image[0, 3] = -9
        source_mask = np.zeros(image.shape, bool)
        source_mask[20, 20:22] = True
        original = image.copy()
        corrected, sky, mask = subtract_estimated_background(
            image, source_mask=source_mask, coverage_mask=coverage
        )
        self.assertEqual(sky, 7)
        self.assertEqual(corrected[20, 20], 0)
        self.assertEqual(corrected[20, 21], 3)
        np.testing.assert_array_equal(corrected[0, :4], original[0, :4])
        np.testing.assert_array_equal(image, original)
        self.assertFalse(mask[0, :4].any())

    def test_clipped_median_and_coverage(self):
        image = np.arange(1600, dtype=float).reshape(40, 40)
        original = image.copy()
        coverage = np.zeros(image.shape, bool)
        coverage[:3] = True
        corrected, background, mask = subtract_estimated_background(
            image, coverage_mask=coverage
        )
        self.assertEqual(background, np.median(image[mask]))
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(corrected[coverage], image[coverage])
        np.testing.assert_allclose(
            corrected[~coverage], np.maximum(image[~coverage] - background, 0)
        )

    def test_outliers_and_negative_residuals(self):
        image = np.random.default_rng(12).normal(7, 0.1, (100, 100))
        image[2:5, 2:5] = 100
        corrected, background, mask = subtract_estimated_background(image)
        self.assertLess(abs(background - 7), 0.02)
        self.assertFalse(mask[2:5, 2:5].any())
        self.assertTrue((corrected == 0).any())
        self.assertTrue((corrected >= 0).all())
        np.testing.assert_allclose(corrected, np.maximum(image - background, 0))

    def test_filter_rejects_source_if_any_band_reaches_edge(self):
        y, x = np.indices((100, 100))
        compact = 50 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 100) + 7
        broad = 50 * np.exp(-((x - 50) ** 2) / 3000 - (y - 50) ** 2 / 40) + 7
        with tempfile.TemporaryDirectory() as directory:
            for ident in (1, 2):
                images = [compact] * 4
                if ident == 1:
                    images[-1] = broad
                fits.HDUList(
                    [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in images]
                ).writeto(Path(directory) / f"COSMOSWeb_galaxy_{ident}_image.fits")
            table = Table({"id": [1, 2]})
            filtered = filter_edge_catalog(table, "COSMOS_WEB", directory)
            self.assertEqual(list(filtered["id"]), [2])
            self.assertEqual(len(table), 2)
            empty = filter_edge_catalog(table[:1], "COSMOS_WEB", directory)
            self.assertEqual(len(empty), 0)

    def test_compact_and_extended(self):
        y, x = np.indices((100, 100))
        compact = 50 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 100)
        broad = 50 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 1500)
        self.assertFalse(check_galaxy_edge(compact + 7, 7)["rejected"])
        self.assertTrue(check_galaxy_edge(broad + 7, 7)["rejected"])
        self.assertTrue(check_galaxy_edge(np.zeros((100, 100)), 0)["rejected"])
        # A disconnected edge object does not become the central galaxy.
        compact[:3, :3] = 100
        self.assertFalse(check_galaxy_edge(compact + 7, 7)["rejected"])

    def test_edge_validation_and_invalid_cutouts(self):
        image = np.ones((20, 20))
        for kwargs in (
            {"margin_fraction": 0},
            {"margin_fraction": 0.5},
            {"peak_fraction": 0},
            {"peak_fraction": 1},
            {"noise_threshold": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "Invalid edge"):
                    check_galaxy_edge(image, 0, **kwargs)
        for invalid in (np.ones(9), np.ones((9, 10)), np.full((10, 10), np.nan)):
            self.assertEqual(check_galaxy_edge(invalid, 0)["reason"], "invalid_cutout")

    def test_filter_catalog_validation_and_hst(self):
        with self.assertRaisesRegex(ValueError, "Unsupported catalog"):
            filter_edge_catalog(Table(), "UNKNOWN", "unused")

        y, x = np.indices((40, 40))
        compact = 30 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 20) + 3
        with tempfile.TemporaryDirectory() as directory:
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(compact)]).writeto(
                Path(directory) / "hst.fits"
            )
            table = Table(
                {
                    "IDENT": [7],
                    "GAL_FILENAME": ["hst.fits"],
                    "GAL_HDU": [1],
                    "NOISE_MEAN": [3.0],
                }
            )
            filtered = filter_edge_catalog(table, "HST_COSMOS", directory)
            self.assertEqual(list(filtered["IDENT"]), [7])

    def test_filter_rejects_invalid_and_skyless_cosmos_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "COSMOSWeb_galaxy_1_image.fits"
            small = np.ones((5, 5))
            fits.HDUList(
                [fits.PrimaryHDU()] + [fits.ImageHDU(small) for _ in range(4)]
            ).writeto(path)
            filtered = filter_edge_catalog(
                Table({"id": [1]}), "COSMOS_WEB", directory
            )
            self.assertEqual(len(filtered), 0)

            path.unlink()
            covered = np.ones((40, 40))
            fits.HDUList(
                [fits.PrimaryHDU()] + [fits.ImageHDU(covered) for _ in range(4)]
            ).writeto(path)
            with mock.patch.object(
                galaxy_background,
                "subtract_estimated_background",
                side_effect=ValueError("no sky"),
            ):
                filtered = filter_edge_catalog(
                    Table({"id": [1]}), "COSMOS_WEB", directory
                )
            self.assertEqual(len(filtered), 0)

    def test_catalog_background_uses_supplied_mean(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        corrected, background, _ = subtract_catalog_background(
            image, {"NOISE_MEAN": 12}
        )
        self.assertEqual(background, 12)
        np.testing.assert_array_equal(corrected, np.maximum(image - 12, 0))


@pytest.mark.parametrize("metadata, options, expected", [
    ({"NOISE_MEAN": 3}, None, 3),
    ({"noise_mean": 4}, None, 4),
    ({"SKY": 5}, {"mean_key": "SKY"}, 5),
    ({"NOISE_MEAN": 3}, {"noise_mean": 6}, 6),
    ({}, {"noise_mean": [1, 2, 3, 4]}, 2),
    ({"SKY": [1, 2, 3, 4]}, {"mean_key": "SKY"}, 2),
])
def test_supplied_mean_routing(metadata, options, expected):
    image = np.full((40, 40), 10.)
    with mock.patch.object(galaxy_background, "subtract_estimated_background") as estimate:
        result = galaxy_background.subtract_source_background(
            image, metadata, options, band_index=1, band_count=4
        )
        estimate.assert_not_called()
    assert result[1] == expected
    assert result[2] is None
    np.testing.assert_array_equal(result[0], image - expected)


@pytest.mark.parametrize("metadata", [None, {}, {"NOISE_VARIANCE": 2}])
def test_missing_mean_uses_estimation(metadata):
    result = galaxy_background.subtract_source_background(np.full((40, 40), 7.), metadata)
    assert result[1] == 7
    assert result[2] is not None


@pytest.mark.parametrize("metadata, options", [
    ({"NOISE_MEAN": np.nan}, None),
    ({"NOISE_MEAN": np.ma.masked}, None),
    ({}, {"mean_key": "missing"}),
    ({}, {"noise_mean": [1, 2]}),
    ({}, {"noise_mean": None}),
    ({}, {"noise_mean": np.inf}),
    ({}, {"typo": 2}),
    ({}, {"mean_key": 3}),
])
def test_invalid_supplied_mean_does_not_estimate(metadata, options):
    with mock.patch.object(galaxy_background, "subtract_estimated_background") as estimate:
        with pytest.raises(ValueError):
            galaxy_background.subtract_source_background(np.ones((40, 40)), metadata, options)
        estimate.assert_not_called()


def test_screening_uses_supplied_per_band_means(tmp_path):
    y, x = np.indices((40, 40))
    galaxy = 30 * np.exp(-((x - 20)**2 + (y - 20)**2) / 20)
    means = [1., 2., 3., 4.]
    fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(galaxy + b) for b in means]).writeto(
        tmp_path / "COSMOSWeb_galaxy_1_image.fits"
    )
    with mock.patch.object(
        galaxy_background, "check_galaxy_edge", wraps=check_galaxy_edge
    ) as check:
        filtered = filter_edge_catalog(
            Table({"id": [1]}), "COSMOS_WEB", tmp_path,
            noise_kwargs={"noise_mean": means},
        )
    assert len(filtered) == 1
    # The first call validates settings; the next four check the native bands.
    assert [call.args[1] for call in check.call_args_list[1:]] == means


@pytest.mark.parametrize("band_index", [0, 1, 2, 3])
def test_catalog_noise_data_selects_native_band_without_mutation(band_index):
    """Read custom columns and keep the catalog values unchanged."""
    means = [1., 2., 3., 4.]
    table = Table({"SKY": [means]})
    result = galaxy_background.catalog_noise_data(
        table[0], {"mean_key": "SKY"}, band_index, 4
    )
    assert result == {"noise_mean": means[band_index]}
    np.testing.assert_array_equal(table["SKY"][0], means)


def test_explicit_mean_overrides_column_selection():
    """A supplied value takes priority even if the named column is absent."""
    assert galaxy_background.catalog_noise_data(
        {}, {"noise_mean": 2, "mean_key": "missing"}
    ) == {"noise_mean": 2}


def test_catalog_background_custom_column_and_integer_input():
    """Subtract a fractional mean without changing the integer input."""
    image = np.array([[0, 1], [2, 3]], dtype=int)
    original = image.copy()
    corrected, background, mask = subtract_catalog_background(
        image, {"SKY": 1.5}, mean_key="SKY"
    )
    assert background == 1.5
    assert mask is None
    assert corrected.dtype.kind == "f"
    np.testing.assert_array_equal(corrected, [[0, 0], [0.5, 1.5]])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("catalog_type", ["HST_COSMOS", "COSMOS_WEB"])
def test_filter_empty_catalog_preserves_columns(catalog_type):
    """An empty catalog returns an empty table with the same columns."""
    table = Table({"id": [], "extra": []})
    filtered = filter_edge_catalog(table, catalog_type, "unused")
    assert isinstance(filtered, Table)
    assert len(filtered) == 0
    assert filtered.colnames == table.colnames


def test_filter_rejects_invalid_mean_instead_of_estimating(tmp_path):
    """Invalid metadata raises rather than silently switching methods."""
    path = tmp_path / "COSMOSWeb_galaxy_1_image.fits"
    fits.HDUList(
        [fits.PrimaryHDU()] + [fits.ImageHDU(np.ones((40, 40))) for _ in range(4)]
    ).writeto(path)
    with mock.patch.object(galaxy_background, "subtract_estimated_background") as estimate:
        with pytest.raises(ValueError, match="finite"):
            filter_edge_catalog(Table({"id": [1], "noise_mean": [np.nan]}),
                                "COSMOS_WEB", tmp_path)
        estimate.assert_not_called()


def test_edge_check_preserves_input():
    """Screening only reads the original image."""
    y, x = np.indices((40, 40))
    image = 3 + 20 * np.exp(-((x - 20)**2 + (y - 20)**2) / 20)
    original = image.copy()
    assert not check_galaxy_edge(image, 3)["rejected"]
    np.testing.assert_array_equal(image, original)
