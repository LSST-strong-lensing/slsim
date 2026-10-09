"""Regression checks for simple sky estimation and structural edge
rejection."""

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
    subtract_galaxy_background,
    subtract_hst_catalog_background,
    filter_edge_catalog,
)


class TestGalaxyBackground(unittest.TestCase):
    def test_hst_validation(self):
        image = np.ones((10, 10))
        invalid_metadata = (
            {},
            {"NOISE_MEAN": np.ma.masked},
            {"NOISE_MEAN": "not-a-number"},
            {"NOISE_MEAN": np.nan},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    subtract_hst_catalog_background(image, metadata)
        with self.assertRaisesRegex(ValueError, "2D HST"):
            subtract_hst_catalog_background(np.ones(10), {"NOISE_MEAN": 0})

    def test_background_validation(self):
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
        )
        for kwargs, message in bad_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    subtract_galaxy_background(**kwargs)

        mask = np.ones(image.shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "Insufficient sky pixels"):
            subtract_galaxy_background(image, coverage_mask=mask)

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
                subtract_galaxy_background(image, dilation=0)

    def test_hst_floor_nonfinite_and_nonmutation(self):
        image = np.array([[-2, 2, 5], [np.nan, np.inf, -np.inf]])
        original = image.copy()
        corrected, _, _, stats = subtract_hst_catalog_background(
            image, {"NOISE_MEAN": 2}
        )
        np.testing.assert_array_equal(corrected, [[0, 0, 3], [np.nan, np.inf, -np.inf]])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(stats["clipped_pixels"], 1)
        self.assertEqual(stats["valid_flux_after"], 3)
        for sky in (0, -3):
            result, _, _, _ = subtract_hst_catalog_background(
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
        corrected, sky, mask, stats = subtract_galaxy_background(
            image, source_mask=source_mask, coverage_mask=coverage
        )
        self.assertEqual(sky, 7)
        self.assertEqual(corrected[20, 20], 0)
        self.assertEqual(corrected[20, 21], 3)
        np.testing.assert_array_equal(corrected[0, :4], original[0, :4])
        np.testing.assert_array_equal(image, original)
        self.assertFalse(mask[0, :4].any())
        self.assertEqual(stats["valid_flux_after"], 3)
        self.assertEqual(stats["clipped_pixels"], 1)
        self.assertEqual(stats["edge_median_after"], 0)
        self.assertEqual(stats["edge_rms_about_zero_after"], 0)

    def test_clipped_median_and_coverage(self):
        image = np.arange(1600, dtype=float).reshape(40, 40)
        original = image.copy()
        coverage = np.zeros(image.shape, bool)
        coverage[:3] = True
        corrected, background, mask, _ = subtract_galaxy_background(
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
        corrected, background, mask, _ = subtract_galaxy_background(image)
        self.assertLess(abs(background - 7), 0.02)
        self.assertFalse(mask[2:5, 2:5].any())
        self.assertTrue((corrected == 0).any())
        self.assertTrue((corrected >= 0).all())
        np.testing.assert_allclose(corrected, np.maximum(image - background, 0))

    def test_filter_before_matching(self):
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
            filtered, diagnostics = filter_edge_catalog(table, "COSMOS_WEB", directory)
            self.assertEqual(list(filtered["id"]), [2])
            self.assertEqual(len(table), 2)
            self.assertTrue(diagnostics[0]["rejected"])
            empty, _ = filter_edge_catalog(table[:1], "COSMOS_WEB", directory)
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
            filtered, diagnostics = filter_edge_catalog(table, "HST_COSMOS", directory)
            self.assertEqual(list(filtered["IDENT"]), [7])
            self.assertFalse(diagnostics[0]["rejected"])

    def test_filter_rejects_invalid_and_skyless_cosmos_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "COSMOSWeb_galaxy_1_image.fits"
            small = np.ones((5, 5))
            fits.HDUList(
                [fits.PrimaryHDU()] + [fits.ImageHDU(small) for _ in range(4)]
            ).writeto(path)
            filtered, diagnostics = filter_edge_catalog(
                Table({"id": [1]}), "COSMOS_WEB", directory
            )
            self.assertEqual(len(filtered), 0)
            self.assertTrue(
                all(
                    band["reason"] == "invalid_cutout"
                    for band in diagnostics[0]["bands"]
                )
            )

            path.unlink()
            covered = np.ones((40, 40))
            fits.HDUList(
                [fits.PrimaryHDU()] + [fits.ImageHDU(covered) for _ in range(4)]
            ).writeto(path)
            with mock.patch.object(
                galaxy_background,
                "subtract_galaxy_background",
                side_effect=ValueError("no sky"),
            ):
                _, diagnostics = filter_edge_catalog(
                    Table({"id": [1]}), "COSMOS_WEB", directory
                )
            self.assertTrue(diagnostics[0]["rejected"])
            self.assertTrue(
                all(
                    band["reason"] == "insufficient_sky"
                    for band in diagnostics[0]["bands"]
                )
            )

    def test_hst_uses_metadata(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        corrected, background, _, _ = subtract_hst_catalog_background(
            image, {"NOISE_MEAN": 12}
        )
        self.assertEqual(background, 12)
        np.testing.assert_array_equal(corrected, np.maximum(image - 12, 0))


if __name__ == "__main__":
    unittest.main()
