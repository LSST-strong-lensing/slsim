"""Regression checks for simple sky estimation and structural edge rejection."""
import unittest
import tempfile
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

import numpy as np

from slsim.Util.galaxy_background import (
    check_galaxy_edge,
    subtract_galaxy_background,
    subtract_hst_catalog_background,
    filter_edge_catalog,
)


class TestGalaxyBackground(unittest.TestCase):
    def test_clipped_median_and_coverage(self):
        image = np.arange(1600, dtype=float).reshape(40, 40)
        original = image.copy()
        coverage = np.zeros(image.shape, bool)
        coverage[:3] = True
        corrected, background, mask, _ = subtract_galaxy_background(
            image, coverage_mask=coverage)
        self.assertEqual(background, np.median(image[mask]))
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(corrected[coverage], image[coverage])
        np.testing.assert_allclose(corrected[~coverage], image[~coverage]-background)

    def test_outliers_and_negative_residuals(self):
        image = np.random.default_rng(12).normal(7, 0.1, (100, 100))
        image[2:5, 2:5] = 100
        corrected, background, mask, _ = subtract_galaxy_background(image)
        self.assertLess(abs(background-7), 0.02)
        self.assertFalse(mask[2:5, 2:5].any())
        self.assertTrue((corrected < 0).any())

    def test_filter_before_matching(self):
        y, x = np.indices((100, 100))
        compact = 50*np.exp(-((x-50)**2+(y-50)**2)/100)+7
        broad = 50*np.exp(-(x-50)**2/3000-(y-50)**2/40)+7
        with tempfile.TemporaryDirectory() as directory:
            for ident in (1, 2):
                images = [compact]*4
                if ident == 1:
                    images[-1] = broad
                fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in images]).writeto(
                    Path(directory)/f"COSMOSWeb_galaxy_{ident}_image.fits")
            table = Table({"id": [1, 2]})
            filtered, diagnostics = filter_edge_catalog(table, "COSMOS_WEB", directory)
            self.assertEqual(list(filtered["id"]), [2])
            self.assertEqual(len(table), 2)
            self.assertTrue(diagnostics[0]["rejected"])
            empty, _ = filter_edge_catalog(table[:1], "COSMOS_WEB", directory)
            self.assertEqual(len(empty), 0)

    def test_compact_and_extended(self):
        y, x = np.indices((100, 100))
        compact = 50*np.exp(-((x-50)**2+(y-50)**2)/100)
        broad = 50*np.exp(-((x-50)**2+(y-50)**2)/1500)
        self.assertFalse(check_galaxy_edge(compact+7, 7)["rejected"])
        self.assertTrue(check_galaxy_edge(broad+7, 7)["rejected"])
        self.assertTrue(check_galaxy_edge(np.zeros((100, 100)), 0)["rejected"])
        # A disconnected edge object does not become the central galaxy.
        compact[:3, :3] = 100
        self.assertFalse(check_galaxy_edge(compact+7, 7)["rejected"])

    def test_hst_uses_metadata(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        corrected, background, _, _ = subtract_hst_catalog_background(
            image, {"NOISE_MEAN": 12})
        self.assertEqual(background, 12)
        np.testing.assert_array_equal(corrected, image-12)


if __name__ == "__main__":
    unittest.main()
