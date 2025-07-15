import numpy as np
import os

from astropy.table import Table, join
from astropy.io import fits

from slsim.Sources.SourceTypes.source_base import SourceBase
from lenstronomy.Util.param_util import ellipticity2phi_q


class COSMOSSource(SourceBase):
    """Class to manage source with single sersic light profile."""

    def __init__(self, source_dict, cosmos_path):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift, a magnitude in any band,
         sersic index, angular size in arcsec, and ellipticities e1 and e2.
         eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10,
         "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        :param cosmos_path: path to the directory called "COSMOS_23.5_training_sample", e.g.
         cosmos_path = "/home/data/COSMOS_23.5_training_sample".
         This directory should contain "real_galaxy_catalog_23.5.fits",
         "real_galaxy_catalog_23.5_fits.fits", and various other fits files.
         This entire directory can be downloaded at https://zenodo.org/records/3242143.
        :type cosmos_path: string

        """
        super().__init__(source_dict=source_dict)
        # Read in the COSMOS catalogs and downselect sources, then store as a class variable
        # If many instances of COSMOSSource are created, this will only execute the first time
        if not hasattr(COSMOSSource, "final_catalog"):
            catalog1_path = os.path.join(cosmos_path, "real_galaxy_catalog_23.5.fits")
            catalog2_path = os.path.join(
                cosmos_path, "real_galaxy_catalog_23.5_fits.fits"
            )
            cat1 = Table.read(catalog1_path, format="fits", hdu=1)
            cat2 = Table.read(catalog2_path, format="fits", hdu=1)
            COSMOSSource.final_catalog = self._process_catalog(cat1, cat2)
            COSMOSSource.cosmos_path = cosmos_path

    def _process_catalog(self, cat1, cat2):
        """This function filters out sources in the catalog so that only
        the nearby, well-resolved galaxies with high SNR remain. Thus, we
        perform the following cuts:
        1. redshift < 1
        2. apparent magnitude < 20
        3. half light radius > 10 pixels

        :param cat1: first COSMOS catalog
        :type cat1: astropy table
        :param cat2: second COSMOS catalog
        :type cat2: astropy table
        :return: merged astropy table with only the well-resolved galaxies
        """

        # These sources are excluded because they are too close to other objects
        source_exclusion_list = [
            79,
            309,
            1512,
            5515,
            7138,
            7546,
            9679,
            14180,
            14914,
            19494,
            22095,
            28335,
            32696,
            32778,
            33527,
            34946,
            36344,
            38328,
            40837,
            41255,
            44363,
            44871,
            49652,
            51303,
            52021,
            55803,
            1368,
            1372,
            1626,
            5859,
            6161,
            6986,
            7312,
            8108,
            8405,
            9349,
            9326,
            9349,
            9745,
            9854,
            9948,
            10146,
            10446,
            11209,
            12397,
            14642,
            14909,
            15473,
            17775,
            17904,
            20256,
            20489,
            21597,
            21693,
            22380,
            23054,
            23390,
            23790,
            24110,
            24966,
            26135,
            27222,
            27781,
            28297,
            29550,
            30089,
            30898,
            30920,
            31548,
            32025,
            33699,
            35553,
            36409,
            36268,
            36576,
            37198,
            37969,
            38873,
            40286,
            40286,
            40924,
            41731,
            44045,
            45066,
            45929,
            45929,
            46575,
            47517,
            48137,
            49441,
            52270,
            52730,
            52759,
            52891,
            54924,
            54445,
            55153,
            10584,
            22051,
            22365,
            23951,
            42334,
            42582,
            51492,
            32135,
            37106,
            37593,
            38328,
            45618,
            47829,
            26145,
        ]

        max_z = 1.0
        faintest_apparent_mag = 20
        min_flux_radius = 10.0

        is_ok = np.ones(len(cat2), dtype=bool)
        is_ok &= cat2["zphot"] < max_z
        is_ok &= cat2["mag_auto"] < faintest_apparent_mag
        is_ok &= cat2["flux_radius"] > min_flux_radius

        # Drop any catalog indices that are in the exclusion list
        is_ok &= np.invert(
            np.isin(np.arange(len(cat2)), source_exclusion_list)
        )

        filtered_catalog = join(cat1[is_ok], cat2[is_ok], keys='IDENT')

        # This is the half light radius that is the geometric mean of the major and minor axis lengths
        # calculated using sqrt(q) * R_half, where R_half is the half-light radius measured along the major axis
        # We then convert this from units of pixels to arcseconds
        q = filtered_catalog["sersicfit"][:, 3]
        R_half = filtered_catalog["sersicfit"][:, 1]
        filtered_catalog["angular_size"] = (
            np.sqrt(q) * R_half * filtered_catalog["PIXEL_SCALE"]
        )

        # drop extraneous data
        keep_columns = [
            "GAL_FILENAME",
            "GAL_HDU",
            "PIXEL_SCALE",
            "sersicfit",
            "angular_size",
        ]

        for col in filtered_catalog.colnames:
            if col not in keep_columns:
                filtered_catalog.remove_column(col)

        return filtered_catalog

    def _match_source(self):
        """This function matches the parameters in source_dict to find a
        corresponding source in the COSMOS catalog. The parameters being
        matched are:

        1. axis ratio q
        2. angular size
        3. n_sersic

        When many matches are found, the match with the best n_sersic is taken.
        The COSMOS image is then rotated to match the desired angle and saved.

        NOTE: To save time when generating a population of lenses, the matching is only done
        when an image is simulated, not when this class is initialized.
        """
        n_sersic = self.source_dict["n_sersic"]
        e1 = self.source_dict["e1"]
        e2 = self.source_dict["e2"]

        # Match with COSMOS catalog based off of axis ratio
        phi, q = ellipticity2phi_q(e1, e2)
        matched_catalog = self.final_catalog[
            np.abs(self.final_catalog["sersicfit"][:, 3].data - q) <= 0.1
        ]

        # Match based off of angular size
        size_ratio = (
            self.source_dict["angular_size"] / matched_catalog["angular_size"].data
        )
        matched_catalog = matched_catalog[size_ratio < 1.5]

        # Match based off of n_sersic
        index = np.argsort(np.abs(matched_catalog["sersicfit"][:, 2].data - n_sersic))[
            0
        ]
        matched_source = matched_catalog[index]

        # load and save image
        fname = matched_source["GAL_FILENAME"]
        hdu = int(matched_source["GAL_HDU"])
        path = os.path.join(self.cosmos_path, fname)
        with fits.open(path) as file:
            self._image = file[hdu].data  # flux per pixel

        # Scale the angular size of the COSMOS image so that it matches the source_dict
        self._scale = (
            matched_source["PIXEL_SCALE"]
            * self.source_dict["angular_size"]
            / matched_source["angular_size"]
        )

        # Rotate the COSMOS image so that it matches the angle given in source_dict
        self._phi = np.pi / 2 - matched_source["sersicfit"][7] - phi

    @property
    def redshift(self):
        """Returns source redshift."""

        return self.source_dict["z"]

    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return self.source_dict["angular_size"]

    @property
    def ellipticity(self):
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return self.source_dict["e1"], self.source_dict["e2"]

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """
        column_names = self.source_dict.colnames
        if "mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "mag_" + band
        source_mag = self.source_dict[band_string]
        return source_mag

    def kwargs_extended_source_light(
        self, reference_position=None, draw_area=None, band=None
    ):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
            reference_position=reference_position, draw_area=draw_area
        )

        if not hasattr(self, "_image"):
            self._match_source()

        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "image": self._image,  # Use the potentially reshaped image
                "center_x": center_source[0],
                "center_y": center_source[1],
                "phi_G": self._phi,
                "scale": self._scale,
            }
        ]
        return kwargs_extended_source

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extended source model.
        """

        source_models_list = ["INTERPOL"]
        return source_models_list
