from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util import catalog_util

CATALOG_TYPES = ["COSMOS"]


class CatalogSource(SourceBase):
    """Class to match sersic parameters to a real source in a given catalog.

    The sources in the catalog must have parameters that have been
    obtained by performing a sersic fit.
    """

    def __init__(self, source_dict, cosmo, catalog_type, catalog_path):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift, a magnitude in any band,
         sersic index, angular size in arcsec, and ellipticities e1 and e2.
         eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10,
         "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: instance of astropy cosmology
        :param catalog_type: specifies which catalog to use. Currently the options are:
         1. "COSMOS" - this catalog can be downloaded from https://zenodo.org/records/3242143
        :type catalog_type: string
        :param catalog_path: path to the directory containing the source catalog. For
         example, if catalog_type = "COSMOS", then catalog_path can be
         catalog_path = "/home/data/COSMOS_23.5_training_sample".
        :type catalog_path: string
        """
        ang_dist = cosmo.angular_diameter_distance(source_dict["z"])
        print(ang_dist.value)
        print(source_dict["angular_size"])
        source_dict["physical_size"] = (
            source_dict["angular_size"] * 4.84814e-6 * ang_dist.value * 1000
        )  # kPc

        super().__init__(source_dict=source_dict)

        # Process catalog and store as class attribute
        # If multiple instances of the class are created, this is only executed once
        if catalog_type == "COSMOS":
            if not hasattr(CatalogSource, "final_cosmos_catalog"):
                CatalogSource.final_cosmos_catalog = (
                    catalog_util.process_cosmos_catalog(
                        cosmo=cosmo, catalog_path=catalog_path
                    )
                )
        else:
            raise ValueError(
                f"Catalog_type {catalog_type} not supported. Currently only {CATALOG_TYPES} are supported."
            )

        self.catalog_type = catalog_type
        self.catalog_path = catalog_path

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
            if self.catalog_type == "COSMOS":
                self._image, self._scale, self._phi = catalog_util.match_cosmos_source(
                    source_dict=self.source_dict,
                    processed_cosmos_catalog=self.final_cosmos_catalog,
                    catalog_path=self.catalog_path,
                )

        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "image": self._image,
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
