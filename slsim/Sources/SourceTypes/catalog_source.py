from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util import catalog_util

CATALOG_TYPES = ["COSMOS"]


class CatalogSource(SourceBase):
    """Class to match sersic parameters to a real source in a given catalog.

    The sources in the catalog must have parameters that have been
    obtained by performing a sersic fit.
    """

    def __init__(
        self,
        angular_size,
        e1,
        e2,
        n_sersic,
        cosmo,
        catalog_type,
        catalog_path,
        **source_dict,
    ):
        """
        :param angular_size: half light radius of object [arcseconds]
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param n_sersic: Sersic index
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift, a magnitude in any band,
         sersic index, angular size in arcsec, and ellipticities e1 and e2.
         eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10,
         "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: instance of astropy cosmology
        :param catalog_type: specifies which catalog to use. Curently the options are:
         1. "COSMOS" - this catalog can be downloaded from https://zenodo.org/records/3242143
        :type catalog_type: string
        :param catalog_path: path to the directory containing the source catalog. For
         example, if catalog_type = "COSMOS", then catalog_path can be
         catalog_path = "/home/data/COSMOS_23.5_training_sample".
        :type catalog_path: string
        """
        super().__init__(extended_source=True, point_source=False, **source_dict)
        self.name = "GAL"
        self._angular_size = angular_size
        self._e1, self._e2 = e1, e2
        self._n_sersic = n_sersic
        self._cosmo = cosmo

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

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position

        if not hasattr(self, "_image"):
            if self.catalog_type == "COSMOS":
                self._image, self._scale, self._phi = catalog_util.match_cosmos_source(
                    angular_size=self.angular_size,
                    physical_size=self.physical_size(cosmo=self._cosmo),
                    e1=self._e1,
                    e2=self._e2,
                    n_sersic=self._n_sersic,
                    processed_cosmos_catalog=self.final_cosmos_catalog,
                    catalog_path=self.catalog_path,
                )
        light_model_list = ["INTERPOL"]
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
        return light_model_list, kwargs_extended_source
