import numpy as np
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourcePopulation.galaxies import convert_catalog_to_source
from slsim.Deflectors.deflector import Deflector
from slsim.Deflectors import deflector_util


class DeflectorsBase(Galaxies):
    """Abstract Base Class to create a class that accesses a set of deflectors.

    All object that inherit from Lensed System must contain the methods
    it contains.
    """

    def __init__(
        self,
        deflector_table,
        kwargs_cut,
        cosmo,
        sky_area,
        gamma_pl=None,
        mass_type="EPL",
        light_type="single_sersic",
        kwargs_mass2light=None,
        catalog_type=None,
    ):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :param mass_type: type of Deflector() mass model class
        :type mass_type: string
        :param light_type: type of Source() model class for the light distribution
        :type light_type: string
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :param kwargs_mass2light: mass-to-light relation
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        """
        super().__init__(
            galaxy_list=deflector_table,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut=kwargs_cut,
            catalog_type=catalog_type,
            size_model=None,
            extended_source_type=light_type,
            downsample_to_dc2=False,
            extended_source_kwargs=None,
        )

        self.mass_type = mass_type
        if kwargs_mass2light is None:
            kwargs_mass2light = {}
        self._kwargs_mass2light = kwargs_mass2light
        self._gamma_pl = gamma_pl
        self._vel_disp_from_stellar_mass = None
        # Will be overwriten by interpolation function deriving velocity dispersion from stellar mass

    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        return self.source_number_selected

    def draw_deflector(self, z_max=None, z_min=None, deflector_index=None):
        """

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param deflector_index: index of galaxy to pic (if provided)
        :return: dictionary of complete parameterization of a deflector
        """

        halo_gal = self.draw_object(
            z_max=z_max, z_min=z_min, galaxy_index=deflector_index
        )

        kwargs_source = convert_catalog_to_source(
            galaxy=halo_gal,
            extended_source_type=self._extended_source_type,
            catalog_type=self._catalog_type,
            size_model=self._size_model,
            cosmo=self._cosmo,
            include_all_keywords=False,
        )
        kwargs_mass = deflector_util.light2mass(
            kwargs_source,
            halo_dict=halo_gal,
            mass_type=self.mass_type,
            **self._kwargs_mass2light,
        )
        kwargs_mass = self._update_mass(
            kwargs_mass=kwargs_mass, kwargs_source=kwargs_source
        )
        z = kwargs_source.pop("z")
        deflector_class = Deflector(
            z=z, kwargs_mass=kwargs_mass, kwargs_light=kwargs_source
        )
        return deflector_class

    def _update_mass(self, kwargs_mass, kwargs_source):
        """Additional updates on mass.

        :param kwargs_mass:
        :param kwargs_source: dictionary matching the Source() input
            class
        :return:
        """
        if self.mass_type in ["EPL"]:
            if self._gamma_pl is not None:
                kwargs_mass["gamma_pl"] = _gamma_pl(self._gamma_pl)
        if (
            "vel_disp" not in kwargs_mass
            and "stellar_mass" in kwargs_source
            and self._vel_disp_from_stellar_mass is not None
        ):
            kwargs_mass["vel_disp"] = self._vel_disp_from_stellar_mass(
                np.log10(kwargs_source["stellar_mass"])
            )
        return kwargs_mass


def _gamma_pl(gamma_pl):
    """Madd density power-law slope (2 = isothermal)

    :param gamma_pl:
    :return:
    """
    if gamma_pl is not None:
        if isinstance(gamma_pl, float):
            return gamma_pl
        elif isinstance(gamma_pl, dict):
            parameters = gamma_pl.keys()
            if "mean" in parameters and "std_dev" in parameters:
                return np.random.normal(loc=gamma_pl["mean"], scale=gamma_pl["std_dev"])
            elif "gamma_min" in parameters and "gamma_max" in parameters:
                return np.random.uniform(
                    low=gamma_pl["gamma_min"],
                    high=gamma_pl["gamma_max"],
                )
            else:
                raise ValueError(
                    "The given quantities in gamma_pl are not recognized."
                    " Please provide the mean and standard deviation for a"
                    " gaussian distribution, or specify the gamma_min and gamma_max "
                    " for a uniform distribution."
                )
        else:
            raise ValueError(
                "The given format of the gamma_pl is not supported."
                " Please provide a float or dictionary. See the documentation"
                " in DeflectorsBase class"
            )
    return 2
