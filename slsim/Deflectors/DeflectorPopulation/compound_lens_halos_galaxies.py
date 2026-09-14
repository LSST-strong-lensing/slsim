from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase


class CompoundLensHalosGalaxies(DeflectorsBase):
    """Class describing compound lens model in which the mass distribution of
    individual lens objects is described by a superposition of dark matter and
    stellar components.

    This is effectively the DeflectorsBase() class with specific
    settings for default inputs.
    """

    def __init__(
        self,
        deflector_table,
        kwargs_cut,
        cosmo,
        sky_area,
        mass_type="NFW_HERNQUIST",
        light_type="hernquist",
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
            deflector_table=deflector_table,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=None,
            mass_type=mass_type,
            light_type=light_type,
            kwargs_mass2light=kwargs_mass2light,
            catalog_type=catalog_type,
        )


# TODO: read in 'e_h', 'p_h',  'p_g', 'tb' as SLHammock catalog
