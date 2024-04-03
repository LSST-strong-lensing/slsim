from slsim.Deflectors.deflectors_base import DeflectorsBase


class GalaxyClusterLenses(DeflectorsBase):
    """Class describing a population of cluster-scale halos + subhaloes."""

    def __init__(self, halo_list, subhalo_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area):
        """

        :param halo_list: list of dictionary with halo parameters
        :param subhalo_list: list of dictionary with subhalo parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        """
        super().__init__(
            deflector_table=halo_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        self._subhalo_list = subhalo_list
        ...
        self._num_select = 0

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """

        pass
