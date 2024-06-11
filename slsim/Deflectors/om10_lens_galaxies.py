from slsim.Deflectors.deflector_base import DeflectorBase


class OM10Lens(DeflectorBase):
    def __init__(self, deflector_table, kwargs_cut, cosmo, sky_area):
        """
        :param deflector_table: dict-like containing lens properties
        :param kwargs_cut: cuts to impose on lens properties
        :param cosmo: cosmology used
        :param sky_area: area of sky used

        """
        super().__init__(deflector_table, kwargs_cut, cosmo, sky_area)
        self.deflector_table = deflector_table

    @property
    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        return len(self.deflector_table)

    def draw_deflector(self, index):
        """
        :param index: id number of lens and source

        :return: dictionary of complete parameterization of deflector
        """
        self._chosen_deflector = dict(self.deflector_table.loc[index])
        return self._chosen_deflector
