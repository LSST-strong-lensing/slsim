from sim_pipeline.Deflectors.deflector_base import DeflectorBase


class OM10Lens(DeflectorBase):
    def __init__(self, deflector_table, kwargs_cut, cosmo, sky_area):
        """
        :param deflector_table: csv file containing lens properties
        :param kwargs_cut: cuts to impose on lens properties
        :param cosmo: cosmology used
        :param sky_area:
        """
        super().__init__(deflector_table, kwargs_cut, cosmo, sky_area)

    def deflector_number(self):
        return super().deflector_number()

    def draw_deflector(self):
        return super().draw_deflector()
