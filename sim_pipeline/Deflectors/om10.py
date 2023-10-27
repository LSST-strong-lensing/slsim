from sim_pipeline.Deflectors.deflector_base import DeflectorBase
import pandas as pd
import numpy as np
from sim_pipeline.Util import param_util


class OM10Lens(DeflectorBase):
    def __init__(self, deflector_table, kwargs_cut, cosmo, sky_area, index):
        """
        :param deflector_table: csv file containing lens properties
        :param kwargs_cut: cuts to impose on lens properties
        :param cosmo: cosmology used
        :param sky_area: area of sky used
        :param index: id number of lens and source
        """
        super().__init__(deflector_table, kwargs_cut, cosmo, sky_area)
        self.deflector_table = pd.read_csv(deflector_table)
        self._chosen_deflector = dict(self.deflector_table.iloc[index])

    @property
    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        return len(self.deflector_table)

    def draw_deflector(self):
        """
        :return: dictionary of complete parameterization of deflector
        """

        self._mass_ellipticity = self._chosen_deflector["ELLIP"]
        self._mass_eccentricity = param_util.epsilon2e(self._mass_ellipticity)
        self._mass_phi_e = self._chosen_deflector["PHIE"]
        self._chosen_deflector["e1_mass"] = self._mass_eccentricity * np.cos(
            self._mass_phi_e
        )
        self._chosen_deflector["e2_mass"] = self._mass_eccentricity * np.sin(
            self._mass_phi_e
        )

        self._light_eccentricity = 0.5 * (
            self._mass_eccentricity - np.random.normal(loc=0, scale=0.1)
        )
        self._light_phi_e = self._mass_phi_e - np.random.normal(loc=0, scale=0.1)
        self._chosen_deflector["e1_light"] = self._light_eccentricity * np.cos(
            self._light_phi_e
        )
        self._chosen_deflector["e2_light"] = self._light_eccentricity * np.sin(
            self._light_phi_e
        )
        self._chosen_deflector["n_sersic"] = np.random.normal(loc=4, scale=0.001)
