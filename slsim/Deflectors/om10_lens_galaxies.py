from slsim.Deflectors.deflector_base import DeflectorBase
import numpy as np


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
        self._mass_eccentricity = self._chosen_deflector["ELLIP"]
        # self._mass_eccentricity = param_util.epsilon2e(self._mass_ellipticity)
        self._mass_phi_e = self._chosen_deflector["PHIE"] * np.pi / 180
        self._chosen_deflector["e1_mass"] = self._mass_eccentricity * np.cos(
            2 * self._mass_phi_e
        )
        self._chosen_deflector["e2_mass"] = self._mass_eccentricity * np.sin(
            2 * self._mass_phi_e
        )

        # self._light_eccentricity = param_util.epsilon2e(self._chosen_deflector["ellipticity_true"]
        # )
        self._light_eccentricity = self._chosen_deflector["ellipticity_true"]
        self._light_phi_e = self._chosen_deflector["PHIE"] * np.pi / 180
        self._chosen_deflector["e1_light"] = self._light_eccentricity * np.cos(
            2 * self._light_phi_e
        )
        self._chosen_deflector["e2_light"] = self._light_eccentricity * np.sin(
            2 * self._light_phi_e
        )
        self._chosen_deflector["n_sersic"] = np.random.normal(loc=4, scale=0.001)
        return self._chosen_deflector
