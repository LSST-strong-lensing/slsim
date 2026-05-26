from slsim.Sources.Events.BNSMerger.BNSMerger_pop import BNSMergerRate
from slsim.Sources.Supernovae.supernovae_pop import SNIaRate

"""References:
  SNIa population: Oguri and Marshall 2010
  BNS merger population: Kuwahara et al. 2025
"""


class EventPopulation(object):
    """Class to select and calculate event population models."""

    def __init__(self, model, cosmo, z_max):
        """
        BNS model calls the function to calculate the rate of BNS merger. (Eq 4 - Kuwahara et al. 2025)
        :return: BNS merger rate R_m(z), in (M_sol)yr^(-1)Gpc^(-3)
        :return type: array-like

        SNIa model calls the function to calculate the rate of SN Ia. (Eq 15 - Oguri and Marshall 2010)
        :return: SN Ia rate n(z) in [(h)yr^(-1)Mpc^(-3)]
        :return type: array-like
        """

        self.model_name = model

        if model == "BNS":
            self._model = BNSMergerRate(cosmo=cosmo, z_max=z_max)

        elif model == "SNIa":
            self._model = SNIaRate(cosmo=cosmo, z_max=z_max)

        else:
            raise ValueError("model should be chosen from 'BNS' or 'SNIa'")

    def calculate_event_rate(self, z):
        return self._model.calculate_event_rate(z)
