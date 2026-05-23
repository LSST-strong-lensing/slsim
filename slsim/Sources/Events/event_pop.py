from slsim.Sources.Events.BNSMerger.BNSMerger_pop import BNSMergerRate
from slsim.Sources.Supernovae.supernovae_pop import SNIaRate

"""References:
  SNIa population: Oguri and Marshall 2010
  BNS merger population: Kuwahara et al. 2025
"""


class EventPopulation(object):
    """Class to select and calculate event population models."""

    def __init__(self, model, cosmo, z_max):
        self.model_name = model

        if model == "BNS":
            self._model = BNSMergerRate(cosmo=cosmo, z_max=z_max)

        elif model == "SNIa":
            self._model = SNIaRate(cosmo=cosmo, z_max=z_max)

        else:
            raise ValueError("model should be chosen from 'BNS' or 'SNIa'")
    
    def calculate_event_rate(self, z):
        if self.model_name == "BNS":
            return self._model.calculate_event_rate(z)

        elif self.model_name == "SNIa":
            return self._model.calculate_SNIa_rate(z)
