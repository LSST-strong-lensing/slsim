from slsim.Sources.Events.BNSMerger.bns_merger_pop import BNSMergerRate
from slsim.Sources.Events.Supernovae.supernovae_pop import SNIaRate

"""References:
  SNIa population: Oguri and Marshall 2010
  BNS merger population: Kuwahara et al. 2025
"""


class EventPopulation(object):
    """Class to select and calculate event population models."""

    def __init__(self, model, cosmo, z_max):
        """
        :param model: event population model, chosen from "BNS" or "SNIa"

        :param cosmo: cosmology object
        :param z_max: maximum redshift for the event population model
        """

        self.model_name = model

        if model == "BNS":
            self._model = BNSMergerRate(cosmo=cosmo, z_max=z_max)

        elif model == "SNIa":
            self._model = SNIaRate(cosmo=cosmo, z_max=z_max)

        else:
            raise ValueError("model should be chosen from 'BNS' or 'SNIa'")

    def event_rate(self, z):
        """Call function to calculate the event rate for the selected event
        population model in source frame.

        :param z: an array of redshifts (z>=0). No need to be sorted.
        :return: event rate in [yr^(-1) Mpc^(-3)]
        :return type: array-like
        """
        return self._model.event_rate(z)
