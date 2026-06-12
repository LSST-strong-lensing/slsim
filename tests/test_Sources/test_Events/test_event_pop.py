from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.Events.event_pop import EventPopulation
from slsim.Sources.Events.BNSMerger.bns_merger_pop import BNSMergerRate
from slsim.Sources.Events.Supernovae.supernovae_pop import SNIaRate
import numpy.testing as npt
import pytest


class TestEventPop:
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_max = 10
        self.z_array = [0, 1, 2, 3]

    def test_BNSMerger_pop(self):
        event_pop = EventPopulation(
            model="BNS",
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

        bnsm_pop = BNSMergerRate(
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

        rate_array = event_pop.event_rate(self.z_array)
        bnsm_array = bnsm_pop.event_rate(self.z_array)

        for i in range(len(self.z_array)):
            npt.assert_almost_equal(bnsm_array[i], rate_array[i], decimal=3)

    def test_SNIa_pop(self):
        event_pop = EventPopulation(
            model="SNIa",
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

        sne_pop = SNIaRate(
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

        rate_array = event_pop.event_rate(self.z_array)
        sne_array = sne_pop.event_rate(self.z_array)

        for i in range(len(self.z_array)):
            npt.assert_almost_equal(sne_array[i], rate_array[i], decimal=3)

    def test_invalid_model(self):
        with pytest.raises(ValueError) as error:
            EventPopulation(
                model="invalid_model",
                cosmo=self.cosmo,
                z_max=self.z_max,
            )

        assert str(error.value) == "model should be chosen from 'BNS' or 'SNIa'"
