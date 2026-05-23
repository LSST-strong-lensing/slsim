from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.Events.event_pop import EventPopulation
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

        rate_array = event_pop.calculate_event_rate(self.z_array)

        npt.assert_almost_equal(rate_array[0], 0.03081, decimal=3)
        npt.assert_almost_equal(rate_array[1], 0.09757, decimal=3)
        npt.assert_almost_equal(rate_array[2], 0.09297, decimal=3)
        npt.assert_almost_equal(rate_array[3], 0.05915, decimal=3)

    def test_SNIa_pop(self):
        event_pop = EventPopulation(
            model="SNIa",
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

        rate_array = event_pop.calculate_event_rate(self.z_array)

        npt.assert_almost_equal(rate_array[0], 0.000041006, decimal=3)
        npt.assert_almost_equal(rate_array[1], 0.0001191, decimal=3)
        npt.assert_almost_equal(rate_array[2], 0.0001349, decimal=3)
        npt.assert_almost_equal(rate_array[3], 0.00008008, decimal=3)

    def test_invalid_model(self):
        with pytest.raises(ValueError) as error:
            EventPopulation(
                model="invalid_model",
                cosmo=self.cosmo,
                z_max=self.z_max,
            )

        assert str(error.value) == "model should be chosen from 'BNS' or 'SNIa'"
