from slsim.Sources.Events.event_lightcone import EventLightcone
from slsim.Sources.Events.event_pop import EventPopulation
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy import units
from scipy.stats import ks_2samp
import numpy.testing as npt
import numpy as np
import pytest


class TestEventLightcone(object):
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.redshifts = np.linspace(0, 5, 20)
        self.sky_area = Quantity(value=1, unit="deg2")
        self.noise = False
        self.time_interval = 1 * units.year

        self.bns_pop = EventPopulation(model="BNS", cosmo=self.cosmo, z_max=5)

        self.bns_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="BNS",
        )

        self.sne_pop = EventPopulation(model="SNIa", cosmo=self.cosmo, z_max=5)

        self.sne_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="SNIa",
        )

    def test_redshift_order(self):
        redshift_inverse = np.linspace(5, 0, 20)
        model = ["BNS", "SNIa"]

        for name in model:
            with pytest.raises(ValueError) as error:
                EventLightcone(
                    cosmo=self.cosmo,
                    redshifts=redshift_inverse,
                    sky_area=self.sky_area,
                    noise=self.noise,
                    time_interval=self.time_interval,
                    model="BNS",
                )

            assert (
                str(error.value)
                == "redshifts must be sorted in strictly increasing order."
            )

    def test_convert_density(self):
        npt.assert_(isinstance(self.time_interval, units.Quantity))

        self.time_interval = 2 * units.day

        self.bns_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="BNS",
        )
        test_density = self.bns_lightcone.convert_density(
            self.bns_pop.event_rate(self.redshifts) / (1 + self.redshifts)
        )[2]
        npt.assert_approx_equal(
            test_density,
            (
                ((self.bns_pop.event_rate(self.redshifts)[2] / 365.25) * 2)
                / (1 + self.redshifts[2])
            ),
            significant=4,
        )

        self.sne_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="SNIa",
        )
        test_density = self.sne_lightcone.convert_density(
            self.sne_pop.event_rate(self.redshifts) / (1 + self.redshifts)
        )[2]
        npt.assert_approx_equal(
            test_density,
            (
                ((self.sne_pop.event_rate(self.redshifts)[2] / 365.25) * 2)
                / (1 + self.redshifts[2])
            ),
            significant=4,
        )

        self.time_interval = 1 * units.year

        self.bns_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="BNS",
        )
        test_density = self.bns_lightcone.convert_density(
            self.bns_pop.event_rate(self.redshifts) / (1 + self.redshifts)
        )[0]
        npt.assert_approx_equal(
            test_density,
            (self.bns_pop.event_rate(self.redshifts)[0] / (1 + self.redshifts[0])),
            significant=4,
        )

        self.sne_lightcone = EventLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
            model="SNIa",
        )
        test_density = self.sne_lightcone.convert_density(
            self.sne_pop.event_rate(self.redshifts) / (1 + self.redshifts)
        )[0]
        npt.assert_approx_equal(
            test_density,
            (self.sne_pop.event_rate(self.redshifts)[0] / (1 + self.redshifts[0])),
            significant=4,
        )

    def test_event_sample(self):
        lightcones = [
            (self.bns_lightcone, self.bns_pop),
            (self.sne_lightcone, self.sne_pop),
        ]

        for lightcone, event_pop in lightcones:
            # Observed event redshift locations using event_sample()
            sample_output = lightcone.event_sample()

            # Observed counts using Lightcone()
            observed_counts, bin_edges = np.histogram(
                sample_output, bins=self.redshifts, density=True
            )

            # Expected counts based on event comoving density
            dN_dz = (
                self.cosmo.differential_comoving_volume(self.redshifts) * self.sky_area
            ).to_value("Mpc3")
            expected_density = lightcone.convert_density(
                event_pop.event_rate(self.redshifts) / (1 + self.redshifts)
            )

            dN_dz *= expected_density
            bin_widths = np.diff(self.redshifts)
            expected_counts = dN_dz[:-1] * bin_widths

            # Normalize expected counts
            expected_counts *= len(sample_output) / np.sum(expected_counts)

            # KS test to compare observed and expected distributions
            observed_cdf = np.cumsum(observed_counts)
            observed_cdf /= observed_cdf[-1]

            expected_cdf = np.cumsum(expected_counts)
            expected_cdf /= expected_cdf[-1]

            _, p_value = ks_2samp(observed_cdf, expected_cdf)

            assert p_value > 0.05, f"KS Test failed: p-value = {p_value}"


if __name__ == "__main__":
    pytest.main()
