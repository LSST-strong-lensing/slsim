from astropy.table import Table, hstack
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from slsim.Sources import random_supernovae
import numpy as np


class SupernovaeCatalog(object):
    """Class to generate a supernovae catalog."""

    def __init__(
        self,
        sn_type,
        band,
        lightcurve_time,
        absolute_mag_band,
        mag_zpsys,
        cosmo,
        skypy_config,
        sky_area,
        absolute_mag,
    ):
        """

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param band: observation band
        :type band: str. eg: 'i', 'g', 'r', or any other supported band
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
        :param absolute_mag_band: Band used to normalize to absolute magnitude
        :type absolute_mag_band: str or `~sncosmo.Bandpass`
        :param mag_zpsys: Optional, AB or Vega (AB default)
        :type mag_zpsys: str
        :param cosmo: astropy.cosmology instance
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self.sn_type = sn_type
        self.band = band
        self.lightcurve_time = lightcurve_time
        self.absolute_mag = absolute_mag
        self.absolute_mag_band = absolute_mag_band
        self.mag_zpsys = mag_zpsys
        self.cosmo = cosmo
        self.skypy_config = skypy_config
        self.sky_area = sky_area

    def host_galaxy_catalog(self):
        """Generates galaxy catalog and those galaxies can be used as supernovae host
        galaxies.

        :return: supernovae host galaxy catalog
        """
        pipeline = SkyPyPipeline(
            skypy_config=self.skypy_config,
            sky_area=self.sky_area,
            filters=None,
            cosmo=self.cosmo,
        )
        galaxy_table = pipeline.blue_galaxies
        galaxy_table_cut = galaxy_table[galaxy_table["z"] <= 0.9329]
        return galaxy_table_cut

    def supernovae_catalog(self, host_galaxy=True):
        """Generates supernovae lightcurves for each host galaxies.

        :param host_galaxy: kwargs to decide whether catalog should include host
            galaxies or not.
        :return: supernovae catalog
        """
        band_string = "lsst" + self.band
        host_galaxies = self.host_galaxy_catalog()
        time = []
        magnitude = []
        for z in host_galaxies["z"]:
            lightcurve_class = random_supernovae.RandomizedSupernova(
                self.sn_type,
                z,
                self.absolute_mag,
                self.absolute_mag_band,
                self.mag_zpsys,
                self.cosmo,
            )
            time.append(self.lightcurve_time)
            magnitude.append(
                lightcurve_class.get_apparent_magnitude(
                    self.lightcurve_time, band_string, zpsys=self.mag_zpsys
                )
            )
        if host_galaxy is True:
            ra_off = np.random.uniform(-5, 5, len(host_galaxies["z"]))
            dec_off = np.random.uniform(-5, 5, len(host_galaxies["z"]))
            lightcurve_table = Table(
                [time, magnitude, ra_off, dec_off],
                names=("MJD", "ps_mag_" + self.band, "ra_off", "dec_off"),
            )
            supernovae_table = hstack([lightcurve_table, host_galaxies])
        else:
            supernovae_table = Table(
                [host_galaxies["z"], time, magnitude],
                names=("z", "MJD", "ps_mag_" + self.band),
            )
        return supernovae_table
