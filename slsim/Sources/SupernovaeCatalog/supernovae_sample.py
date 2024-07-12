from astropy.table import Table, hstack
from slsim.Sources import random_supernovae
from slsim.Sources.Supernovae.supernovae_lightcone import SNeLightcone
from slsim.Sources.galaxy_catalog import GalaxyCatalog
from slsim.Sources.supernovae_host_match import SupernovaeHostMatch
import numpy as np
from astropy import units


class SupernovaeCatalog(object):
    """Class to generate a supernovae catalog."""

    def __init__(
        self,
        sn_type,
        band_list,
        lightcurve_time,
        absolute_mag_band,
        mag_zpsys,
        cosmo,
        skypy_config,
        sky_area,
        absolute_mag,
        sn_modeldir=None,
    ):
        """

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param band: observation band. It sould be a list of bands. Eg: ["i"], ["i","r"]
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
        :param sn_modeldir: sn_modeldir is the path to the directory containing files
         needed to initialize the sncosmo.model class. For example,
         sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
         be downloaded from https://github.com/LSST-strong-lensing/data_public .
         For more detail, please look at the documentation of RandomizedSupernovae
         class.
        :type sn_modeldir: str
        """
        self.sn_type = sn_type
        self.band_list = band_list
        self.lightcurve_time = lightcurve_time
        self.absolute_mag = absolute_mag
        self.absolute_mag_band = absolute_mag_band
        self.mag_zpsys = mag_zpsys
        self.cosmo = cosmo
        self.skypy_config = skypy_config
        self.sky_area = sky_area
        self.sn_modeldir = sn_modeldir

    def supernovae_catalog(self, host_galaxy=True, lightcurve=True):
        """Generates supernovae catalog for given redshifts.

        :param host_galaxy: kwargs to decide whether catalog should include host
            galaxies or not. True or False.
        :param lightcurve: kwargs for the lightcurve, if lightcurve is True, it returns
            extracts lightcurve for each supernovae redshift.
        :return: Astropy Table of supernovae catalog containg redshift, lightcurves,
            ra_off, dec_off, and host galaxy properties. If host_galaxy is set to False,
            it returns catalog without host galaxy properties. Light curves are
            generated using RandomizedSupernova class. Light curves are saved as an
            array of observation time and array of corresponding magnitudes in specified
            bands in different columns of the Table.
        """
        sne_lightcone = SNeLightcone(
            self.cosmo,
            redshifts=np.linspace(0, 2.379, 50),
            sky_area=self.sky_area,
            noise=True,
            time_interval=1 * units.year,
        )
        supernovae_redshift = sne_lightcone.supernovae_sample()

        if host_galaxy is True:

            galaxy_catalog = GalaxyCatalog(
                cosmo=self.cosmo,
                skypy_config=self.skypy_config,
                sky_area=self.sky_area,
            )
            host_galaxy_catalog = galaxy_catalog.galaxy_catalog()
            matching_catalogs = SupernovaeHostMatch(
                supernovae_catalog=supernovae_redshift,
                galaxy_catalog=host_galaxy_catalog,
            )
            matched_table = matching_catalogs.match()

        time = []
        # Initialize a list attribute for each band in self.band_list
        for band in self.band_list:
            setattr(self, f"magnitude_{band}", [])

        # Generate lightcurve for each supernovae.
        if lightcurve is True:
            for z in supernovae_redshift:
                lightcurve_class = random_supernovae.RandomizedSupernova(
                    self.sn_type,
                    z,
                    self.absolute_mag,
                    self.absolute_mag_band,
                    self.mag_zpsys,
                    self.cosmo,
                    self.sn_modeldir,
                )
                time.append(self.lightcurve_time)
                for band in self.band_list:
                    mag = lightcurve_class.get_apparent_magnitude(
                        self.lightcurve_time, "lsst" + band, zpsys=self.mag_zpsys
                    )
                    getattr(self, f"magnitude_{band}").append(mag)
            lightcurve_data = {"MJD": time}
            for band in self.band_list:
                lightcurve_data["ps_mag_" + band] = getattr(self, f"magnitude_{band}")
        else:
            lightcurve_data = {}

        # Get ra_off and dec_off if host galaxy is true.
        if host_galaxy is True:
            ra_off, dec_off = galaxy_catalog.supernovae_host_galaxy_offset(
                len(supernovae_redshift)
            )
            lightcurve_data["ra_off"] = ra_off
            lightcurve_data["dec_off"] = dec_off
            lightcurve_table = Table(lightcurve_data)
            supernovae_table = hstack([lightcurve_table, matched_table])

        # Only saves supernovae redshift and corresponding lightcurves
        else:
            lightcurve_data["z"] = supernovae_redshift
            supernovae_table = Table(lightcurve_data)

        return supernovae_table
