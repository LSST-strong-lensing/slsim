from astropy.table import Table, hstack
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from slsim.Sources import random_supernovae
import numpy as np


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

    def supernovae_catalog(self, redshift=None, host_galaxy=True):
        """Generates supernovae lightcurves for given redshifts or from host galaxy
        redshift.

        :param host_galaxy: kwargs to decide whether catalog should include host
            galaxies or not. True or False.
        :return: Astropy Table of supernovae catalog containg redshift, lightcurves,
            ra_off, dec_off, and host galaxy properties. If host_galaxy is set to False,
            it returns catalog without host galaxy properties. Light curves are
            generated using RandomizedSupernova class. Light curves are saved as an
            array of observation time and array of corresponding magnitudes in specified
            bands in different columns of the Table.
        """
        if host_galaxy is True:
            host_galaxies = self.host_galaxy_catalog()
        else:
            host_galaxies = None
        time = []
        # Initialize a list attribute for each band in self.band_list
        for band in self.band_list:
            setattr(self, f"magnitude_{band}", [])
        if redshift is None:
            if host_galaxy is True:
                supernovae_redshift = host_galaxies["z"]
            else:
                raise ValueError(
                    "host_galaxy should be True while redshift is None."
                    "Either set host_galaxy to True or provide redshift list"
                )
        else:
            supernovae_redshift = redshift
        # generate lightcurve for each supernovae.
        for z in supernovae_redshift:
            lightcurve_class = random_supernovae.RandomizedSupernova(
                self.sn_type,
                z,
                self.absolute_mag,
                self.absolute_mag_band,
                self.mag_zpsys,
                self.cosmo,
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
        # astropy table of supernovae lightcurves
        lightcurve_table = Table(lightcurve_data)
        ## get ra_off and dec_off if host galaxy is true.
        if host_galaxy is True:
            ra_off, dec_off = self.supernovae_host_galaxy_offset(
                len(host_galaxies["z"])
            )
            lightcurve_table["ra_off"] = ra_off
            lightcurve_table["dec_off"] = dec_off
            supernovae_table = hstack([lightcurve_table, host_galaxies])
        ## only saves supernovae redshift and corresponding lightcurves
        else:
            lightcurve_table["z"] = redshift
            supernovae_table = lightcurve_table
        return supernovae_table

    def supernovae_host_galaxy_offset(self, supernovae_number):
        """This function generates random supernovae offsets from their host galaxy
        center.

        # TODO: use supernovae and host galaxy parameters to compute more realistic
        offset.

        :param supernovae_number: number of supernovae
        :return: random ra_off and dec_off for each supernovae.
        """
        ## limits used here are mostly arbitrary. more realistic supernovae-host galaxy
        # offset is needed.
        ra_off = np.random.uniform(-5, 5, supernovae_number)
        dec_off = np.random.uniform(-5, 5, supernovae_number)
        return ra_off, dec_off
