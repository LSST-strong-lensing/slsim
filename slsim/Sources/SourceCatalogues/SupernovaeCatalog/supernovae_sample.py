from astropy.table import Table, hstack
from slsim.Sources.Supernovae import random_supernovae
from slsim.Sources.Supernovae.supernovae_lightcone import SNeLightcone
from slsim.Sources.SourceCatalogues.skypy_galaxy_catalog import GalaxyCatalog
from slsim.Sources.Supernovae.supernovae_host_match import SupernovaeHostMatch
import numpy as np
from astropy import units
from scipy import stats
from slsim.Sources.SourcePopulation.galaxies import galaxy_projected_eccentricity
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Util.param_util import elliptical_distortion_product_average


def supernovae_host_galaxy_offset(host_galaxy_catalog):
    """This function generates random supernovae offsets from their host galaxy
    center based on observed data. (Wang et al. 2013)

    :param host_galaxy_catalog: catalog of host galaxies matched with
        supernovae (must have 'angular_size' and 'ellipticity' columns)
    :type host_galaxy_catalog: astropy Table
    :return: offsets x and y [arcsec] selected for each supernovae based
        on observed distribution; e1 and e2 projected eccentricities
        calculated for each host galaxy
    :return type: list; float
    """
    # Select offset ratios based on observed offset distribution (Wang et al. 2013)
    offset_ratios = list(
        # Parameters (s, loc, and scale) obtained from fitting the observed data (Wang et al. 2013)
        # to lognorm distribution with distfit
        stats.lognorm.rvs(
            0.764609, loc=-0.0284546, scale=0.450885, size=len(host_galaxy_catalog)
        )
    )

    offsets = []
    position_angle_galaxy = []
    position_angle_supernovae = []
    original_x_off = []
    original_y_off = []
    e1 = []
    e2 = []
    transformed_x_off = []
    transformed_y_off = []

    for i in range(len(host_galaxy_catalog)):

        # Set a limit on maximum SN Ia offset ratio from host galaxy center
        while offset_ratios[i] > 3:
            offset_ratios[i] = stats.lognorm.rvs(
                0.764609, loc=-0.0284546, scale=0.450885, size=1
            )[0]

        # Calculate offsets [rad]
        offset = offset_ratios[i] * list(host_galaxy_catalog["angular_size"])[i]
        offsets.append(offset)

        galaxy_angle = np.random.uniform(0, np.pi)
        supernova_angle = np.random.uniform(0, 2 * np.pi)
        position_angle_galaxy.append(galaxy_angle)
        position_angle_supernovae.append(supernova_angle)

        # Calculate the x and y coordinates of the offset [arcsec]
        x_off = ((np.cos(supernova_angle * units.rad)) * (offset * units.rad)).to(
            units.arcsec
        )
        y_off = ((np.sin(supernova_angle * units.rad)) * (offset * units.rad)).to(
            units.arcsec
        )
        original_x_off.append(x_off)
        original_y_off.append(y_off)

        # Calculate projected eccentricities
        slsim_e1, slsim_e2 = galaxy_projected_eccentricity(
            host_galaxy_catalog["ellipticity"][i], galaxy_angle * units.rad
        )
        e1.append(slsim_e1)
        e2.append(slsim_e2)

        # Transform the offset coordinates with eccentricities e1, e2 into elliptical coordinate
        # system
        lens_e1, lens_e2 = ellipticity_slsim_to_lenstronomy(slsim_e1, slsim_e2)

        transform_x_off, transform_y_off = elliptical_distortion_product_average(
            x_off.value, y_off.value, lens_e1, lens_e2, 0, 0
        )
        transformed_x_off.append(transform_x_off)
        transformed_y_off.append(transform_y_off)

    return transformed_x_off, transformed_y_off, e1, e2


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
        host_galaxy_candidate=None,
        redshift_max=5,
    ):
        """

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param band_list: observation band. It sould be a list of bands. Eg: ["i"], ["i","r"]
        :type band_list: str. eg: 'i', 'g', 'r', or any other supported band
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array or None
        :param absolute_mag_band: Band used to normalize to absolute magnitude
        :type absolute_mag_band: str or `~sncosmo.Bandpass`
        :param mag_zpsys: Optional, AB or Vega (AB default)
        :type mag_zpsys: str
        :param cosmo: astropy.cosmology instance
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string or None
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
        :param host_galaxy_candidate: Galaxy catalog in an Astropy table. This catalog
         is used to match with the supernova population. If None, the galaxy catalog is
         generated within this class.
        :param redshift_max: Maximum redshift for supernovae sample. Default is 5.
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
        self.host_galaxy_candidate = host_galaxy_candidate
        self.redshift_max = redshift_max

    def supernovae_catalog(self, host_galaxy=True, lightcurve=True):
        """Generates supernovae catalog for given redshifts.

        :param host_galaxy: kwargs to decide whether catalog should
            include host galaxies or not. True or False.
        :param lightcurve: kwargs for the lightcurve, if lightcurve is
            True, it returns extracts lightcurve for each supernovae
            redshift.
        :return: Astropy Table of supernovae catalog containg redshift,
            lightcurves, ra_off, dec_off, and host galaxy properties. If
            host_galaxy is set to False, it returns catalog without host
            galaxy properties. Light curves are generated using
            RandomizedSupernova class. Light curves are saved as an
            array of observation time and array of corresponding
            magnitudes in specified bands in different columns of the
            Table.
        """
        sne_lightcone = SNeLightcone(
            self.cosmo,
            redshifts=np.linspace(0, self.redshift_max, 500),
            sky_area=self.sky_area,
            noise=True,
            time_interval=1 * units.year,
        )
        supernovae_redshift = sne_lightcone.supernovae_sample()

        if host_galaxy is True:
            if self.host_galaxy_candidate is None:
                galaxy_catalog = GalaxyCatalog(
                    cosmo=self.cosmo,
                    skypy_config=self.skypy_config,
                    sky_area=self.sky_area,
                )
                host_galaxy_catalog = galaxy_catalog.galaxy_catalog()
            else:
                host_galaxy_catalog = self.host_galaxy_candidate
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
            x_off, y_off, e1, e2 = supernovae_host_galaxy_offset(matched_table)
            matched_table["x_off"] = x_off
            matched_table["y_off"] = y_off
            matched_table["e1"] = e1
            matched_table["e2"] = e2
            lightcurve_table = Table(lightcurve_data)
            supernovae_table = hstack([lightcurve_table, matched_table])

        # Only saves supernovae redshift and corresponding lightcurves
        else:
            lightcurve_data["z"] = supernovae_redshift
            supernovae_table = Table(lightcurve_data)

        return supernovae_table
