import os
import random
# lenstronomy module import
import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import constants
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from skypy.galaxies.velocity_dispersion import schechter_vdf
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import make_lupton_rgb
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST

from get_vd_from_stallermass import get_vd_from_stallermass

class ObservationLSST:
    """
     Defining a class for getting LSST observation band details, return the band details for coadd year of 5, Gaussian type
     """
    def __init__(self, band, psf_type='GAUSSIAN', coadd_years=5):
        """

        :param band: g,r,i...
        :param psf_type: string, type of PSF, ('GAUSSIAN' and 'PIXEL' supported, but not in this one)
        :param coadd_years:
        """
        self.band = band
        self.psf_type = psf_type
        self.coadd_years = coadd_years

    def get_kwargs(self):
        """
        this is the function for returning the band information
        """
        lsst = LSST(band=self.band, psf_type=self.psf_type, coadd_years=self.coadd_years)
        return lsst.kwargs_single_band()



class get_galaxyparas_from_pipeline:
    """
    creat a class for getting galaxy parameters for g, r and i bands, distance, redshift from Pipeline in Lenstronomy
    """
    def __init__(self, pipeline, mag_cut, z_min, z_max, galaxy_type, position_scatter,cosmo= FlatLambdaCDM(H0=67.66, Om0=0.30966)):
        """
        Constructor of the class that initializes the object with specified parameters

        :param pipeline: the results access from Skypipeline
        :param mag_cut
        :param z_min: minimum for the redshift in the cut
        :param z_max: maximum for the redshift in the cur
        :param galaxy_type: 'blue' or 'red' (source or lens)
        :param position_scatter:
        """
        self.pipeline = pipeline
        self.mag_cut = mag_cut
        self.z_min = z_min
        self.z_max = z_max
        self.galaxy_type = galaxy_type
        self.position_scatter = position_scatter
        self.cosmo = cosmo

    def get_source(self):
        """
        Define the source cut based on redshift, magnitude and galaxy type
        return the information
        """
        source_cut = (self.pipeline[self.galaxy_type]['z'] > self.z_min) & (
                    self.pipeline[self.galaxy_type]['z'] < self.z_max) & (
                                 self.pipeline[self.galaxy_type]['mag_g'] < self.mag_cut)
        pipeline_cut = self.pipeline[self.galaxy_type][source_cut]
        n = len(pipeline_cut)
        index = random.randint(0, n - 1)
        source = pipeline_cut[index]
        return source


    def get_galaxy_stellarmass(self):
            """
            Method to get the galaxy parameters to be used in the simulation
            return the stellar mass; (for velocity dispersion)

            """
            source = self.get_source()
            stellar_mass = source['stellar_mass']
            return stellar_mass

    def get_galaxy_distance(self):
            """
            Method to get the galaxy parameters to be used in the simulation
            return: the distance between the galaxy and observer

            """
            source = self.get_source()
            redz = source['z']
            distan = [self.cosmo.angular_diameter_distance(redz).value]  # distance between galaxy and observer
            return distan

    def get_galaxy_redshif(self):
            """
            Method to get the galaxy parameters to be used in the simulation
            :return: redz_source : source redshift

            """
            source = self.get_source()
            redz = source['z']  # distance between galaxy and observer
            return redz

    def get_band_kwargs(self):
            """
            Method to get the band kwargs to be used in the simulation
                :return:  [kwargs_galaxy_g, kwargs_galaxy_r, kwargs_galaxy_i]:the band information g,r,i; (including the magnitude, size, shape, and position under random scatter)

            """
            source = self.get_source()
            size_arcsec = source['angular_size'] / constants.arcsec  # convert radian to arc seconds
            mag_g, mag_r, mag_i = source['mag_g'], source['mag_r'], source['mag_i']
            mag_apparent = source['M']
            e = source['ellipticity']
            phi = np.random.uniform(0, np.pi)
            e1 = e * np.cos(phi)
            e2 = e * np.sin(phi)
            if self.galaxy_type == 'blue':
                n_sersic = 1
            else:
                n_sersic = 4
            center_x, center_y = np.random.uniform(-self.position_scatter, self.position_scatter), np.random.uniform(
                -self.position_scatter, self.position_scatter)
            kwargs_galaxy_g = [{'magnitude': mag_g, 'R_sersic': size_arcsec, 'n_sersic': n_sersic,
                                'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y}]
            kwargs_galaxy_r = [{'magnitude': mag_r, 'R_sersic': size_arcsec, 'n_sersic': n_sersic,
                                'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y}]
            kwargs_galaxy_i = [{'magnitude': mag_i, 'R_sersic': size_arcsec, 'n_sersic': n_sersic,
                                'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y}]
            return [kwargs_galaxy_g, kwargs_galaxy_r, kwargs_galaxy_i]


class GGLensingSystem_kwargs:
    """
    This Class is for execute the galaxies-galaxies lensing system information.
            including source galaxies information, lensing galaxies information, lensing-system information like the Einstein radius. As well as the lensing picture information.
    """
    def __init__(self,configblue,configred):
        self.configblue = configblue
        self.configred = configred
        self.cosmo = self.configblue.cosmo
        self.redz_len = self.configred.get_galaxy_redshif()
        self.redz_source = self.configblue.get_galaxy_redshif()

    def get_lensing_system_parameters(self):
        """
        Get the lensing system parameters, including lens model parameters and Einstein radius.

        :return: List of dictionaries containing lens model parameters
        """
        lens_stellarmass = self.configred.get_galaxy_stellarmass()
        v_sigma = schechter_vdf(alpha=2.32, beta=2.67, vd_star=161, vd_min=200,
                                vd_max=500)  # Define a Schechter function with specific parameters to calculate velocity dispersion v_sigma
        v_sigmamass =get_vd_from_stallermass(lens_stellarmass)
        lenscos = LensCosmo(z_lens=self.redz_len, z_source=self.redz_source,
                            cosmo=self.cosmo)  # Define a LensCosmo object with specific redshift values and cosmology
        theta_E = lenscos.sis_sigma_v2theta_E(v_sigma=v_sigmamass)  # Calculate the Einstein radius using velocity dispersion

        kwargs_lens = [
            {'theta_E': theta_E, 'e1': 0.2, 'e2': -0.1, 'center_x': 0, 'center_y': 0},  # SIE model
            {'gamma1': 0.03, 'gamma2': 0.01, 'ra_0': 0, 'dec_0': 0}  # SHEAR model
        ]

        return kwargs_lens

class GGlensing_image:
    """
    A class to simulate gravitational lensing images' g, r, and i bands using the provided configurations for the source (blue) and lens (red) galaxies.

    :param configblue: configuration object for the source (blue) galaxy (pipeline, mag_cut, z_min, z_max, galaxy_type,position_scatter,cosmo)
    :param configred: configuration object for the lens (red) galaxy (pipeline, mag_cut, z_min, z_max, galaxy_type,position_scatter,cosmo)
    :param cosmo: astropy cosmology object, optional (default is FlatLambdaCDM with H0=67.66, Om0=0.30966)
    :param numpix: number of pixels for the simulated image, default is 64
    """

    def __init__(self,configblue,configred,cosmo,numpix=64):
        self.configred = configred
        self.configblue = configblue
        self.numpix = numpix
        self.sim_g = None
        self.sim_r = None
        self.sim_i = None

        if cosmo is not None:
            self.cosmo = cosmo
        elif hasattr(configred, 'cosmo'):
            self.cosmo = configred.cosmo
        elif hasattr(configblue, 'cosmo'):
            self.cosmo = configblue.cosmo
        else:
            self.cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966)

    def get_irg_from_pipeline(self):
        """
        Method to generate gravitational lensing images in i, r, and g bands using the provided source and lens configurations.

        :return: image_i, image_r, image_g: Simulated lensing images in i, r, and g bands, respectively, without noise
        """
        lens_light=self.configred.get_band_kwargs()
        source_light=self.configblue.get_band_kwargs()
        gg_lensing_system_kwargs = GGLensingSystem_kwargs(self.configblue, self.configred)
        kwargs_lens = gg_lensing_system_kwargs.get_lensing_system_parameters()

        g_band = ObservationLSST('g').get_kwargs()
        r_band = ObservationLSST('r').get_kwargs()
        i_band = ObservationLSST('i').get_kwargs()

        kwargs_numerics = {'point_source_supersampling_factor': 1, 'supersampling_factor': 3}

        kwargs_model = {'lens_model_list': ['SIE', 'SHEAR'],  # list of lens models to be used
                        'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                        'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
                        }

        self.sim_g = SimAPI(numpix=self.numpix, kwargs_single_band=g_band, kwargs_model=kwargs_model)
        self.sim_r = SimAPI(numpix=self.numpix, kwargs_single_band=r_band, kwargs_model=kwargs_model)
        self.sim_i = SimAPI(numpix=self.numpix, kwargs_single_band=i_band, kwargs_model=kwargs_model)

        kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = self.sim_g.magnitude2amplitude(lens_light[0],source_light[0])
        kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = self.sim_r.magnitude2amplitude(lens_light[1],source_light[1])
        kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = self.sim_i.magnitude2amplitude(lens_light[2],source_light[2])

        imSim_g = self.sim_g.image_model_class(kwargs_numerics)
        imSim_r = self.sim_r.image_model_class(kwargs_numerics)
        imSim_i = self.sim_i.image_model_class(kwargs_numerics)

        image_g = imSim_g.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g, kwargs_ps_g)
        image_r = imSim_r.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, kwargs_ps_r)
        image_i = imSim_i.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)

        return image_i,image_r,image_g

    def get_irg_noise(self):
        """
        Method to generate gravitational lensing images in i, r, and g bands with added noise using the provided source and lens configurations.

        :return: image_i, image_r, image_g: Simulated lensing images with noise in i, r, and g bands, respectively
        """

        image_i, image_r, image_g = self.get_irg_from_pipeline()
        # add noise
        image_g += self.sim_g.noise_for_model(model=image_g)
        image_r += self.sim_r.noise_for_model(model=image_r)
        image_i += self.sim_i.noise_for_model(model=image_i)

        return image_i,image_r,image_g


class GGlensing_images_plot:
    """
    A class to create and display simulated gravitational lensing images using the provided configurations for the source (blue) and lens (red) galaxies.

    :param configblue: configuration object for the source (blue) galaxy (pipeline, mag_cut, z_min, z_max, galaxy_type,position_scatter,cosmo)
    :param configred: configuration object for the lens (red) galaxy (pipeline, mag_cut, z_min, z_max, galaxy_type,position_scatter,cosmo)
    :param cosmo: astropy cosmology object, optional (default is FlatLambdaCDM with H0=67.66, Om0=0.30966)
    :param numpix: number of pixels for the simulated image, default is 64
    """
    def __init__(self,configblue,configred,cosmo,numpix=64):
        self.configred = configred
        self.configblue = configblue
        self.numpix = numpix
        if cosmo is not None:
            self.cosmo = cosmo
        elif hasattr(configred, 'cosmo'):
            self.cosmo = configred.cosmo
        elif hasattr(configblue, 'cosmo'):
            self.cosmo = configblue.cosmo
        else:
            self.cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966)

        self.ggsystem =  GGlensing_image(configblue,configred,cosmo,numpix)
        self.image = None

    def plot_one_image(self, use_noise=True):
        """
        Method to generate a single simulated lensing image with or without noise.

        :param use_noise: boolean flag, set to True to add noise to the image, default is True
        """
        if use_noise:
            image_i, image_r, image_g = self.ggsystem.get_irg_noise()
        else:
            image_i, image_r, image_g = self.ggsystem.get_irg_from_pipeline()

        self.image = make_lupton_rgb(image_i, image_r, image_g, stretch=0.5)

    def plot_images(self, use_noise=True, n_horizont=1, n_vertical=1):
        """
        Method to generate and display a grid of simulated gravitational lensing images with or without noise.

        :param use_noise: boolean flag, set to True to add noise to the images, default is True
        :param n_horizont: number of images to display horizontally, default is 1
        :param n_vertical: number of images to display vertically, default is 1
        """
        fig, axes = plt.subplots(n_vertical, n_horizont, figsize=(n_horizont * 3, n_vertical * 3))
        if use_noise:
            for i in range(n_horizont):
               for j in range(n_vertical):
                    ax = axes[j, i]
                    self.plot_one_image(use_noise=True)
                    ax.imshow(self.image, aspect='equal', origin='lower')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.autoscale(False)
        else:
            for i in range(n_horizont):
               for j in range(n_vertical):
                    ax = axes[j, i]
                    self.plot_one_image(use_noise=False)
                    ax.imshow(self.image, aspect='equal', origin='lower')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.autoscale(False)

        fig.tight_layout()
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        plt.show()


