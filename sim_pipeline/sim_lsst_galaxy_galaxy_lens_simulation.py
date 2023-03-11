import os
import random
# lenstronomy module import
import lenstronomy
import matplotlib.pyplot as plt
import numpy as np
import skypy
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import constants
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from skypy.pipeline import Pipeline
from skypy.galaxies.velocity_dispersion import schechter_vdf
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import make_lupton_rgb
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.ImSim.image_model import ImageModel
cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966)

print('SkyPy version ==', skypy.__version__)
print('lenstronomy version ==', lenstronomy.__version__)

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


class Skypipeline:
    """
    This Class is for execute the pipeline from Skypy and return the current pipeline state
    """
    def __init__(self, skypy_config):
        """
        :param skypy_config: os.path.join("certain file")
        """
        self.skypy_config = skypy_config
        self.pipeline = Pipeline.read(skypy_config)  # create a Pipeline instance using the specified skypy_config file

    def run_pipeline(self):
        """

        Run the pipeline and return the pipeline state.
        """
        self.pipeline.execute()
        return self.pipeline


numpix = 64  # number of pixels per axis of the image to be modelled

kwargs_model = {'lens_model_list': ['SIE', 'SHEAR'],  # list of lens models to be used
                'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                'source_light_model_list': ['SERSIC_ELLIPSE'],  # list of extended source models to be used
                }
# here we define the numerical options used in the ImSim module.
# Have a look at the ImageNumerics class for detailed descriptions.
# If not further specified, the default settings are used.
kwargs_numerics = {'point_source_supersampling_factor': 1, 'supersampling_factor': 3}

g_band = ObservationLSST('g').get_kwargs()
r_band = ObservationLSST('r').get_kwargs()
i_band = ObservationLSST('i').get_kwargs()

sim_g = SimAPI(numpix=numpix, kwargs_single_band=g_band, kwargs_model=kwargs_model)
sim_r = SimAPI(numpix=numpix, kwargs_single_band=r_band, kwargs_model=kwargs_model)
sim_i = SimAPI(numpix=numpix, kwargs_single_band=i_band, kwargs_model=kwargs_model)

imSim_g = sim_g.image_model_class(kwargs_numerics)
imSim_r = sim_r.image_model_class(kwargs_numerics)
imSim_i = sim_i.image_model_class(kwargs_numerics)

path = os.getcwd()
dirpath, _ = os.path.split(path)
module_path, _ = os.path.split(dirpath)
skypy_config = os.path.join(module_path, 'data\SkyPy\lsst-like.yml') #read the file
skycon = Skypipeline(skypy_config).run_pipeline()# access the results
print(skycon.state)

class LenstronomyConfig:
    """
    creat a class for getting galaxy parameters for g, r and i bands, distance, redshift
    """
    def __init__(self, pipeline, mag_cut, z_min, z_max, galaxy_type, position_scatter):
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

    def get_params(self):
        """
        Method to get the galaxy parameters to be used in the simulation
        return the band information g,r,i; (including the magnitude, size, shape, and position under random scatter)
               the distance between the galaxy and observer;
               the redshift of the galaxy;
               the stellar mass; (for velocity dispersion)

        """
        source= self.get_source()
        stellar_mass = source['stellar_mass']
        redz = source['z']
        distan = [self.cosmo.angular_diameter_distance(redz).value]    # distance between galaxy and observer
        size_arcsec = source['angular_size'] / constants.arcsec   # convert radian to arc seconds
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
        return [kwargs_galaxy_g, kwargs_galaxy_r, kwargs_galaxy_i], distan, source['z'], stellar_mass

def get_vd_from_stallermass(smass):
    """
    function for calculate the velocity from the staller mass for velocity dispersion

    :param smass: stellar mass in the unit of solar mass,

    return vdmass: the velocity dispersion ("km/s")
    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and total mass correlations of massive early-type galaxies." The Astrophysical Journal 724.1 (2010): 511.
    """
    stellarmass = smass
    vdmass = (np.power(10,2.32) * np.power(stellarmass/1e11, 0.24))
    return vdmass

configblue = LenstronomyConfig(pipeline=skycon, mag_cut=22, z_min=0.8, z_max=2, galaxy_type='blue', position_scatter=0.5)
configred = LenstronomyConfig(pipeline=skycon, mag_cut=20, z_min=0.1, z_max=0.4, galaxy_type='red', position_scatter=0.0)

def process_image(ax, sim_g, sim_r, sim_i, cosmolo): #
    """
    function for creating image but also Einstein radius calculated

    :param ax: [int,int] position of the image
    :param sim_g:
    :param sim_i:
    :param sim_r:
    :param cosmolo: cosmology model from astropy.cosmology

    plot the lensing image and
    :return:
            v_sigma :velocity dispersion generated by Schechter function
            theta_E :Einstein radius
            redz_source : source redshift
            redz_len : lens redshift
    """

    kwargs_source, distan_source, redz_source, source_stellmass = configblue.get_params()
    kwargs_lens_light, distan_len, redz_len, len_stellmass = configred.get_params()
    v_sigmamass = get_vd_from_stallermass(len_stellmass)
    v_sigma = schechter_vdf(alpha=2.32, beta=2.67, vd_star=161, vd_min=200, vd_max=500) # Define a Schechter function with specific parameters to calculate velocity dispersion v_sigma
    lensCos = LensCosmo(z_lens=redz_len, z_source=redz_source, cosmo=cosmolo) # Define a LensCosmo object with specific redshift values and cosmology
#   theta_E = lensCos.sis_sigma_v2theta_E(v_sigma=v_sigma) # Calculate Einstein radius (theta_E) using lensCos and v_sigma
    theta_E = lensCos.sis_sigma_v2theta_E(v_sigma=v_sigmamass)

    # Define lensing models as dictionaries with specific parameters
    kwargs_lens = [
        {'theta_E': theta_E, 'e1': 0.2, 'e2': -0.1, 'center_x': 0, 'center_y': 0},  # SIE model
        {'gamma1': 0.03, 'gamma2': 0.01, 'ra_0': 0, 'dec_0': 0}  # SHEAR model
    ]
    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = sim_g.magnitude2amplitude(kwargs_lens_light[0], kwargs_source[0])
    kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = sim_r.magnitude2amplitude(kwargs_lens_light[1], kwargs_source[1])
    kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = sim_i.magnitude2amplitude(kwargs_lens_light[2], kwargs_source[2])

    # Generate lensed images for each band (g, r, i)
    image_g = imSim_g.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g, kwargs_ps_g)
    image_r = imSim_r.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, kwargs_ps_r)
    image_i = imSim_i.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)

    # add noise
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)
    image_i += sim_i.noise_for_model(model=image_i)

    # use color with astropy
    image = make_lupton_rgb(image_i, image_r, image_g, stretch=0.5)

    # and plot it
    ax.imshow(image, aspect='equal', origin='lower')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)

    print(v_sigmamass)
    return v_sigma,theta_E,redz_source,redz_len

n_horizont, n_vertical = 5, 3
f, axes = plt.subplots(n_vertical, n_horizont, figsize=(n_horizont * 3, n_vertical * 3))
sourcez_values = [] # First, create a list of z values
lensz_values = []
v_sigma_theta_Evalues = {'list1':[],'list2':[]}

for i in range(n_horizont):  # plt image
    for j in range(n_vertical):
        process_image(ax=axes[j, i],sim_i=sim_i,sim_g=sim_g,sim_r=sim_r,cosmolo=cosmo)
                      # add z to the list
        vsigma, theE, zsource, zlens= process_image(ax=axes[j, i],sim_i=sim_i,sim_g=sim_g,sim_r=sim_r,cosmolo=cosmo)
        sourcez_values.append(zsource)
        lensz_values.append(zlens)
        print(theE)

        # extract the values of v_sigma and theta_E
        v_sigma_theta_Evalues['list1'].append(vsigma)
        v_sigma_theta_Evalues['list2'].append(theE)

f.tight_layout()
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
plt.show()

# Plot the histogram of the z values
plt.hist(sourcez_values, bins=10, histtype='step', density=True, color='blue', alpha=0.5, label='source')
plt.hist(lensz_values, bins=10, histtype='step', density=True, color='red', alpha=0.5, label='lens')
plt.xlabel('z value')
plt.ylabel('Density')
plt.title('Histogram of z values')
plt.show()

#plot the histograms of thetaE and v_sigama

plt.xlabel('v_sigma')
plt.ylabel('theta_E')
plt.title('v_sigma and theta_E')
xplt = np.array(v_sigma_theta_Evalues['list1'])
yplt = np.array(v_sigma_theta_Evalues['list2'])
coefficients = np.polyfit(xplt, yplt, deg=3)
plt.scatter(v_sigma_theta_Evalues['list1'], v_sigma_theta_Evalues['list2'])
plt.plot(xplt, np.polyval(coefficients, xplt), color='yellow')
#plt.legend()
plt.show()