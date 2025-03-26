__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented
# from Microlensing import HENRYS_AMOEBA_PATH
# import sys
# sys.path.append(HENRYS_AMOEBA_PATH)
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.magnification_map import MagnificationMap as AmoebaMagnificationMap
import amoeba.Util.util as util

from slsim.Microlensing.magmap import MagnificationMap
from slsim.Sources.source import Source
from slsim.lens import Lens
# from slsim.Sources.agn import agn_bounds_dict # to set the limits for the AGN Disk parameters

import numpy as np
import astropy.constants as const
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.LensModel.lens_model import LensModel

class MicrolensingLightCurve(object):
    """Class to generate microlensing lightcurve(s) for a single source based on the magnification map, and lens properties."""

    def __init__(
        self,
        magnification_map: MagnificationMap,
        time_duration: float,
    ):
        """
        :param magnification_map: MagnificationMap object, if not provided, it will be generated for each source in the lens_class
        :param time_duration: Time duration for which the lightcurve is needed (in days).
        """
        self.magnification_map = magnification_map
        self.time_duration = time_duration
    
    def _generate_point_source_lightcurve(self):
        """Generate lightcurve for a point source."""
        pass

    def _generate_supernova_lightcurve(self):
        """Generate lightcurve for a supernova."""
        pass

    def _generate_agn_lightcurve(self, 
                                 source_redshift, 
                                 deflector_redshift,
                                 lightcurve_type = 'magnitude', # 'magnitude' or 'flux'
                                 v_transverse=1000, # Transverse velocity in source plane (in km/s) #TODO: figure out this velocity based on the model in Section 3.3 of https://ui.adsabs.harvard.edu/abs/2020MNRAS.495..544N
                                 num_lightcurves=1, # Number of lightcurves to generate
                                 corona_height=10,
                                 smbh_mass_exp=8.0,
                                 inclination_angle=0,
                                 observer_frame_wavelength_in_nm = 600, # Wavelength in nanometers used to determine black body flux. For the surface flux density of the AccretionDisk at desired wavelength.  
                                 eddington_ratio=0.15, # Eddington ratio of the accretion disk
                                 OmM=0.3, H0=70, # Cosmological parameters, must be updated in Child classes based on lens cosmology!
                                 mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg), # Mean mass of the microlenses in kg
                                 min_disk_radius=6, # Minimum radius of the accretion disk in gravitational radii
                                 return_track_coords=False,
                                 return_time_array=False
                                 ):
        """Generate microlensing lightcurves for a quasar(AGN) with the AccretionDisk model
        from amoeba. Returns a list of lightcurves based on the number of lightcurves requested.
        
        :param source_redshift: Redshift of the source
        :param deflector_redshift: Redshift of the deflector
        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'flux'. Default is 'magnitude'.
        :param v_transverse: Transverse velocity in source plane (in km/s). Default is 1000 km/s.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.
        :param corona_height: Height of the corona above the disk in gravitational radii. Default is 10.
        :param smbh_mass_exp: Exponent of the mass of the supermassive black hole in kg. Default is 8.0.
        :param inclination_angle: Inclination angle of the disk in degrees. Default is 0.
        :param observer_frame_wavelength_in_nm: Wavelength in nanometers used to determine black body flux. For the surface flux density of the AccretionDisk at desired wavelength. Default is 600 nm.
        :param eddington_ratio: Eddington ratio of the accretion disk. Default is 0.15.
        :param OmM: Omega Matter, cosmological parameter. Default is 0.3.
        :param H0: Hubble constant in km/s/Mpc. Default is 70.
        :param mean_microlens_mass_in_kg: Mean mass of the microlenses in kg. Default is 1 * const.M_sun.to(u.kg).
        :param min_disk_radius: Minimum radius of the accretion disk in gravitational radii. Default is 6.
        :param return_track_coords: Whether to return the track coordinates of the lightcuve(s) or not. Default is False.
        :param return_time_array: Whether to return the time array used for the lightcurve(s) or not. Default is False.
        """
        mag_map_2d = self.magnification_map.magnifications

        # Disk parameters:
        x = np.linspace(
            -2000, 2000, 2001
        )  # TODO: ask if the x and y pixel ranges should be changed?
        y = np.linspace(-2000, 2000, 2001)
        X, Y = np.meshgrid(x, y)
        radii, phi_array = util.convert_cartesian_to_polar(X, Y)
        # mass_kg = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        mass_sm = 10**smbh_mass_exp
        grav_rad = util.calculate_gravitational_radius(mass_sm)
        temp_map = util.accretion_disk_temperature(
            radii * grav_rad,
            min_disk_radius * grav_rad,
            mass_sm,
            eddington_ratio=eddington_ratio,
            corona_height=corona_height,
        )

        Disk = AccretionDisk(
            smbh_mass_exp=smbh_mass_exp,
            redshift_source=source_redshift,
            inclination_angle=inclination_angle,
            corona_height=corona_height,
            temp_array=temp_map,
            phi_array=phi_array,
            g_array=np.ones(np.shape(temp_map)),
            radii_array=radii,
            height_array=np.zeros(np.shape(radii)),
        )

        disk_projection = Disk.calculate_surface_intensity_map(observer_frame_wavelength_in_nm=observer_frame_wavelength_in_nm) #TODO: ask if this should be connected to the band used for observations?

        MagMap = AmoebaMagnificationMap(
            source_redshift,
            deflector_redshift,
            mag_map_2d,
            self.magnification_map.kappa_tot,  # TODO: ask if this is correct for "Convergence of the lensing potential at the location of the image"?
            self.magnification_map.shear,
            mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,  # TODO: ask if this can be related to theta_star in MagnificationMap class?
            total_microlens_einstein_radii=self.magnification_map.half_length_x
            * 2,  # assuming square map
            OmM=OmM,
            H0=H0,
        )

        convolution = MagMap.convolve_with_flux_projection(disk_projection)
        self.amoeba_convolution = convolution
        # convolved_flux_map = convolution.magnification_array

        LCs = []
        tracks = []
        time_arrays = []

        time_duration_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years
        for jj in range(num_lightcurves):
            curve, tracks_x, tracks_y = convolution.pull_light_curve(
                v_transverse, time_duration_years, return_track_coords=True
            )
            LCs.append(curve)
            tracks.append([tracks_x, tracks_y])
            time_arrays.append(np.linspace(0, self.time_duration, len(curve)))
        
        if lightcurve_type == 'magnitude':
            mean_flux = np.mean(convolution.magnification_array) #TODO: ask if this is correct?
            LCs = [-2.5 * np.log10(LC/mean_flux) for LC in LCs]
        

        if return_track_coords and not(return_time_array):
            return LCs, tracks
        
        if return_time_array and not(return_track_coords):
            return LCs, time_arrays
        
        if return_track_coords and return_time_array:
            return LCs, tracks, time_arrays
        
        if not(return_track_coords) and not(return_time_array):
            return LCs

    def _plot_agn_lightcurve(
        self, lightcurves, tracks=None, lightcurve_type="magnitude"
    ):
        """Plot the AGN lightcurve."""
        fig, ax = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[2, 1])

        time_array = np.linspace(0, self.time_duration, len(lightcurves[0]))  # in days

        # light curves
        for i in range(len(lightcurves)):
            ax[0].plot(time_array, lightcurves[i], label=f"Lightcurve {i+1}")
        ax[0].set_xlabel("Time (days)")
        if lightcurve_type == "magnitude":
            ax[0].set_ylabel("Magnitude")
        elif lightcurve_type == "flux":
            ax[0].set_ylabel("Flux")

        ax[0].legend()

        # magmap
        conts = ax[1].imshow(
            self.magnification_map.magnitudes,
            cmap="viridis_r",
            extent=[
                -self.magnification_map.half_length_x,
                self.magnification_map.half_length_x,
                -self.magnification_map.half_length_y,
                self.magnification_map.half_length_y,
            ],
        )
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(conts, cax=cax)
        cbar.set_label("Microlensing $\\Delta m$ (magnitudes)")
        ax[1].set_xlabel("$x / \\theta_★$")
        ax[1].set_ylabel("$y / \\theta_★$")

        # tracks are in pixel coordinates
        # to map them to the magmap coordinates, we need to convert them to the physical coordinates
        delta_x = (
            2
            * self.magnification_map.half_length_x
            / self.magnification_map.num_pixels_x
        )
        delta_y = (
            2
            * self.magnification_map.half_length_y
            / self.magnification_map.num_pixels_y
        )
        mid_x_pixel = self.magnification_map.num_pixels_x // 2
        mid_y_pixel = self.magnification_map.num_pixels_y // 2

        if tracks is not None:
            for j in range(len(tracks)):
                ax[1].plot(
                    (tracks[j][0] - mid_x_pixel) * delta_x,
                    (tracks[j][1] - mid_y_pixel) * delta_y,
                    "w-",
                    lw=1,
                )
                ax[1].text(
                    (tracks[j][0][0] - mid_x_pixel) * delta_x,
                    (tracks[j][1][0] - mid_y_pixel) * delta_y,
                    f"Track {j+1}",
                    color="w",
                )


class MicrolensingLightCurveFromLensModel(object):
    """Class to generate microlensing lightcurves based on the lens class provided."""

    def __init__(
        self,
        lens_class: Lens,
        # time_duration: float, # in days
    ):
        """
        :param lens_class: SLSim Lens object, which contains the lens model and source objects.
        """
        self.lens_class = lens_class
        self.sources = lens_class.source # list of sources in the lens class
    
    def generate_point_source_microlensing_magnitudes(self, band, time, source_class: Source, kwargs_MagnificationMap = {}, 
                                                      kwargs_AccretionDisk = {}):
        """Generate lightcurve magnitudes normalized to the mean magnification for agn point sources.
            For single source only, it produces the lightcurve magnitudes for all images of the source.

        Returns a numpy array of microlensing magnitudes with the shape (num_images, len(time)).
        """
        lightcurves, __tracks, __time_arrays = self._generate_agn_lightcurve(
            band, source_class, time, kwargs_MagnificationMap, kwargs_AccretionDisk
        )
        # Here we choose just 1 lightcurve for the point source
        lightcurves_single = np.zeros((len(lightcurves), len(time)))
        for i in range(len(lightcurves)):
            lightcurves_single[i] = lightcurves[i][0]
        
        return lightcurves_single

    def _generate_agn_lightcurve(self, band, source_class: Source,
                                 time,
                                 kwargs_MagnificationMap = {}, 
                                 kwargs_AccretionDisk = {}): #TODO: add the actual kwargs in the definition of the function
        """Generate lightcurves for one single quasar(AGN) source, 
        but for all images of that source based on the lens model.

        The generated lightcurves will have the same length of time as the lightcurve_time provided in the source_class.
        
        :param band: Band for which the lightcurve is needed.
        :param source_class: Source object for which the lightcurve is needed.
        :param time: Time array for which the lightcurve is needed.
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class.
        :param kwargs_AccretionDisk: Keyword arguments for the AccretionDisk class.

        Returns a tuple of:
        lightcurves: a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks: a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays: corresponding to each lightcurve

        """
        lightcurve_time = source_class.lightcurve_time # array of times (days) at which the lightcurve is needed
        lightcurve_duration = np.max(lightcurve_time) - np.min(lightcurve_time) # in days
        
        #TODO: is there an easier way to access convergence and shear from the lens model?
        lenstronomy_kwargs = self.lens_class.lenstronomy_kwargs(band=band)
        lens_model_lenstronomy = LensModel(lens_model_list = lenstronomy_kwargs[0]['lens_model_list'])
        lenstronomy_kwargs_lens = lenstronomy_kwargs[1]['kwargs_lens']

        # TODO: these should go into a child function of this current function
        if kwargs_MagnificationMap == {}:
            kwargs_MagnificationMap = {"rectangular":True, "half_length_x":25, "half_length_y":25, 
                                       "mass_function":'kroupa', "m_lower":0.08, "m_upper":100,
                                       "num_pixels_x":5000, "num_pixels_y":5000,}
        
        if kwargs_AccretionDisk == {}:
            kwargs_AccretionDisk = {"smbh_mass_exp":8.0, "corona_height":10, "inclination_angle":0,
                                    "observer_frame_wavelength_in_nm":600, "eddington_ratio":0.15, "OmM":0.3, "H0":70,
                                    "mean_microlens_mass_in_kg":1 * const.M_sun.to(u.kg), "min_disk_radius":6}

        # kappa, shear and kappa_star for each image of the source
        image_positions_x, image_positions_y = self.lens_class._point_source_image_positions(source_class)
        kappa_star_images = [] # kappa_star for each image of this source
        kappa_tot_images = []
        shear_images = []
        magmaps_images = []
        for i in range(len(image_positions_x)):
            ra = image_positions_x[i] # TODO: is this correct?
            dec = image_positions_y[i]

            kappa_smooth = lens_model_lenstronomy.kappa(ra, dec, lenstronomy_kwargs_lens) # TODO: This is smooth convergence right?
            shear_smooth_vec = lens_model_lenstronomy.gamma(ra, dec, lenstronomy_kwargs_lens)
            shear_smooth = np.sqrt(shear_smooth_vec[0]**2 + shear_smooth_vec[1]**2)

            kappa_star_in_lensing_convergence_units = self.lens_class.kappa_star(ra, dec) # TODO: in the kappa_star function definition it's mentioned that the output is in units of lensing convergence.
            kappa_star = kappa_star_in_lensing_convergence_units * kappa_smooth # based on above comment, Is this line correct?

            kappa_star_images.append(kappa_star[0])
            kappa_tot_images.append(kappa_smooth + kappa_star)
            shear_images.append(shear_smooth)

            # generate magnification maps for each image of the source
            magmap = MagnificationMap(
                kappa_tot = kappa_smooth + kappa_star,
                shear = shear_smooth,
                kappa_star = kappa_star,
                **kwargs_MagnificationMap #TODO: refactor the size of the magmap later!
            )
            magmaps_images.append(magmap)
        
        # save the variables for later use
        self.kappa_star_images = kappa_star_images
        self.kappa_tot_images = kappa_tot_images
        self.shear_images = shear_images
        self.magmaps_images = magmaps_images

        # generate lightcurves for each image of the source
        lightcurves = [] # a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks = [] # a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays = [] # corresponding to each lightcurve
        for i in range(len(image_positions_x)):
            ml_lc = MicrolensingLightCurve(
                magnification_map = magmaps_images[i],
                time_duration = lightcurve_duration
            )
            curr_lightcurves, curr_tracks, curr_time_arrays = ml_lc._generate_agn_lightcurve(
                source_redshift = source_class.redshift,
                deflector_redshift = self.lens_class.deflector_redshift,
                num_lightcurves=1, #TODO: make a decision on how many lightcurves to generate!
                return_track_coords = True,
                return_time_array = True,
                **kwargs_AccretionDisk #TODO: this might need updating depending on how we decide the parameters for the AccretionDisk
            )

            # interpolate the lightcurves to the time array provided
            curr_lightcurves_interpolated = []
            updated_curr_time_arrays = []
            for j in range(len(curr_lightcurves)):
                curr_lightcurves_interpolated.append(
                    self._interpolate_light_curve(
                        curr_lightcurves[j], curr_time_arrays[j], time
                    )
                )
                updated_curr_time_arrays.append(time)

            lightcurves.append(
                curr_lightcurves_interpolated
            )
            tracks.append(
                curr_tracks
            )
            time_arrays.append(
                updated_curr_time_arrays
            )
        
        # light curves is a list with first len being number of images and second len being number of lightcurves for each image
        # tracks is a list with first len being number of images and second len being number of tracks for each image
        # time_arrays is a list with first len being number of images and second len being number of lightcurves for each image

        return lightcurves, tracks, time_arrays

    def _interpolate_light_curve(self, lightcurve, time_array, time_array_new):
        """Interpolate the lightcurve to a new time array.
        Assuming "lightcurve" and "time_array" are 1D arrays of the same length.
        "time_array_new" is a 1D array of the new time array.
        """
        return np.interp(time_array_new, time_array, lightcurve)
        