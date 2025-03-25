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

import numpy as np
import astropy.constants as const
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MicrolensingLightCurve(object):
    """Class to generate microlensing lightcurve for a single source based on the magnification map, and lens properties."""

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
                                 return_track_coords=False
                                 ):
        """Generate lightcurves for a quasar(AGN) with the AccretionDisk model
        from amoeba. Returns a list of lightcurves based on the number of lightcurves requested."""
        mag_map_2d = self.magnification_map.magnifications

        # Disk parameters:
        x = np.linspace(-2000, 2000, 2001) #TODO: ask if the x and y pixel ranges should be changed?
        y = np.linspace(-2000, 2000, 2001)
        X, Y = np.meshgrid(x, y)
        radii, phi_array = util.convert_cartesian_to_polar(X, Y)
        # mass_kg = 10**smbh_mass_exp * const.M_sun.to(u.kg)
        mass_sm = 10**smbh_mass_exp
        grav_rad = util.calculate_gravitational_radius(mass_sm)
        temp_map = util.accretion_disk_temperature(
            radii * grav_rad, min_disk_radius * grav_rad, mass_sm, eddington_ratio=eddington_ratio, corona_height=corona_height
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

        disk_projection = Disk.calculate_surface_intensity_map(observer_frame_wavelength_in_nm=observer_frame_wavelength_in_nm)

        MagMap = AmoebaMagnificationMap(
            source_redshift,
            deflector_redshift,
            mag_map_2d,
            self.magnification_map.kappa_tot, #TODO: ask if this is correct for "Convergence of the lensing potential at the location of the image"?
            self.magnification_map.shear,
            mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,#TODO: ask if this can be related to theta_star in MagnificationMap class?
            total_microlens_einstein_radii=self.magnification_map.half_length_x*2, # assuming square map
            OmM=OmM,
            H0=H0,
        )

        convolution = MagMap.convolve_with_flux_projection(disk_projection)
        self.amoeba_convolution = convolution
        # convolved_flux_map = convolution.magnification_array

        LCs = []
        tracks = []

        time_duration_years = self.time_duration / 365.25 # converting time_duration from days to years
        for jj in range(num_lightcurves):
            curve, tracks_x, tracks_y = convolution.pull_light_curve(
                v_transverse, time_duration_years, return_track_coords=True
            )
            LCs.append(curve)
            tracks.append([tracks_x, tracks_y])
        
        if lightcurve_type == 'magnitude':
            mean_flux = np.mean(convolution.magnification_array) #TODO: ask if this is correct?
            LCs = [-2.5 * np.log10(LC/mean_flux) for LC in LCs]
        

        if return_track_coords:
            return LCs, tracks
        else:
            return LCs

    
    def _plot_agn_lightcurve(self, lightcurves, tracks=None, lightcurve_type='magnitude'):
        """Plot the AGN lightcurve."""
        fig, ax = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[2, 1])
        
        time_array = np.linspace(0, self.time_duration, len(lightcurves[0])) # in days

        # light curves
        for i in range(len(lightcurves)):
            ax[0].plot(time_array, lightcurves[i], label=f"Lightcurve {i+1}")
        ax[0].set_xlabel("Time (days)")
        if lightcurve_type == 'magnitude':
            ax[0].set_ylabel("Magnitude")
        elif lightcurve_type == 'flux':
            ax[0].set_ylabel("Flux")

        ax[0].legend()

        # magmap
        conts = ax[1].imshow(self.magnification_map.magnitudes, cmap="viridis_r", extent = [-self.magnification_map.half_length_x, self.magnification_map.half_length_x, 
                                                                                            -self.magnification_map.half_length_y, self.magnification_map.half_length_y])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(conts, cax=cax)
        cbar.set_label("Microlensing $\\Delta m$ (magnitudes)")
        ax[1].set_xlabel("$y_1 / \\theta_★$")
        ax[1].set_ylabel("$y_2 / \\theta_★$")

        # tracks are in pixel coordinates
        # to map them to the magmap coordinates, we need to convert them to the physical coordinates
        delta_x = 2 * self.magnification_map.half_length_x / self.magnification_map.num_pixels_x
        delta_y = 2 * self.magnification_map.half_length_y / self.magnification_map.num_pixels_y
        mid_x_pixel = self.magnification_map.num_pixels_x // 2
        mid_y_pixel = self.magnification_map.num_pixels_y // 2

        if tracks is not None:
            for j in range(len(tracks)):
                ax[1].plot((tracks[j][0] - mid_x_pixel)*delta_x, (tracks[j][1] - mid_y_pixel)*delta_y, "w-", lw=1)
                ax[1].text((tracks[j][0][0]- mid_x_pixel)*delta_x, (tracks[j][1][0] - mid_y_pixel)*delta_y, f"Track {j+1}", color="w")


# class MicrolensingLightCurve(object):
#     """Class to generate microlensing lightcurves based on the lens."""

#     def __init__(
#         self,
#         lens_class: Lens,
#     ):
#         """
#         :param magnification_map: MagnificationMap object, if not provided, it will be generated for each source in the lens_class
#         :param image_observing_time: time at which the image is observed
#         """
#         self.lens_class = lens_class

#     def generate_lightcurve_magnitude(self, time_array, lens_class):
#         """Generate lightcurve based on the source type."""
#         pass

#     def _generate_agn_lightcurve(self, source_class, lens_class):
#         """Generate lightcurve for a quasar(AGN) with the accretion disk model
#         from amoeba."""
#         pass

#     def _generate_point_source_lightcurve(self):
#         """Generate lightcurve for a point source."""
#         pass

#     def _generate_supernova_lightcurve(self):
#         """Generate lightcurve for a supernova."""
#         pass
