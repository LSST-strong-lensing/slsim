__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented

import gc  # for garbage collection
import numpy as np
from scipy.signal import fftconvolve
import astropy.constants as const
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import rescale

from slsim.Microlensing.magmap import MagnificationMap

from slsim.Util.astro_util import (
    calculate_accretion_disk_emission,
    calculate_gravitational_radius,
    extract_light_curve,
)

class MicrolensingLightCurve(object):
    """Class to generate microlensing lightcurve(s) for a single source based
    on the magnification map, and lens properties."""

    def __init__(
        self,
        magnification_map: MagnificationMap,
        time_duration: float,
    ):
        """
        :param magnification_map: MagnificationMap object, if not provided.
        :param time_duration: Time duration for which the lightcurve is needed (in days).
        """
        self.magnification_map = magnification_map
        self.time_duration = time_duration

    def _get_convolved_map(self, source_size, return_source_kernel=False):
        """Compute the convolved magnification map."""
        # we compute the convolved map here
        # convolve the magnification map with a Gaussian kernel
        mag_map_2d = self.magnification_map.magnifications
        # optimize the magnification map for the convolution
        if mag_map_2d.dtype != np.float32:
            mag_map_2d = mag_map_2d.astype(np.float32)
        # make source size map
        xs = np.linspace(
            self.magnification_map.center_x - self.magnification_map.half_length_x,
            self.magnification_map.center_y + self.magnification_map.half_length_x,
            self.magnification_map.num_pixels_x,
        )
        ys = np.linspace(
            self.magnification_map.center_x - self.magnification_map.half_length_y,
            self.magnification_map.center_y + self.magnification_map.half_length_y,
            self.magnification_map.num_pixels_y,
        )
        X, Y = np.meshgrid(xs, ys)
        # Calculate 2D Gaussian map using float32
        sigma = source_size
        # use normal distribution for the source kernel
        source_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        source_kernel /= np.sum(source_kernel)  # normalize the kernel

        ## MEMORY MANAGEMENT ##
        # Optionally delete intermediate arrays if X, Y are very large and not needed again
        del X, Y
        gc.collect()  # Suggest garbage collection
        convolved_map = fftconvolve(mag_map_2d, source_kernel, mode="same")
        self.convolved_map = convolved_map  # Store the convolved map
        # Delete input arrays if they are no longer needed in this function scope
        del mag_map_2d
        gc.collect()
        ########################

        if return_source_kernel:
            return convolved_map, source_kernel
        else:
            del source_kernel

        return convolved_map

    def generate_point_source_lightcurve(
        self,
        # deflector_redshift,
        source_redshift,
        cosmology,
        source_size,
        # mean_microlens_mass_in_kg=1 * const.M_sun.to(u.kg),  # Mean mass of the microlenses in kg
        effective_transverse_velocity=1000,  # Transverse velocity in source plane (in km/s)
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        num_lightcurves=1,  # Number of lightcurves to generate
        return_track_coords=False,
        return_time_array=False,
    ):
        """Generate lightcurves for a point source with certain size.

        The lightcurves are generated based on the microlensing map convolved with the source
        size.

        The generated lightcurves will have the same length of time as the "time" array provided.

        :param deflector_redshift: Redshift of the deflector
        :param source_redshift: Redshift of the source
        :param cosmology: Cosmology object for the lens class
        :param source_size: Size of the source in arcseconds
        :param mean_microlens_mass_in_kg: Mean mass of the microlenses in kg
        :param effective_transverse_velocity: Transverse velocity in source plane (in km/s)
        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'magnification'. If 'magnitude', the lightcurve is returned in magnitudes normalized to the macro magnification.
                                If 'magnification', the lightcurve is returned in magnification without normalization. Default is 'magnitude'.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.
        :param return_track_coords: Whether to return the track coordinates of the lightcuve(s) or not. Default is False.
        :param return_time_array: Whether to return the time array used for the lightcurve(s) or not. Default is False.

        Returns a tuple of:
        light_curve: a numpy array of lightcurves for the point source.
        tracks: if requested, a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        """
        mag_map_2d = self.magnification_map.magnifications

        # optimize the magnification map for the convolution
        if mag_map_2d.dtype != np.float32:
            mag_map_2d = mag_map_2d.astype(np.float32)

        # Get the convolved magmap after convolving the magnification map with a Gaussian kernel
        convolved_map = self._get_convolved_map(
            source_size=source_size, return_source_kernel=False
        )

        light_curve_time_in_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years

        # get pixel size of the magnification map in meters
        # magnification map pixel size
        pixel_size_arcsec = (
            self.magnification_map.pixel_size
        )  # TODO: Is kpc_proper_per_arcmin the correct conversion factor?
        pixel_size_kpc = (
            ((cosmology.kpc_proper_per_arcmin(source_redshift)).to(u.kpc / u.arcsec))
            * pixel_size_arcsec
            * u.arcsec
        )
        pixel_size_magnification_map = (
            pixel_size_kpc.to(u.m)
        ).value  # pixel size in meters
        # print("pixel_size_magnification_map: ", pixel_size_magnification_map, "m")

        # randomly generates a light curves based on the convolved map
        light_curves_list = (
            []
        )  # list of light curves       # has a shape of (num_lightcurves, length of light curve)
        tracks = (
            []
        )  # list of [x_positions,y_positions]     # has a shape of (num_lightcurves, 2, length of light curve)
        time_arrays = []  # list of time arrays for each light curve

        mean_magnification_convolved_map = np.nanmean(self.convolved_map)
        # print("mean_magnification_convolved_map: ", mean_magnification_convolved_map)
        # print("mu_ave from magnification map: ", self.magnification_map.mu_ave)

        for _ in range(num_lightcurves):
            light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=convolved_map,
                pixel_size=pixel_size_magnification_map,  # TODO: Make sure that the units for theta_star and pixel_size are in arcsec
                effective_transverse_velocity=effective_transverse_velocity,
                light_curve_time_in_years=light_curve_time_in_years,
                pixel_shift=0,
                x_start_position=None,
                y_start_position=None,
                phi_travel_direction=None,
                return_track_coords=True,
                random_seed=None,
            )

            if lightcurve_type == "magnitude":
                # print("Extracting magnitude for light curve...")
                light_curve = -2.5 * np.log10(
                    light_curve / np.abs(mean_magnification_convolved_map)
                )
            elif lightcurve_type == "magnification":
                # print("Extracting magnification for light curve...")
                light_curve = light_curve
            else:
                raise ValueError(
                    "Lightcurve type not recognized. Please use 'magnitude' or 'magnification'."
                )

            light_curves_list.append(light_curve)
            tracks.append(np.array([x_positions, y_positions]))
            time_arrays.append(np.linspace(0, self.time_duration, len(light_curve)))

        # convert to numpy array
        light_curves_list = np.array(
            light_curves_list
        )  # has a shape of (num_lightcurves, length of light curve)
        tracks = np.array(
            tracks
        )  # has a shape of (num_lightcurves, 2, length of light curve)

        if return_track_coords and not (return_time_array):
            return light_curves_list, tracks
        if return_time_array and not (return_track_coords):
            return light_curves_list, time_arrays
        if return_track_coords and return_time_array:
            return light_curves_list, tracks, time_arrays
        if not (return_track_coords) and not (return_time_array):
            return light_curves_list

    def _generate_supernova_lightcurve(self):
        """Generate lightcurve for a supernova."""
        pass

    def generate_agn_lightcurve(
        self,
        source_redshift,
        # deflector_redshift,
        cosmology,
        lightcurve_type="magnitude",
        v_transverse=1000,  # TODO: figure out this velocity based on the model in Section 3.3 of https://ui.adsabs.harvard.edu/abs/2020MNRAS.495..544N
        num_lightcurves=1,
        r_out=1000,
        r_resolution=1000,
        smbh_mass_exp=8.0,
        inclination_angle=0,
        black_hole_spin=0,  # Spin of the black hole
        observer_frame_wavelength_in_nm=600,  # Wavelength in nanometers used to determine black body flux. For the surface flux density of the AccretionDisk at desired wavelength.
        eddington_ratio=0.15,  # Eddington ratio of the accretion disk
        mean_microlens_mass_in_kg=1
        * const.M_sun.to(u.kg),  # Mean mass of the microlenses in kg
        return_track_coords=False,
        return_time_array=False,
    ):
        """Generate microlensing lightcurves for a quasar(AGN) with the
        AccretionDisk model from amoeba. Returns a list of lightcurves based on
        the number of lightcurves requested.

        :param source_redshift: Redshift of the source
        :param deflector_redshift: Redshift of the deflector
        :param cosmology: Cosmology object for the lens class
        :param lightcurve_type: Type of lightcurve to generate, either
            'magnitude' or 'magnification'. Default is 'magnitude'.
        :param v_transverse: Transverse velocity in source plane (in
            km/s). Default is 1000 km/s.
        :param num_lightcurves: Number of lightcurves to generate.
            Default is 1.
        :param r_out: Outer radius of the accretion disk in
            gravitational radii. This typically can be chosen as 10^3 to
            10^5 [R_g] Default is 1000.
        :param r_resolution: Resolution of the accretion disk in
            gravitational radii. Default is 1000.
        :param smbh_mass_exp: Exponent of the mass of the supermassive
            black hole in kg. Default is 8.0.
        :param inclination_angle: Inclination angle of the disk in
            degrees. Default is 0.
        :param observer_frame_wavelength_in_nm: Wavelength in nanometers
            used to determine black body flux. For the surface flux
            density of the AccretionDisk at desired wavelength. Default
            is 600 nm.
        :param eddington_ratio: Eddington ratio of the accretion disk.
            Default is 0.15.
        :param mean_microlens_mass_in_kg: Mean mass of the microlenses
            in kg. Default is 1 * const.M_sun.to(u.kg).
        :param min_disk_radius: Minimum radius of the accretion disk in
            gravitational radii. Default is 6.
        :param return_track_coords: Whether to return the track
            coordinates of the lightcuve(s) or not. Default is False.
        :param return_time_array: Whether to return the time array used
            for the lightcurve(s) or not. Default is False.
        """

        mag_map_2d = self.magnification_map.magnifications

        #########
        # removed the dependency on the amoeba package by using source morphology from slsim.Util.astro_util functions
        #########

        # invert redshifts to find locally emitted wavelengths
        redshiftfactor = 1 / (1 + source_redshift)
        totalshiftfactor = redshiftfactor  # * self.g_array # we are not using the g_array for now, # TODO: Check with Henry if this is needed?
        rest_frame_wavelength = totalshiftfactor * observer_frame_wavelength_in_nm

        accretion_disk_emission_map = calculate_accretion_disk_emission(
            r_out=r_out,
            r_resolution=r_resolution,
            inclination_angle=inclination_angle,
            rest_frame_wavelength_in_nanometers=rest_frame_wavelength,
            black_hole_mass_exponent=smbh_mass_exp,
            black_hole_spin=black_hole_spin,
            eddington_ratio=eddington_ratio,
            return_spectral_radiance_distribution=True,
        )
        accretion_disk_emission_map = np.array(accretion_disk_emission_map)

        # since we are using the accretion disk emission map as a kernel, we need to normalize it
        normalized_emission_map = accretion_disk_emission_map / np.sum(
            accretion_disk_emission_map
        )
        # print("normalized_emission_map finished, shape: ", normalized_emission_map.shape)

        ## determine physical pixel sizes in source plane
        # emission map pixel size
        gravitational_radius_of_smbh = calculate_gravitational_radius(smbh_mass_exp)
        pixel_size_emission_map = (
            2
            * (r_out * gravitational_radius_of_smbh)
            / np.size(normalized_emission_map, 0)
        )
        pixel_size_emission_map = pixel_size_emission_map.to(
            u.m
        ).value  # convert to meters
        # print("pixel_size_emission_map: ", pixel_size_emission_map, "m")

        # magnification map pixel size
        pixel_size_arcsec = (
            self.magnification_map.pixel_size
        )  # TODO: Is kpc_proper_per_arcmin the correct conversion factor?
        pixel_size_kpc = (
            ((cosmology.kpc_proper_per_arcmin(source_redshift)).to(u.kpc / u.arcsec))
            * pixel_size_arcsec
            * u.arcsec
        )
        pixel_size_magnification_map = (
            pixel_size_kpc.to(u.m)
        ).value  # pixel size in meters
        # print("pixel_size_magnification_map: ", pixel_size_magnification_map, "m")

        # rescale the emission map pixels to the magnification array pixels
        pixel_ratio = pixel_size_emission_map / pixel_size_magnification_map
        # print("pixel_ratio: ", pixel_ratio)
        rescaled_emission_map = rescale(normalized_emission_map, pixel_ratio)
        rescaled_emission_map = rescaled_emission_map / np.sum(
            rescaled_emission_map
        )  # normalize the kernel, just in case
        # print("rescaled_emission_map finished, shape: ", rescaled_emission_map.shape)
        # print("rescaled_emission_map: ", rescaled_emission_map)

        # convolve the magnification map with the emission map
        self.convolved_map = fftconvolve(mag_map_2d, rescaled_emission_map, mode="same")
        # print("convolved_map finished, shape: ", self.convolved_map.shape)

        LCs = []
        tracks = []
        time_arrays = []

        time_duration_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years

        mean_magnification_convolved_map = np.nanmean(self.convolved_map)
        # print("mean_magnification_convolved_map: ", mean_magnification_convolved_map)
        # print("mu_ave from magnification map: ", self.magnification_map.mu_ave)

        for _ in range(num_lightcurves):
            light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=self.convolved_map,
                pixel_size=pixel_size_magnification_map,  # Make sure that the units for theta_star and pixel_size are in arcsec
                effective_transverse_velocity=v_transverse,
                light_curve_time_in_years=time_duration_years,
                pixel_shift=0,
                x_start_position=None,
                y_start_position=None,
                phi_travel_direction=None,
                return_track_coords=True,
                random_seed=None,
            )

            if lightcurve_type == "magnitude":
                # print("Extracting magnitude for light curve...")
                light_curve = -2.5 * np.log10(
                    light_curve / mean_magnification_convolved_map
                )
            elif lightcurve_type == "magnification":
                # print("Extracting magnification for light curve...")
                light_curve = light_curve
            else:
                raise ValueError(
                    "Lightcurve type not recognized. Please use 'magnitude' or 'magnification'."
                )
            LCs.append(light_curve)
            tracks.append(np.array([x_positions, y_positions]))
            time_arrays.append(np.linspace(0, self.time_duration, len(light_curve)))

        if return_track_coords and not (return_time_array):
            return LCs, tracks

        if return_time_array and not (return_track_coords):
            return LCs, time_arrays

        if return_track_coords and return_time_array:
            return LCs, tracks, time_arrays

        if not (return_track_coords) and not (return_time_array):
            return LCs

    def plot_lightcurves_and_magmap(
        self, lightcurves, tracks=None, lightcurve_type="magnitude"
    ):
        """Plot the point source lightcurve."""
        fig, ax = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[2, 1])

        time_array = np.linspace(0, self.time_duration, len(lightcurves[0]))  # in days

        # light curves
        for i in range(len(lightcurves)):
            ax[0].plot(time_array, lightcurves[i], label=f"Lightcurve {i+1}")
        ax[0].set_xlabel("Time (days)")

        mean_magnification_convolved_map = np.nanmean(self.convolved_map)

        if lightcurve_type == "magnitude":
            ax[0].set_ylabel(
                "Magnitude $\\Delta m = -2.5 \\log_{10} (\\mu / \\mu_{\\text{av}})$"
            )
            im_to_show = -2.5 * np.log10(
                self.convolved_map / np.abs(mean_magnification_convolved_map)
            )  # TODO: should you divide by mu_ave of original map or the convolved map?
        elif lightcurve_type == "magnification":
            ax[0].set_ylabel("Magnification $\\mu$")
            im_to_show = self.convolved_map

        ax[0].set_ylim(np.nanmin(im_to_show), np.nanmax(im_to_show))
        ax[0].legend()

        # magmap
        conts = ax[1].imshow(
            im_to_show,
            cmap="viridis_r",
            extent=[
                (self.magnification_map.center_x - self.magnification_map.half_length_x)
                / self.magnification_map.theta_star,
                (self.magnification_map.center_x + self.magnification_map.half_length_x)
                / self.magnification_map.theta_star,
                (self.magnification_map.center_y - self.magnification_map.half_length_y)
                / self.magnification_map.theta_star,
                (self.magnification_map.center_y + self.magnification_map.half_length_y)
                / self.magnification_map.theta_star,
            ],
            origin="lower",
        )
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(conts, cax=cax)
        if lightcurve_type == "magnitude":
            cbar.set_label(
                "Microlensing $\\Delta m = -2.5 \\log_{10} (\\mu / \\mu_{\\text{av}})$ (magnitudes)"
            )
        elif lightcurve_type == "magnification":
            cbar.set_label("Microlensing magnification $\\mu$")
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
                    (tracks[j][1] - mid_x_pixel)
                    * delta_x
                    / self.magnification_map.theta_star,
                    (tracks[j][0] - mid_y_pixel)
                    * delta_y
                    / self.magnification_map.theta_star,
                    "w-",
                    lw=1,
                )
                ax[1].text(
                    (tracks[j][1][0] - mid_x_pixel)
                    * delta_x
                    / self.magnification_map.theta_star,
                    (tracks[j][0][0] - mid_y_pixel)
                    * delta_y
                    / self.magnification_map.theta_star,
                    str(j + 1),
                    color="white",
                    fontsize=16,
                )

        return ax
