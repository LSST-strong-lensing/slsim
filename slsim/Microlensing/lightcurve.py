__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented
# from Microlensing import HENRYS_AMOEBA_PATH
# import sys
# sys.path.append(HENRYS_AMOEBA_PATH)
# amoeba must be installed in the environment!
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.magnification_map import MagnificationMap as AmoebaMagnificationMap
import amoeba.Util.util as util


from slsim.Microlensing.magmap import MagnificationMap

# from slsim.Sources.agn import agn_bounds_dict # to set the limits for the AGN Disk parameters

import gc  # for garbage collection
import numpy as np
from scipy.signal import fftconvolve
import astropy.constants as const
# from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        source_redshift,
        cosmology,
        kwargs_PointSource={},
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        num_lightcurves=1,  # Number of lightcurves to generate
        return_track_coords=False,
        return_time_array=False,
    ):
        """Generate lightcurves for a point source with certain size.

        The lightcurves are generated based on the microlensing map convolved with the source
        size.

        The generated lightcurves will have the same length of time as the "time" array provided.

        :param source_redshift: Redshift of the source
        :param cosmology: Cosmology object for the lens class
        :param kwargs_PointSource: Keyword arguments for the Point Source Model. An example is:
                                   kwargs_PointSource = {"source_size": 0.1, "effective_transverse_velocity": 1000} with units in arcsec and km/s for size and velocity respectively.
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

        if "source_size" in kwargs_PointSource:
            source_size = kwargs_PointSource[
                "source_size"
            ]  # TODO: Make sure this is supplied in arc sec units
        else:
            raise ValueError(
                "Source size not provided in kwargs_PointSource. Please provide a source size in arc seconds."
            )

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
        convolved_map = self._get_convolved_map(
            source_size=source_size, return_source_kernel=False
        )

        # get parameters for the light curve
        if "effective_transverse_velocity" in kwargs_PointSource:
            effective_transverse_velocity = kwargs_PointSource[
                "effective_transverse_velocity"
            ]  # TODO: Make sure this is supplied in km/s units
        else:
            raise ValueError(
                "Effective transverse velocity not provided in kwargs_PointSource. Please provide a effective transverse velocity in km/s in the  Source Plane."
            )

        light_curve_time_in_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years

        # extract_light_curve function requires the pixel size in meters
        pixel_size_arcsec = (
            self.magnification_map.pixel_size
        )  # TODO: Is kpc_proper_per_arcmin the correct conversion factor?
        pixel_size_kpc = (
            ((cosmology.kpc_proper_per_arcmin(source_redshift)).to(u.kpc / u.arcsec))
            * pixel_size_arcsec
            * u.arcsec
        )
        pixel_size_meter = (pixel_size_kpc.to(u.m)).value  # pixel size in meters

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

        for i in range(num_lightcurves):
            light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=convolved_map,
                pixel_size=pixel_size_meter,  # TODO: Make sure that the units for theta_star and pixel_size are in arcsec
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

    def _generate_supernova_lightcurve(self):
        """Generate lightcurve for a supernova."""
        pass

    def generate_agn_lightcurve(
        self,
        source_redshift,
        deflector_redshift,
        cosmology,
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        v_transverse=1000,  # Transverse velocity in source plane (in km/s) #TODO: figure out this velocity based on the model in Section 3.3 of https://ui.adsabs.harvard.edu/abs/2020MNRAS.495..544N
        num_lightcurves=1,  # Number of lightcurves to generate
        corona_height=10,
        smbh_mass_exp=8.0,
        inclination_angle=0,
        observer_frame_wavelength_in_nm=600,  # Wavelength in nanometers used to determine black body flux. For the surface flux density of the AccretionDisk at desired wavelength.
        eddington_ratio=0.15,  # Eddington ratio of the accretion disk
        mean_microlens_mass_in_kg=1
        * const.M_sun.to(u.kg),  # Mean mass of the microlenses in kg
        min_disk_radius=6,  # Minimum radius of the accretion disk in gravitational radii
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
        :param corona_height: Height of the corona above the disk in
            gravitational radii. Default is 10.
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

        # AMOEBA CODE!
        # ------------------------------------------------------------------------------------------
        MagMap = AmoebaMagnificationMap(
            source_redshift,
            deflector_redshift,
            mag_map_2d,
            self.magnification_map.kappa_tot,  # TODO: ask if this is correct for "Convergence of the lensing potential at the location of the image"?
            self.magnification_map.shear,
            mean_microlens_mass_in_kg=mean_microlens_mass_in_kg,  # TODO: ask if this can be related to theta_star in MagnificationMap class?
            total_microlens_einstein_radii=2
            * self.magnification_map.half_length_x
            / self.magnification_map.theta_star
            * 2,  # assuming square map
            OmM=cosmology.Om0,
            H0=cosmology.H0.to(u.km / (u.s * u.Mpc)).value,
        )

        disk_projection = Disk.calculate_surface_intensity_map(
            observer_frame_wavelength_in_nm=observer_frame_wavelength_in_nm
        )  # TODO: ask if this should be connected to the band used for observations? YES!
        disk_projection.flux_array = disk_projection.flux_array / np.sum(
            disk_projection.flux_array
        )  # disk projection is normalized to 1.
        convolution = MagMap.convolve_with_flux_projection(disk_projection)
        self.convolved_map = convolution.magnification_array

        # del MagMap, Disk, disk_projection

        # self.amoeba_convolution = convolution
        # convolved_mag_map = convolution.magnification_array

        # ------------------------------------------------------------------------------------------

        LCs = []
        tracks = []
        time_arrays = []

        time_duration_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years

        # extract_light_curve function requires the pixel size in meters
        pixel_size_arcsec = (
            self.magnification_map.pixel_size
        )  # TODO: Is kpc_proper_per_arcmin the correct conversion factor?
        pixel_size_kpc = (
            ((cosmology.kpc_proper_per_arcmin(source_redshift)).to(u.kpc / u.arcsec))
            * pixel_size_arcsec
            * u.arcsec
        )
        pixel_size_meter = (pixel_size_kpc.to(u.m)).value  # pixel size in meters

        mean_magnification_convolved_map = np.nanmean(self.convolved_map)
        # print("mean_magnification_convolved_map: ", mean_magnification_convolved_map)
        # print("mu_ave from magnification map: ", self.magnification_map.mu_ave)

        for jj in range(num_lightcurves):
            # light_curve, x_positions, y_positions = convolution.pull_light_curve(
            #     v_transverse, time_duration_years, return_track_coords=True
            # )

            light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=self.convolved_map,
                pixel_size=pixel_size_meter,  # TODO: Make sure that the units for theta_star and pixel_size are in arcsec
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


class MicrolensingLightCurveFromLensModel(object):
    """Class to generate microlensing lightcurves based on the microlensing
    parameters for each image of a source."""

    def generate_point_source_microlensing_magnitudes(
        self,
        time,
        source_redshift,
        deflector_redshift,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        cosmology,
        kwargs_MagnificationMap={},
        kwargs_PointSource={},
        kwargs_AccretionDisk={},
    ):
        """Generate lightcurve magnitudes normalized to the mean magnification
        for gaussian point sources. For single source only, it produces the
        lightcurve magnitudes for all images of the source.

        Returns a numpy array of microlensing magnitudes (does not
        include macro-magnifications) with the shape (num_images,
        len(time)).
        """
        # if time is a number
        if isinstance(time, (int, float)):
            time_array = np.array([time])
        elif isinstance(time, np.ndarray):
            time_array = time
        elif isinstance(time, list):
            time_array = np.array(time)
        else:
            raise ValueError(
                "Time array not provided in the correct format. Please provide a time array in days."
            )

        if kwargs_AccretionDisk != {} and kwargs_PointSource == {}:
            # if kwargs_AccretionDisk is provided, use the AGN lightcurve method
            lightcurves, __tracks, __time_arrays = self.generate_agn_lightcurve(
                time_array,
                source_redshift,
                deflector_redshift,
                kappa_star_images,
                kappa_tot_images,
                shear_images,
                cosmology,
                kwargs_MagnificationMap=kwargs_MagnificationMap,
                kwargs_AccretionDisk=kwargs_AccretionDisk,
                lightcurve_type="magnitude",
                num_lightcurves=1,
            )

        if kwargs_PointSource != {} and kwargs_AccretionDisk == {}:
            # if kwargs_PointSource is provided, use the Point Source lightcurve method
            lightcurves, __tracks, __time_arrays = (
                self.generate_point_source_lightcurve(
                    time_array,
                    source_redshift,
                    kappa_star_images,
                    kappa_tot_images,
                    shear_images,
                    cosmology,
                    kwargs_MagnificationMap=kwargs_MagnificationMap,
                    kwargs_PointSource=kwargs_PointSource,
                    lightcurve_type="magnitude",
                    num_lightcurves=1,
                )
            )

        if kwargs_PointSource != {} and kwargs_AccretionDisk != {}:
            # if both kwargs_PointSource and kwargs_AccretionDisk are provided, use the AGN lightcurve method
            lightcurves, __tracks, __time_arrays = self.generate_agn_lightcurve(
                time_array,
                source_redshift,
                deflector_redshift,
                kappa_star_images,
                kappa_tot_images,
                shear_images,
                cosmology,
                kwargs_MagnificationMap=kwargs_MagnificationMap,
                kwargs_AccretionDisk=kwargs_AccretionDisk,
                lightcurve_type="magnitude",
                num_lightcurves=1,
            )

        # Here we choose just 1 lightcurve for the point sources
        lightcurves_single = np.zeros(
            (len(lightcurves), len(time_array))
        )  # has shape (num_images, len(time))
        for i in range(len(lightcurves)):
            lightcurves_single[i] = lightcurves[i][0]

        if isinstance(time, (int, float)):
            # if time is a number, return the magnitude for the first time
            lightcurves_single = lightcurves_single[:, 0]

        return lightcurves_single

    def generate_point_source_lightcurve(
        self,
        time,
        source_redshift,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        cosmology,
        kwargs_MagnificationMap={},
        kwargs_PointSource={},
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        num_lightcurves=1,  # Number of lightcurves to generate
    ):
        """Generate lightcurves for one single point source with certain size,
        but for all images of that source based on the lens model. The point
        source is simulated as a gaussian source with a certain size. The size
        is given in arc seconds.

        The lightcurves are generated based on the microlensing map convolved with the source
        size.

        The generated lightcurves will have the same length of time as the "time" array provided.

        :param time: Time array for which the lightcurve is needed.
        :param source_redshift: Redshift of the source
        :param kappa_star_images: list containing the kappa star for each image of the source.
        :param kappa_tot_images: list containing the kappa total for each image of the source.
        :param shear_images: list containing the shear for each image of the source.
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class.
        :param kwargs_PointSource: Keyword arguments for the Point Source Model.
        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'magnification'. If 'magnitude', the lightcurve is returned in magnitudes normalized to the macro magnification.
                                If 'magnification', the lightcurve is returned in magnification without normalization. Default is 'magnitude'.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.

        Returns a tuple of:
        lightcurves: a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks: a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays: corresponding to each lightcurve
        """

        # generate magnification maps for each image of the source
        magmaps_images = self.generate_magnification_maps_from_microlensing_params(
            kappa_star_images=kappa_star_images,
            kappa_tot_images=kappa_tot_images,
            shear_images=shear_images,
            kwargs_MagnificationMap=kwargs_MagnificationMap,
        )

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and len(time) > 1:
            lightcurve_duration = time[-1] - time[0]
        elif (isinstance(time, np.ndarray) or isinstance(time, list)) and len(
            time
        ) == 1:
            lightcurve_duration = time[0]  # TODO: check if this is correct thing to do?
        else:
            raise ValueError(
                "Time array not provided in the correct format. Please provide a time array in days."
            )

        # generate lightcurves for each image of the source
        lightcurves = (
            []
        )  # a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks = (
            []
        )  # a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays = []  # corresponding to each lightcurve
        for i in range(len(kappa_star_images)):
            ml_lc = MicrolensingLightCurve(
                magnification_map=magmaps_images[i], time_duration=lightcurve_duration
            )
            curr_lightcurves, curr_tracks, curr_time_arrays = (
                ml_lc.generate_point_source_lightcurve(
                    source_redshift=source_redshift,
                    cosmology=cosmology,
                    kwargs_PointSource=kwargs_PointSource,
                    lightcurve_type=lightcurve_type,
                    num_lightcurves=num_lightcurves,
                    return_track_coords=True,
                    return_time_array=True,
                )
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

            lightcurves.append(curr_lightcurves_interpolated)
            tracks.append(curr_tracks)
            time_arrays.append(updated_curr_time_arrays)

        # light curves is a list with first len being number of images and second len being number of lightcurves for each image
        # tracks is a list with first len being number of images and second len being number of tracks for each image
        # time_arrays is a list with first len being number of images and second len being number of lightcurves for each image

        return lightcurves, tracks, time_arrays

    def generate_magnification_maps_from_microlensing_params(
        self,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        kwargs_MagnificationMap={},
    ):
        """Generate magnification maps for each image of the source based on
        the image positions and the lens model. It requires the following
        parameters:

        :param kappa_star_images: Kappa star for each image of the source.
        :param kappa_tot_images: Kappa total for each image of the source.
        :param shear_images: Shear for each image of the source.
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class.

        Returns:
        magmaps_images: a list which contains the [magnification map for each image of the source].
        """
        # generate magnification maps for each image of the source
        magmaps_images = []
        for i in range(len(kappa_star_images)):
            # generate magnification maps for each image of the source
            magmap = MagnificationMap(
                kappa_tot=kappa_tot_images[i],
                shear=shear_images[i],
                kappa_star=kappa_star_images[i],
                **kwargs_MagnificationMap,  # TODO: refactor the size of the magmap later!
            )
            magmaps_images.append(magmap)

        return magmaps_images

    def generate_agn_lightcurve(
        self,
        time,
        source_redshift,
        deflector_redshift,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        cosmology,
        kwargs_MagnificationMap={
            "rectangular": True,
            "half_length_x": 25,
            "half_length_y": 25,
            "mass_function": "kroupa",
            "m_lower": 0.08,
            "m_upper": 100,
            "num_pixels_x": 5000,
            "num_pixels_y": 5000,
        },
        kwargs_AccretionDisk={
            "smbh_mass_exp": 8.0,
            "corona_height": 10,
            "inclination_angle": 0,
            "observer_frame_wavelength_in_nm": 600,  # TODO: this needs to be connected with the band used for observations
            "eddington_ratio": 0.15,
            "mean_microlens_mass_in_kg": 1 * const.M_sun.to(u.kg),
            "min_disk_radius": 6,
        },
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        num_lightcurves=1,  # Number of lightcurves to generate
    ):  # TODO: add the actual kwargs in the definition of the function
        """Generate lightcurves for one single quasar(AGN) source, but for all
        images of that source based on the lens model.

        The generated lightcurves will have the same length of time as the lightcurve_time provided in the source_class.

        :param time: Time array for which the lightcurve is needed.
        :param source_redshift: Redshift of the source
        :param deflector_redshift: Redshift of the deflector
        :param kappa_star_images: list containing the kappa star for each image of the source.
        :param kappa_tot_images: list containing the kappa total for each image of the source.
        :param shear_images: list containing the shear for each image of the source.
        :param cosmology: Cosmology object for the lens class
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class.
        :param kwargs_AccretionDisk: Keyword arguments for the AccretionDisk class.
        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'magnification'. If 'magnitude', the lightcurve is returned in magnitudes normalized to the macro magnification.
                                If 'magnification', the lightcurve is returned in magnification without normalization. Default is 'magnitude'.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.

        Returns a tuple of:
        lightcurves: a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks: a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays: corresponding to each lightcurve
        """
        if isinstance(time, np.ndarray) or isinstance(time, list) and len(time) > 1:
            time_duration = time[-1] - time[0]
        elif isinstance(time, np.array) or isinstance(time, list) and len(time) == 1:
            time_duration = time[0]
        else:
            raise ValueError(
                "Time array not provided in the correct format. Please provide a time array in days."
            )

        # generate magnification maps for each image of the source
        magmaps_images = self.generate_magnification_maps_from_microlensing_params(
            kappa_star_images=kappa_star_images,
            kappa_tot_images=kappa_tot_images,
            shear_images=shear_images,
            kwargs_MagnificationMap=kwargs_MagnificationMap,
        )

        # # save the variables for later use
        # self.kappa_star_images = kappa_star_images
        # self.kappa_tot_images = kappa_tot_images
        # self.shear_images = shear_images
        # self.magmaps_images = magmaps_images

        # generate lightcurves for each image of the source
        lightcurves = (
            []
        )  # a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks = (
            []
        )  # a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays = []  # corresponding to each lightcurve
        for i in range(len(kappa_star_images)):
            ml_lc = MicrolensingLightCurve(
                magnification_map=magmaps_images[i], time_duration=time_duration
            )
            curr_lightcurves, curr_tracks, curr_time_arrays = (
                ml_lc.generate_agn_lightcurve(
                    source_redshift=source_redshift,
                    deflector_redshift=deflector_redshift,
                    cosmology=cosmology,
                    lightcurve_type=lightcurve_type,
                    num_lightcurves=1,  # TODO: make a decision on how many lightcurves to generate!
                    return_track_coords=True,
                    return_time_array=True,
                    **kwargs_AccretionDisk,  # TODO: this might need updating depending on how we decide the parameters for the AccretionDisk
                )
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

            lightcurves.append(curr_lightcurves_interpolated)
            tracks.append(curr_tracks)
            time_arrays.append(updated_curr_time_arrays)

        # light curves is a list with first len being number of images and second len being number of lightcurves for each image
        # tracks is a list with first len being number of images and second len being number of tracks for each image
        # time_arrays is a list with first len being number of images and second len being number of lightcurves for each image

        return lightcurves, tracks, time_arrays

    def _interpolate_light_curve(self, lightcurve, time_array, time_array_new):
        """Interpolate the lightcurve to a new time array.

        Assuming "lightcurve" and "time_array" are 1D arrays of the same
        length. "time_array_new" is a 1D array of the new time array.
        """
        return np.interp(time_array_new, time_array, lightcurve)


## The credits for the following functions go to Henry Best (https://github.com/Henry-Best-01/Amoeba)


def pull_value_from_grid(array_2d, x_position, y_position):
    """This approximates the point (x_position, y_position) in a 2d array of
    values. x_position and y_position may be decimals, and are assumed to be
    measured in pixels.

    :param array_2d: 2 dimensional array of values.
    :param x_position: x coordinate in array_2d
    :param y_position: y coordinate in array_2d
    :return: approximation of array_2d at point (x_position, y_position)
    """
    assert x_position >= 0 and y_position >= 0
    assert x_position < np.size(array_2d, 0) and y_position < np.size(array_2d, 1)
    x_int = x_position // 1
    y_int = y_position // 1
    decx = x_position % 1
    decy = y_position % 1
    # baseval = array_2d[int(x_int), int(y_int)]
    # Calculate 1d gradients, allow for edge values
    if int(x_int) + 1 == np.size(array_2d, 0):
        dx = (
            (-1)
            * (array_2d[int(x_int) - 1, int(y_int)] - array_2d[int(x_int), int(y_int)])
            * decx
        )
    else:
        dx = (
            array_2d[int(x_int) + 1, int(y_int)] - array_2d[int(x_int), int(y_int)]
        ) * decx
    if int(y_int) + 1 == np.size(array_2d, 1):
        dy = (
            (-1)
            * (array_2d[int(x_int), int(y_int) - 1] - array_2d[int(x_int), int(y_int)])
            * decy
        )
    else:
        dy = (
            array_2d[int(x_int), int(y_int) + 1] - array_2d[int(x_int), int(y_int)]
        ) * decy
    return array_2d[int(x_int), int(y_int)] + dx + dy


# TODO: modify it so that we can use this function in units of theta_star or unitless.
def extract_light_curve(
    convolution_array,
    pixel_size,
    effective_transverse_velocity,
    light_curve_time_in_years,
    pixel_shift=0,
    x_start_position=None,
    y_start_position=None,
    phi_travel_direction=None,
    return_track_coords=False,
    random_seed=None,
):
    """Extracts a light curve from the convolution between two arrays by
    selecting a trajectory and calling pull_value_from_grid at each relevant
    point.

    :param convolution_array: The convolution between a flux distribtion
        and the magnification array due to microlensing. Note
        coordinates on arrays have (y, x) signature.
    :param pixel_size: Physical size of a pixel in the source plane, in
        meters
    :param effective_transverse_velocity: effective transverse velocity
        in the source plane, in km / s
    :param light_curve_time_in_years: duration of the light curve to
        generate, in years
    :param pixel_shift: offset of the SMBH with respect to the convolved
        map, in pixels
    :param x_start_position: the x coordinate to start pulling a light
        curve from, in pixels
    :param y_start_position: the y coordinate to start pulling a light
        curve from, in pixels
    :param phi_travel_direction: the angular direction of travel along
        the convolution, in degrees
    :param return_track_coords: bool switch allowing a list of relevant
        positions to be returned
    :return: list representing the microlensing light curve
    """
    rng = np.random.default_rng(seed=random_seed)

    if isinstance(effective_transverse_velocity, u.Quantity):
        effective_transverse_velocity = effective_transverse_velocity.to(u.m / u.s)
    else:
        effective_transverse_velocity *= u.km.to(u.m)
    if isinstance(light_curve_time_in_years, u.Quantity):
        light_curve_time_in_years = light_curve_time_in_years.to(u.s)
    else:
        light_curve_time_in_years *= u.yr.to(u.s)

    # check convolution if the map was large enough. Otherwise return original total flux.
    # Note the convolution should be weighted by the square of the pixel shift to conserve flux.
    if pixel_shift >= np.size(convolution_array, 0) / 2:
        print(
            "warning, flux projection too large for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    # determine the path length of the light curve in the source plane and include endpoints
    pixels_traversed = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    )

    n_points = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    ) + 2

    # ignore convolution artifacts
    if pixel_shift > 0:
        safe_convolution_array = convolution_array[
            pixel_shift:-pixel_shift, pixel_shift:-pixel_shift
        ]
    else:
        safe_convolution_array = convolution_array

    # guarantee that we will be able to extract a light curve from the safe region for any random start point
    if pixels_traversed >= np.size(safe_convolution_array, 0):
        print(
            "warning, light curve is too long for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    if x_start_position is not None:
        if x_start_position < 0:
            print(
                "Warning, chosen position lays in the convolution artifact region. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        x_start_position = rng.integers(0, np.size(safe_convolution_array, 0))

    if y_start_position is not None:
        if y_start_position < 0:
            print(
                "Warning, chosen position lays in the convolution artifact region. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        y_start_position = rng.integers(0, np.size(safe_convolution_array, 1))

    if phi_travel_direction is not None:
        angle = phi_travel_direction * np.pi / 180
        delta_x = pixels_traversed * np.cos(angle)
        delta_y = pixels_traversed * np.sin(angle)

        if (
            x_start_position + delta_x >= np.size(safe_convolution_array, 0)
            or y_start_position + delta_y >= np.size(safe_convolution_array, 1)
            or x_start_position + delta_x < 0
            or y_start_position + delta_y < 0
        ):
            print(
                "Warning, chosen track leaves the convolution array. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        # One quadrant will have enough space to extract the light curve
        success = None
        angle = rng.random() * 360 * np.pi / 180
        while success is None:
            angle += np.pi / 2
            delta_x = pixels_traversed * np.cos(angle)
            delta_y = pixels_traversed * np.sin(angle)
            if (
                x_start_position + delta_x < np.size(safe_convolution_array, 0)
                and y_start_position + delta_y < np.size(safe_convolution_array, 1)
                and x_start_position + delta_x >= 0
                and y_start_position + delta_y >= 0
            ):
                break

    # generate each (x, y) coordinate on the convolution
    x_positions = np.linspace(
        x_start_position, x_start_position + delta_x, int(n_points)
    )
    y_positions = np.linspace(
        y_start_position, y_start_position + delta_y, int(n_points)
    )

    light_curve = []
    for position in range(int(n_points)):
        light_curve.append(
            pull_value_from_grid(
                safe_convolution_array, x_positions[position], y_positions[position]
            )
        )
    if return_track_coords:
        return (
            np.asarray(light_curve),
            x_positions + pixel_shift,
            y_positions + pixel_shift,
        )
    return np.asarray(light_curve)
