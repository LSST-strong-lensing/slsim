__author__ = "Paras Sharma"


# import gc  # for garbage collection
import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import rescale

from slsim.Microlensing.magmap import MagnificationMap

from slsim.Util.astro_util import (
    extract_light_curve,
)

from slsim.Microlensing.source_morphology import (
    GaussianSourceMorphology,
    AGNSourceMorphology,
)


class MicrolensingLightCurve(object):
    """Class to generate microlensing lightcurve(s) for a single source based
    on the magnification map, source morphology, and lens properties."""

    def __init__(
        self,
        magnification_map: MagnificationMap,
        time_duration: float,
        point_source_morphology: str = "gaussian",  # 'gaussian' or 'agn' or 'supernovae' #TODO: supernovae not implemented yet!
        kwargs_source_morphology: dict = {},
    ):
        """
        :param magnification_map: MagnificationMap object, if not provided.
        :param time_duration: Time duration for which the lightcurve is needed (in days).
        :param point_source_morphology: Type of source morphology to use. Default is 'gaussian'. Options are 'gaussian' or 'agn' (Accretion Disk) or 'supernovae'.
        :param kwargs_source_morphology: Dictionary of keyword arguments for the source morphology class. This should be as per the source morphology type.

            For example, for Gaussian source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmo, "source_size": source_size, }.

            For AGN source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmology,
            "r_out": r_out, "r_resolution": r_resolution, "smbh_mass_exp": smbh_mass_exp, "inclination_angle": inclination_angle,
            "black_hole_spin": black_hole_spin, "observer_frame_wavelength_in_nm": observer_frame_wavelength_in_nm,
            "eddington_ratio": eddington_ratio, }.
        """

        self.magnification_map = magnification_map
        self.time_duration = time_duration
        self.point_source_morphology = point_source_morphology
        self.kwargs_source_morphology = kwargs_source_morphology

        # Initialize the convolved map and source morphology
        self.convolved_map = None
        self.source_morphology = None

    def get_convolved_map(
        self,
        return_source_morphology=False,
    ):
        """Get the convolved map based on the source morphology."""
        if self.point_source_morphology == "gaussian":
            # Gaussian source morphology
            source_morphology = GaussianSourceMorphology(
                **self.kwargs_source_morphology,  # sets the source size, redshift, and cosmology
                length_x=self.magnification_map.half_length_x * 2,
                length_y=self.magnification_map.half_length_y * 2,
                num_pix_x=self.magnification_map.num_pixels_x,
                num_pix_y=self.magnification_map.num_pixels_y,
                center_x=0,
                center_y=0,
            )

            # convolve the magnification map with the Gaussian kernel
            self.convolved_map = fftconvolve(
                self.magnification_map.magnifications,
                source_morphology.kernel_map,
                mode="same",
            )

        elif self.point_source_morphology == "agn":
            # AGN source morphology
            source_morphology = AGNSourceMorphology(
                **self.kwargs_source_morphology,
            )
            cosmo = source_morphology.cosmo
            source_redshift = source_morphology.source_redshift

            # magnification map pixel size
            pixel_size_magnification_map = self.magnification_map.get_pixel_size_meters(
                source_redshift=source_redshift, cosmo=cosmo
            )

            # source kernel pixel size
            pixel_size_kernel_map = source_morphology.pixel_scale_m

            # rescale the kernel to the pixel size of the magnification map
            pixel_ratio = pixel_size_kernel_map / pixel_size_magnification_map
            # print(f"pixel size of magnification map: {pixel_size_magnification_map}")
            # print(f"pixel size of kernel map: {pixel_size_kernel_map}")
            # print(f"Pixel ratio: {pixel_ratio}")
            rescaled_kernel_map = rescale(source_morphology.kernel_map, pixel_ratio)

            # normalize the rescaled kernel, just in case
            rescaled_kernel_map = rescaled_kernel_map / np.nansum(rescaled_kernel_map)

            # convolve the magnification map with the kernel map, #TODO: make this a cross-correlation
            self.convolved_map = fftconvolve(
                self.magnification_map.magnifications, rescaled_kernel_map, mode="same"
            )

        elif self.point_source_morphology == "supernovae":
            # Supernovae source morphology
            source_morphology = None
            raise NotImplementedError(
                "Supernovae source morphology is not implemented yet."
            )

        else:
            raise ValueError(
                "Invalid source morphology type. Choose 'gaussian', 'agn', or 'supernovae'."
            )

        if source_morphology:
            self.source_morphology = source_morphology

        if return_source_morphology:
            return self.convolved_map, source_morphology
        else:
            return self.convolved_map

    def generate_lightcurves(
        self,
        source_redshift,
        cosmo,
        lightcurve_type="magnitude",
        effective_transverse_velocity=1000,  # Transverse velocity in source plane (in km/s)
        num_lightcurves=1,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
        return_track_coords=False,
        return_time_array=False,
    ):
        """Generate lightcurves for a point source based on the convolved map.

        :param source_redshift: Redshift of the source
        :param cosmo: Cosmology object for the lens class
        :param lightcurve_type: Type of lightcurve to generate, either
            'magnitude' or 'magnification'. If 'magnitude', the
            lightcurve is returned in magnitudes normalized to the macro
            magnification. If 'magnification', the lightcurve is
            returned in magnification without normalization. Default is
            'magnitude'.
        :param effective_transverse_velocity: Transverse velocity in
            source plane (in km/s)
        :param num_lightcurves: Number of lightcurves to generate.
            Default is 1.
        :param x_start_position: Starting x position of the lightcurve
            in pixel coordinates. Default is None.
        :param y_start_position: Starting y position of the lightcurve
            in pixel coordinates. Default is None.
        :param phi_travel_direction: Angle of the travel direction in
            degrees. Default is None.
        :param return_track_coords: Whether to return the track
            coordinates of the lightcuve(s) or not. Default is False.
        :param return_time_array: Whether to return the time array used
            for the lightcurve(s) or not. Default is False.
        :return: A tuple of lightcurves, tracks, and time arrays if
            requested. If only lightcurves are requested, a list of
            lightcurves is returned.
        """

        # Get the convolved magmap after convolving the magnification map with a Gaussian kernel
        convolved_map = self.get_convolved_map(
            return_source_morphology=False,
        )

        # determine physical pixel sizes in source plane
        pixel_size_magnification_map = self.magnification_map.get_pixel_size_meters(
            source_redshift=source_redshift, cosmo=cosmo
        )

        return self._generate_lightcurves(
            convolved_map=convolved_map,
            pixel_size_magnification_map=pixel_size_magnification_map,
            num_lightcurves=num_lightcurves,
            lightcurve_type=lightcurve_type,
            effective_transverse_velocity=effective_transverse_velocity,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            phi_travel_direction=phi_travel_direction,
            return_track_coords=return_track_coords,
            return_time_array=return_time_array,
        )

    def _generate_lightcurves(
        self,
        convolved_map,
        pixel_size_magnification_map,
        num_lightcurves=1,
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        effective_transverse_velocity=1000,  # Transverse velocity in source plane (in km/s)
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
        return_track_coords=False,
        return_time_array=False,
    ):
        """Generate lightcurves for a point source based on the convolved map.

        :param convolved_map: Convolved magnification map
        :param pixel_size_magnification_map: Pixel size of the
            magnification map in meters
        :param num_lightcurves: Number of lightcurves to generate.
            Default is 1.
        :param lightcurve_type: Type of lightcurve to generate, either
            'magnitude' or 'magnification'. If 'magnitude', the
            lightcurve is returned in magnitudes normalized to the macro
            magnification. If 'magnification', the lightcurve is
            returned in magnification without normalization. Default is
            'magnitude'.
        :param effective_transverse_velocity: Transverse velocity in
            source plane (in km/s)
        :param x_start_position: Starting x position of the lightcurve
            in pixel coordinates. Default is None.
        :param y_start_position: Starting y position of the lightcurve
            in pixel coordinates. Default is None.
        :param phi_travel_direction: Angle of the travel direction in
            degrees. Default is None.
        :param return_track_coords: Whether to return the track
            coordinates of the lightcuve(s) or not. Default is False.
        :param return_time_array: Whether to return the time array used
            for the lightcurve(s) or not. Default is False.
        :return: A tuple of lightcurves, tracks, and time arrays if
            requested. If only lightcurves are requested, a list of
            lightcurves is returned.
        """

        LCs = []
        tracks = []
        time_arrays = []

        time_duration_years = (
            self.time_duration / 365.25
        )  # converting time_duration from days to years

        for _ in range(num_lightcurves):
            light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=convolved_map,
                pixel_size=pixel_size_magnification_map,  # Make sure that the units for theta_star and pixel_size are in arcsec
                effective_transverse_velocity=effective_transverse_velocity,
                light_curve_time_in_years=time_duration_years,
                pixel_shift=0,
                x_start_position=x_start_position,
                y_start_position=y_start_position,
                phi_travel_direction=phi_travel_direction,
                return_track_coords=True,
                random_seed=None,
            )

            if lightcurve_type == "magnitude":
                # print("Extracting magnitude for light curve...")
                light_curve = -2.5 * np.log10(
                    light_curve / np.abs(self.magnification_map.mu_ave)
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

        if lightcurve_type == "magnitude":
            ax[0].set_ylabel(
                "Magnitude $\\Delta m = -2.5 \\log_{10} (\\mu / \\mu_{\\text{av}})$"
            )
            im_to_show = -2.5 * np.log10(
                self.convolved_map / np.abs(self.magnification_map.mu_ave)
            )
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
