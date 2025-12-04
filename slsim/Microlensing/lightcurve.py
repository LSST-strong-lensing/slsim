__author__ = "Paras Sharma"


import numpy as np
from skimage.transform import rescale

from slsim.Microlensing.magmap import MagnificationMap

from slsim.Util.astro_util import (
    extract_light_curve,
)

from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology
from slsim.Microlensing.source_morphology.gaussian import GaussianSourceMorphology
from slsim.Util.param_util import convolved_image


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
        :param time_duration: Time duration for which the lightcurve is needed (in days). In observer frame (z = 0).
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

        self._magnification_map = magnification_map
        self._time_duration_observer_frame = time_duration

        self._point_source_morphology = point_source_morphology
        self._kwargs_source_morphology = kwargs_source_morphology

        # Initialize the convolved map and source morphology
        self._convolved_map = None
        self._source_morphology = None

    # properties for convolved map and magnification map and time_duration_observer_frame
    @property
    def convolved_map(self):
        """Get the convolved map i.e., the magnification map convolved with the
        source morphology."""
        if self._convolved_map is None:
            raise ValueError(
                "Convolved map is not initialized. Please call get_convolved_map() first."
            )
        return self._convolved_map

    @property
    def magnification_map(self):
        """Get the magnification map."""
        return self._magnification_map

    @property
    def time_duration_observer_frame(self):
        """Get the lightcurve time duration in observer frame."""
        return self._time_duration_observer_frame

    def get_convolved_map(
        self,
        return_source_morphology=False,
    ):
        """Get the convolved map based on the source morphology.

        :param return_source_morphology: Whether to return the source
            morphology object or not. Default is False.
        :return: The convolved map and the source morphology object if
            requested. Otherwise, only the convolved map is returned.
        :rtype: numpy.ndarray or tuple
        """
        if self._point_source_morphology == "gaussian":
            # Gaussian source morphology
            source_morphology = GaussianSourceMorphology(
                **self._kwargs_source_morphology,  # sets the source size, redshift, and cosmology
                length_x=self._magnification_map.half_length_x * 2,
                length_y=self._magnification_map.half_length_y * 2,
                num_pix_x=self._magnification_map.num_pixels_x,
                num_pix_y=self._magnification_map.num_pixels_y,
                center_x=0,
                center_y=0,
            )

            # convolve the magnification map with the Gaussian kernel
            self._convolved_map = convolved_image(
                self._magnification_map.magnifications,
                source_morphology.kernel_map,
                convolution_type="fft",
            )

        elif self._point_source_morphology == "agn":
            # AGN source morphology
            source_morphology = AGNSourceMorphology(
                **self._kwargs_source_morphology,
            )
            cosmo = source_morphology.cosmo
            source_redshift = source_morphology.source_redshift

            # magnification map pixel size
            pixel_size_magnification_map = (
                self._magnification_map.get_pixel_size_meters(
                    source_redshift=source_redshift, cosmo=cosmo
                )
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
            self._convolved_map = convolved_image(
                self._magnification_map.magnifications,
                rescaled_kernel_map,
                convolution_type="fft",
            )

        elif self._point_source_morphology == "supernovae":
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
            self._source_morphology = source_morphology

        if return_source_morphology:
            return self._convolved_map, source_morphology
        else:
            return self._convolved_map

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
    ):
        """Generate lightcurves for a point source based on the convolved map.

        :param source_redshift: Redshift of the source
        :param cosmo: astropy.cosmology instance for the lens class
        :param lightcurve_type: Type of lightcurve to generate, either
            'magnitude' or 'magnification'. If 'magnitude', the
            lightcurve is returned in magnitudes normalized to the macro
            magnification. If 'magnification', the lightcurve is
            returned in magnification without normalization. Default is
            'magnitude'.
        :param effective_transverse_velocity: Transverse velocity in
            source plane (in km/s). Default is 1000 km/s (typical effective velocity of the source with respect to microlenses/stars).
        :param num_lightcurves: Number of lightcurves to generate.
            Default is 1.
        :param x_start_position: Starting x position of the lightcurve on the magnification map in arcsec. A value of 0 indicates the center of the magnification map. Default is None. If None, a random position is chosen.
        :param y_start_position: Starting y position of the lightcurve on the magnification map in arcsec. A value of 0 indicates the center of the magnification map. Default is None. If None, a random position is chosen.
        :param phi_travel_direction: Angle of the travel direction in
            degrees. Default is None. If None, a random angle is chosen. A value of 0
            implies the positive x-axis of the magnification map.
        :return: A tuple of lightcurves, tracks, and time arrays.

            lightcurves: list of lightcurves

            tracks: x and y positions (in pixels) on the magnification map grid for the paths used to generate the lightcurves.

            time_arrays: list of time arrays for each lightcurve
        """

        # Get the convolved magmap after convolving the magnification map with a Gaussian kernel
        convolved_map = self.get_convolved_map(
            return_source_morphology=False,
        )

        # determine physical pixel sizes in source plane
        pixel_size_magnification_map = self._magnification_map.get_pixel_size_meters(
            source_redshift=source_redshift, cosmo=cosmo
        )

        # convert x and y positions from arcsec to pixel coordinates on the magnification map grid
        if x_start_position is not None:
            x_start_position = (
                (x_start_position / self._magnification_map.half_length_x)
                * self._magnification_map.num_pixels_x
                / 2
            )
            x_start_position = int(
                x_start_position + self._magnification_map.num_pixels_x // 2
            )

        if y_start_position is not None:
            y_start_position = (
                (y_start_position / self._magnification_map.half_length_y)
                * self._magnification_map.num_pixels_y
                / 2
            )
            y_start_position = int(
                y_start_position + self._magnification_map.num_pixels_y // 2
            )

        return self._generate_lightcurves(
            source_redshift=source_redshift,
            convolved_map=convolved_map,
            pixel_size_magnification_map=pixel_size_magnification_map,
            num_lightcurves=num_lightcurves,
            lightcurve_type=lightcurve_type,
            effective_transverse_velocity=effective_transverse_velocity,
            x_start_position=x_start_position,
            y_start_position=y_start_position,
            phi_travel_direction=phi_travel_direction,
        )

    def _generate_lightcurves(
        self,
        source_redshift,
        convolved_map,
        pixel_size_magnification_map,
        num_lightcurves=1,
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        effective_transverse_velocity=1000,  # Transverse velocity in source plane (in km/s)
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
    ):
        """Generate lightcurves for a point source based on the convolved map.

        :param source_redshift: Redshift of the source
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
            in pixel coordinates. Default is None. If None, a random
            position is chosen. A value of 0 indicates the center of the
            magnification map.
        :param y_start_position: Starting y position of the lightcurve
            in pixel coordinates. Default is None. If None, a random
            position is chosen. A value of 0 indicates the center of the
            magnification map.
        :param phi_travel_direction: Angle of the travel direction in
            degrees. Default is None. If None, a random angle is chosen.
            A value of 0 implies the positive x-axis of the
            magnification map.
        :return: A tuple of lightcurves, tracks, and time arrays if
            requested.
        """

        LCs = []
        tracks = []
        time_arrays = []

        # time duration in source frame
        self._time_duration_source_frame = self._time_duration_observer_frame / (
            1 + source_redshift
        )

        time_duration_years = (
            self._time_duration_source_frame / 365.25
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
                    light_curve / np.abs(self._magnification_map.mu_ave)
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
            time_arrays.append(
                np.linspace(0, self._time_duration_observer_frame, len(light_curve))
            )

        return LCs, tracks, time_arrays
