__author__ = "Paras Sharma"

import numpy as np
from skimage.transform import rescale
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

from slsim.Microlensing.magmap import MagnificationMap
from slsim.Util.astro_util import extract_light_curve

from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology
from slsim.Microlensing.source_morphology.gaussian import GaussianSourceMorphology
from slsim.Microlensing.source_morphology.supernovae import SupernovaeSourceMorphology

# Central routing dictionary for source morphology classes
MORPHOLOGY_CLASSES = {
    "gaussian": GaussianSourceMorphology,
    "agn": AGNSourceMorphology,
    "supernovae": SupernovaeSourceMorphology,
}


class MicrolensingLightCurve(object):
    """Class to generate microlensing lightcurve(s) for a single source based
    on the magnification map, source morphology, and lens properties."""

    def __init__(
        self,
        magnification_map: MagnificationMap,
        observation_time_array: np.ndarray,
        point_source_morphology: str = "gaussian",
        kwargs_source_morphology: dict = {},
        source_morphology_instance=None,
    ):
        """
        :param magnification_map: MagnificationMap object, if not provided.
        :param observation_time_array: Array of observation times for which the lightcurve is needed (in days). In observer frame (z = 0).
        :param point_source_morphology: Type of source morphology to use. Default is 'gaussian'. Options are 'gaussian' or 'agn' (Accretion Disk) or 'supernovae'.
        :param kwargs_source_morphology: Dictionary of keyword arguments for the source morphology class. This should be as per the source morphology type.

            For example, for Gaussian source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmo, "source_size": source_size, }.

            For AGN source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmology,
            "r_out": r_out, "r_resolution": r_resolution, "smbh_mass_exp": smbh_mass_exp, "inclination_angle": inclination_angle,
            "black_hole_spin": black_hole_spin, "observer_frame_wavelength_in_nm": observer_frame_wavelength_in_nm,
            "eddington_ratio": eddington_ratio, }.
        :param source_morphology_instance: Optional pre-instantiated source morphology object. If provided, this will be used directly instead of instantiating a new one based on the point_source_morphology and kwargs_source_morphology. This allows for more complex use cases where the source morphology needs to be shared or customized beyond the standard options.
        """

        self._magnification_map = magnification_map
        self._observation_time_array = observation_time_array
        self._time_duration_observer_frame = (
            self._observation_time_array[-1] - self._observation_time_array[0]
        )

        self._point_source_morphology = point_source_morphology
        self._kwargs_source_morphology = kwargs_source_morphology

        # Use the global instance if provided, otherwise build locally (for backwards compatibility)
        if source_morphology_instance is not None:
            self._source_morphology = source_morphology_instance
        else:
            # Instantiate the morphology class
            self._setup_source_morphology()

    def _setup_source_morphology(self):
        """Instantiates the specified source morphology class and caches it."""
        morph_class = MORPHOLOGY_CLASSES.get(self._point_source_morphology)
        if morph_class is None:
            raise ValueError(
                f"Invalid source morphology type: '{self._point_source_morphology}'. "
                f"Available options are: {list(MORPHOLOGY_CLASSES.keys())}"
            )

        if self._point_source_morphology == "gaussian":
            self._source_morphology = morph_class(
                **self._kwargs_source_morphology,
                length_x=self._magnification_map.half_length_x * 2,
                length_y=self._magnification_map.half_length_y * 2,
                num_pix_x=self._magnification_map.num_pixels_x,
                num_pix_y=self._magnification_map.num_pixels_y,
                center_x=0,
                center_y=0,
            )
        else:
            self._source_morphology = morph_class(**self._kwargs_source_morphology)

    @property
    def magnification_map(self):
        """Get the magnification map used for this lightcurve generation.
        :return: MagnificationMap object (see slsim.Microlensing.magmap.MagnificationMap).
        :rtype: MagnificationMap
        """
        return self._magnification_map

    @property
    def time_duration_observer_frame(self):
        """Get the total time duration covered by the observation time array in the observer frame (in days).
        :return: Time duration in observer frame (in days).
        :rtype: float
        """
        return self._time_duration_observer_frame

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

        # Handle time duration first as we need it for SNe anchor calculation
        self._time_duration_source_frame = self._time_duration_observer_frame / (
            1 + source_redshift
        )

        pixel_size_magnification_map = self._magnification_map.get_pixel_size_meters(
            source_redshift=source_redshift, cosmo=cosmo
        )

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
        pixel_size_magnification_map,
        num_lightcurves=1,
        lightcurve_type="magnitude",
        effective_transverse_velocity=1000,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
    ):
        """Generate lightcurves for a point source based on the convolved
        map/cube.

        :param source_redshift: Redshift of the source
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
        time_elapsed_source_years = (
            (self._observation_time_array - self._observation_time_array[0])
            / (1 + source_redshift)
            / 365.25
        )

        for _ in range(num_lightcurves):
            # 1. Extract the raw spatial track from the magnification map
            raw_light_curve, x_positions, y_positions = extract_light_curve(
                convolution_array=self._magnification_map.magnifications,
                pixel_size=pixel_size_magnification_map,
                effective_transverse_velocity=effective_transverse_velocity,
                light_curve_time_in_years=time_elapsed_source_years,
                pixel_shift=0,
                x_start_position=x_start_position,
                y_start_position=y_start_position,
                phi_travel_direction=phi_travel_direction,
                return_track_coords=True,
                random_seed=None,
            )

            n_steps = len(x_positions)

            # Anchor the times to the absolute start and end!
            # Since extract_light_curve handled the time mapping, actual_times_source is perfectly mapped:
            actual_times_observer = np.linspace(
                self._observation_time_array[0],
                self._observation_time_array[-1],
                n_steps,
            )
            actual_times_source = actual_times_observer / (1 + source_redshift)
            light_curve = np.zeros(n_steps)

            # ==========================================================
            # TIME VARYING SOURCES (SUPERNOVAE)
            # ==========================================================
            if self._source_morphology.is_time_varying:
                kernels, pixel_scales_m = (
                    self._source_morphology.get_time_dependent_kernel_maps(
                        actual_times_source
                    )
                )

                max_pad = 0
                rescaled_kernels = []
                for kernel, kernel_pixel_size_m in zip(kernels, pixel_scales_m):
                    pixel_ratio = kernel_pixel_size_m / pixel_size_magnification_map
                    if pixel_ratio * kernel.shape[0] < 1.0:
                        res_k = np.array([[1.0]])
                    else:
                        res_k = rescale(
                            kernel,
                            pixel_ratio,
                            anti_aliasing=False,
                            mode="constant",
                            cval=0.0,
                        )

                    if np.nansum(res_k) > 0:
                        res_k /= np.nansum(res_k)

                    rescaled_kernels.append(res_k)
                    # +4 to leave room for the 4x4 bicubic extraction window
                    max_pad = max(
                        max_pad, res_k.shape[0] // 2 + 4, res_k.shape[1] // 2 + 4
                    )

                padded_mag_map = np.pad(
                    self._magnification_map.magnifications, max_pad, mode="reflect"
                )

                for i in range(n_steps):
                    res_k = rescaled_kernels[i]
                    ky, kx = res_k.shape

                    # Use scipy's native kernel-center convention (fractional allowed)
                    cy_k = (ky - 1) / 2.0
                    cx_k = (kx - 1) / 2.0

                    # Get exact sub-pixel coordinates
                    px = x_positions[i] + max_pad
                    py = y_positions[i] + max_pad

                    # We need a 4x4 valid-convolution grid surrounding (py, px).
                    # fftconvolve(..., mode='valid')[a, b] corresponds to the
                    # convolution evaluated at input position
                    #     (stamp_y_start + cy_k + a, stamp_x_start + cx_k + b).
                    # Pick stamp_y_start so that (py, px) lands in the interior
                    # cell [1, 2) of the 4x4 output grid (ideal for bicubic).
                    stamp_y_start = int(np.floor(py - cy_k)) - 1
                    stamp_x_start = int(np.floor(px - cx_k)) - 1

                    stamp = padded_mag_map[
                        stamp_y_start : stamp_y_start + ky + 3,
                        stamp_x_start : stamp_x_start + kx + 3,
                    ]
                    conv_4x4 = fftconvolve(stamp, res_k, mode="valid")  # (4, 4)

                    # Local coords inside conv_4x4 corresponding to (py, px)
                    local_y = py - stamp_y_start - cy_k  # always in [1, 2)
                    local_x = px - stamp_x_start - cx_k  # always in [1, 2)

                    light_curve[i] = map_coordinates(
                        conv_4x4, [[local_y], [local_x]], order=3
                    )[0]

            # ==========================================================
            # STATIC SOURCES (STATIC AGN, GAUSSIAN)
            # ==========================================================
            else:
                kernel, pixel_scale_m = (
                    self._source_morphology.kernel_map,
                    self._source_morphology.pixel_scale_m,
                )
                pixel_ratio = pixel_scale_m / pixel_size_magnification_map

                if pixel_ratio * kernel.shape[0] < 1.0:
                    res_k = np.array([[1.0]])
                else:
                    res_k = rescale(
                        kernel,
                        pixel_ratio,
                        anti_aliasing=False,
                        mode="constant",
                        cval=0.0,
                    )
                if np.nansum(res_k) > 0:
                    res_k /= np.nansum(res_k)

                max_pad = max(res_k.shape[0] // 2 + 1, res_k.shape[1] // 2 + 1)
                padded_mag_map = np.pad(
                    self._magnification_map.magnifications, max_pad, mode="reflect"
                )

                # 1. Convolve the entire padded map with the static source kernel once
                convolved_padded_map = fftconvolve(padded_mag_map, res_k, mode="same")

                # 2. Shift the continuous track coordinates to account for the padding
                coords = np.vstack((y_positions + max_pad, x_positions + max_pad))

                # 3. Extract the light curve values at the track coordinates using bicubic interpolation (order=3)
                light_curve = map_coordinates(convolved_padded_map, coords, order=3)

            # Convert to Magnitude if required
            if lightcurve_type == "magnitude":
                light_curve = -2.5 * np.log10(
                    light_curve / np.abs(self._magnification_map.mu_ave)
                )
            elif lightcurve_type != "magnification":
                raise ValueError(
                    "Lightcurve type not recognized. Please use 'magnitude' or 'magnification'."
                )

            LCs.append(light_curve)
            tracks.append(np.array([x_positions, y_positions]))
            time_arrays.append(actual_times_observer)

        return LCs, tracks, time_arrays
