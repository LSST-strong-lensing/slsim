__author__ = "Paras Sharma"

import numpy as np
from astropy import units as u


class SourceMorphology:
    """Base class for source morphologies. Handles static and time-varying
    sources, including vectorized interpolation for dynamic arrays."""

    def __init__(self, is_time_varying=False, user_snapshots=None, *args, **kwargs):
        """Initializes the base SourceMorphology class.

        :param is_time_varying: Boolean flag indicating if the source
            varies temporally.
        :param user_snapshots: Optional dictionary containing pre-
            computed snapshots for time-varying sources.
            Must contain:
            - 'times': 1D array of source-frame times (in days).
            - 'kernels': List or 3D array of 2D kernel maps normalized to 1.
            - 'pixel_scales_m': 1D array of pixel scales in meters corresponding to each kernel.
        """
        self.is_time_varying = is_time_varying
        self.user_snapshots = user_snapshots

        if self.user_snapshots is not None:
            self.is_time_varying = True
            self._prepare_snapshots()

    def _prepare_snapshots(self):
        """Pre-processes time-varying snapshots: sorts by time, pads all
        kernels to the maximum shape for consistency, and stacks them into a
        single 3D array for highly optimized vectorized interpolation."""
        sort_idx = np.argsort(self.user_snapshots["times"])
        self._anchor_times = np.array(self.user_snapshots["times"])[sort_idx]
        self._anchor_scales = np.array(self.user_snapshots["pixel_scales_m"])[sort_idx]
        raw_kernels = [self.user_snapshots["kernels"][i] for i in sort_idx]

        # Find maximum dimensions to safely stack expanding grids
        max_y = max(k.shape[0] for k in raw_kernels)
        max_x = max(k.shape[1] for k in raw_kernels)

        padded_kernels = []
        for k in raw_kernels:
            pad_y = max_y - k.shape[0]
            pad_x = max_x - k.shape[1]
            pad_y_top = pad_y // 2
            pad_x_left = pad_x // 2

            # Only pad if necessary to avoid overhead
            if pad_y > 0 or pad_x > 0:
                padded = np.pad(
                    k,
                    (
                        (pad_y_top, pad_y - pad_y_top),
                        (pad_x_left, pad_x - pad_x_left),
                    ),
                )
            else:
                padded = k
            padded_kernels.append(padded)

        # Stack into a 3D array: Shape (N_anchors, Y, X)
        self._anchor_kernels_3d = np.stack(padded_kernels)

    def _interpolate_snapshots(self, requested_times):
        """Highly optimized, fully vectorized engine to interpolate 2D grids
        and pixel scales across time.

        :param requested_times: 1D array of source-frame times (in days)
            to evaluate the kernels at.
        :return: A tuple containing:
            - List of 2D interpolated kernel maps.
            - List of interpolated pixel scales in meters.
        """
        t_req = np.asarray(requested_times)

        # 1. Clip requested times to avoid extrapolation out of bounds
        t_clipped = np.clip(t_req, self._anchor_times[0], self._anchor_times[-1])

        # 2. Vectorized 1D interpolation for pixel scales
        interpolated_scales = np.interp(
            t_clipped, self._anchor_times, self._anchor_scales
        )

        # 3. Vectorized 3D interpolation for the kernels
        # Find the left-bounding anchor index for every requested time
        idx = np.searchsorted(self._anchor_times, t_clipped)
        # Handle edge case where t == exactly the first anchor
        idx = np.clip(idx, 1, len(self._anchor_times) - 1)

        t0 = self._anchor_times[idx - 1]
        t1 = self._anchor_times[idx]

        # Temporal fraction (shape: N_requested)
        f = (t_clipped - t0) / (t1 - t0)

        # Broadcast fraction to 3D so it can multiply the (Y, X) grids: shape (N_requested, 1, 1)
        f_3d = f[:, np.newaxis, np.newaxis]

        # Extract the bounding 3D kernel blocks
        k0 = self._anchor_kernels_3d[idx - 1]
        k1 = self._anchor_kernels_3d[idx]

        # Fast vectorized linear interpolation
        interpolated_kernels_3d = k0 * (1.0 - f_3d) + k1 * f_3d

        # Vectorized normalization
        sums = np.nansum(interpolated_kernels_3d, axis=(1, 2), keepdims=True)
        sums[sums == 0] = 1.0  # Prevent division by zero
        interpolated_kernels_3d /= sums

        # Return as lists to interface perfectly with the existing lightcurve pipeline
        return list(interpolated_kernels_3d), list(interpolated_scales)

    def get_time_dependent_kernel_maps(self, time_anchors):
        """Returns a list of kernel maps and pixel scales for the requested
        times.

        If user_snapshots were provided during initialization, it uses
        the fast vectorized interpolator. Otherwise, it replicates the
        static kernel.
        """
        if self.user_snapshots is not None:
            return self._interpolate_snapshots(time_anchors)

        # Fallback for static sources unifying the pipeline API
        return [self.kernel_map for _ in time_anchors], [
            self.pixel_scale_m for _ in time_anchors
        ]

    def get_kernel_map(self, *args, **kwargs):
        """Returns the 2D array of the kernel map."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def kernel_map(self):
        """Returns the 2D array of the static kernel map."""
        if self.is_time_varying:
            raise AttributeError(
                "Time-varying sources do not have a single static kernel_map. "
                "Use get_time_dependent_kernel_maps() instead."
            )

        if not hasattr(self, "_kernel_map"):
            self._kernel_map = self.get_kernel_map()
        return self._kernel_map

    @property
    def length_x(self):
        """Returns the length of the 2D kernel map in x direction in
        arcseconds."""
        return self._length_x

    @property
    def length_y(self):
        """Returns the length of the 2D kernel map in y direction in
        arcseconds."""
        return self._length_y

    @property
    def num_pix_x(self):
        """Returns the number of pixels in x direction."""
        return self._num_pix_x

    @property
    def num_pix_y(self):
        """Returns the number of pixels in y direction."""
        return self._num_pix_y

    @property
    def pixel_scale_x(self):
        """Returns the pixel scale in x direction in arcseconds."""
        return self._pixel_scale_x

    @property
    def pixel_scale_y(self):
        """Returns the pixel scale in y direction in arcseconds."""
        return self._pixel_scale_y

    @property
    def pixel_scale(self):
        """Returns the geometric mean pixel scale in arcseconds."""
        if not hasattr(self, "_pixel_scale"):
            if hasattr(self, "_pixel_scale_x") and hasattr(self, "_pixel_scale_y"):
                self._pixel_scale = np.sqrt(self._pixel_scale_x * self._pixel_scale_y)
            else:
                raise AttributeError("Pixel scale not defined.")
        return self._pixel_scale

    @property
    def pixel_scale_x_m(self):
        """Returns the pixel scale in x direction in meters."""
        if not hasattr(self, "_pixel_scale_x_m"):
            self._pixel_scale_x_m = self.arcsecs_to_metres(
                self.pixel_scale_x, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_x_m

    @property
    def pixel_scale_y_m(self):
        """Returns the pixel scale in y direction in meters."""
        if not hasattr(self, "_pixel_scale_y_m"):
            self._pixel_scale_y_m = self.arcsecs_to_metres(
                self.pixel_scale_y, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_y_m

    @property
    def pixel_scale_m(self):
        """Returns the geometric mean pixel scale in meters."""
        if not hasattr(self, "_pixel_scale_m"):
            self._pixel_scale_m = self.arcsecs_to_metres(
                self.pixel_scale, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_m

    def arcsecs_to_metres(self, arcsecs, cosmo, redshift):
        """Converts arcseconds to meters in the source plane, given the
        cosmology and redshift.

        :param arcsecs: Arcseconds to be converted.
        :param cosmo: Astropy cosmology object for angle calculations.
        :param redshift: Redshift of the source.
        :return: Transverse distance in meters in the source plane at
            the given redshift.
        """
        # Convert arcseconds to radians
        radians = arcsecs * u.arcsec.to(u.rad)
        # Calculate the angular diameter distance in meters
        angular_diameter_distance = (
            cosmo.angular_diameter_distance(redshift).to(u.m)
        ).value
        # Calculate the transverse distance in meters
        transverse_distance = angular_diameter_distance * radians
        return transverse_distance

    def metres_to_arcsecs(self, metres, cosmo, redshift):
        """Converts meters to arcseconds in the source plane, given the
        cosmology and redshift.

        :param metres: Meters to be converted.
        :param cosmo: Astropy cosmology object for angle calculations.
        :param redshift: Redshift of the source.
        :return: Arcseconds in the source plane at the given redshift.
        """
        # Calculate the angular diameter distance in meters
        angular_diameter_distance = (
            cosmo.angular_diameter_distance(redshift).to(u.m).value
        )
        # Calculate the arcseconds in the source plane
        arcsecs = (metres / angular_diameter_distance) * u.rad.to(u.arcsec)  # .value
        return arcsecs