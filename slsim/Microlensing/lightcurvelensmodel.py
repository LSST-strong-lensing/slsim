__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented

import numpy as np
import astropy.constants as const
from astropy import units as u

from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.lightcurve import MicrolensingLightCurve


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