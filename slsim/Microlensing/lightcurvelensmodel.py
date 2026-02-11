__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented

import numpy as np

# import astropy.constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from slsim.Util.astro_util import theta_star_physical

from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.lightcurve import MicrolensingLightCurve


class MicrolensingLightCurveFromLensModel(object):
    """Class to generate microlensing lightcurves based on the microlensing
    parameters for each image of a source."""

    def __init__(
        self,
        source_redshift,
        deflector_redshift,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        shear_phi_angle_images,
        ra_lens,
        dec_lens,
        deflector_velocity_dispersion,
        cosmology,
        kwargs_magnification_map=None,
        point_source_morphology=None,
        kwargs_source_morphology=None,
    ):
        """Initialize the MicrolensingLightCurveFromLensModel with lens and
        source parameters.

        :param source_redshift: Redshift of the source
        :param deflector_redshift: Redshift of the deflector
        :param kappa_star_images: list containing the kappa star
            (stellar convergence) for each image of the source.
        :param kappa_tot_images: list containing the kappa total (total
            convergence) for each image of the source.
        :param shear_images: list containing the shear for each image of
            the source.
        :param shear_phi_angle_images: list containing the angle of the
            shear vector, w.r.t. the x-axis of the image plane, in
            degrees for each image of the source.
        :param ra_lens: Right Ascension of the lens in degrees.
        :param dec_lens: Declination of the lens in degrees.
        :param deflector_velocity_dispersion: Velocity dispersion of the
            deflector in km/s.
        :param cosmology: Astropy cosmology object to use for the
            calculations.
        :param kwargs_magnification_map: Keyword arguments for the
            MagnificationMap class. An example can look like:
            kwargs_magnification_map = { "theta_star": theta_star, #
            arcsec "rectangular": True, "center_x": 0, # arcsec
            "center_y": 0, # arcsec "half_length_x": 25 * theta_star, #
            arcsec "half_length_y": 25 * theta_star, # arcsec
            "mass_function": "kroupa", "m_solar": 1.0, "m_lower": 0.08,
            "m_upper": 100, "num_pixels_x": 500, "num_pixels_y": 500, }
            Note that theta_star needs be estimated based on the
            cosmology model and redshifts for the source and deflector.
        :param point_source_morphology: Morphology of the point source.
            Options are "gaussian", "agn" (Accretion Disk) or
            "supernovae".
        :param kwargs_source_morphology: Dictionary of keyword arguments
            for the source morphology class. (See
            slsim.Microlensing.source_morphology for more details) Note
            that different parameters are defined for different source
            morphologies. So check the documentation for each
            morphology. For example, for Gaussian source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift":
            source_redshift, "cosmo": cosmo, "source_size": source_size,
            }. For AGN source morphology, it will look like:
            kwargs_source_morphology = {"source_redshift":
            source_redshift, "cosmo": cosmology, "r_out": r_out,
            "r_resolution": r_resolution, "smbh_mass_exp":
            smbh_mass_exp, "inclination_angle": inclination_angle,
            "black_hole_spin": black_hole_spin,
            "observer_frame_wavelength_in_nm":
            observer_frame_wavelength_in_nm, "eddington_ratio":
            eddington_ratio, }
        """
        self._source_redshift = source_redshift
        self._deflector_redshift = deflector_redshift
        self._kappa_star_images = kappa_star_images
        self._kappa_tot_images = kappa_tot_images
        self._shear_images = shear_images
        self._shear_phi_angle_images = shear_phi_angle_images
        self._ra_lens = ra_lens
        self._dec_lens = dec_lens
        self._deflector_velocity_dispersion = deflector_velocity_dispersion
        self._cosmology = cosmology
        self._kwargs_magnification_map = kwargs_magnification_map
        self._point_source_morphology = point_source_morphology
        self._kwargs_source_morphology = kwargs_source_morphology

        if self._kwargs_magnification_map is None:
            self._kwargs_magnification_map = (
                self._get_default_kwargs_magnification_map()
            )
            print(
                "kwargs_magnification_map not in kwargs_microlensing. Using default magnification map kwargs."
            )

        if self._point_source_morphology is None:
            raise ValueError(
                "point_source_morphology not in kwargs_microlensing. Please provide the point source morphology type. It can be either 'gaussian' or 'agn' or 'supernovae'."
            )
        if self._kwargs_source_morphology is None:
            raise ValueError(
                "kwargs_source_morphology not in kwargs_microlensing. Please provide a dictionary of settings required by source morphology calculation."
            )

    def generate_point_source_microlensing_magnitudes(
        self,
        time,
    ):
        """Generate microlensing lightcurve magnitudes normalized to the mean
        magnification for various source morphologies. For single source only,
        it produces the lightcurve magnitudes for all images of the source.

        :param time: Time array for which the lightcurve is needed (in
            days).
        :return: lightcurves_single: numpy array of microlensing
            magnitudes with the shape (num_images, len(time)).
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
                "Time array not provided in the correct format. Supported formats are int, float, array, and list."
            )

        lightcurves, self._tracks, __time_arrays = (
            self.generate_point_source_lightcurves(
                time_array,
                lightcurve_type="magnitude",
                num_lightcurves=1,
            )
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

        self._lightcurves = lightcurves_single  # cache the lightcurves

        return lightcurves_single

    @property
    def lightcurves(self):
        """Returns the cached lightcurves generated for the point source.

        Has shape (num_images, len(time)).
        """
        if hasattr(self, "_lightcurves"):
            return self._lightcurves
        else:
            raise AttributeError(
                "Lightcurves are not set. Please run generate_point_source_microlensing_magnitudes first."
            )

    @property
    def tracks(self):
        """Returns the cached track coordinates on the magnification map,
        generated for the point source."""
        if hasattr(self, "_tracks"):
            return self._tracks
        else:
            raise AttributeError(
                "Tracks are not set. Please run generate_point_source_microlensing_magnitudes first."
            )

    def generate_point_source_lightcurves(
        self,
        time,
        lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
        num_lightcurves=1,  # Number of lightcurves to generate
    ):
        """Generate lightcurves for one single point source with certain size,
        but for all images of that source based on the lens model. The point
        source is simulated as a "gaussian", "agn" (Accretion Disk) or
        "supernovae".

        The lightcurves are generated based on the microlensing map
        convolved with the source morphology kernel.

        The generated lightcurves will have the same length of time as
        the "time" array provided.

        :param time: Time array for which the lightcurve is needed.
        :param lightcurve_type: Type of lightcurve to generate, either
            'magnitude' or 'magnification'. If 'magnitude', the
            lightcurve is returned in magnitudes normalized to the macro
            magnification. If 'magnification', the lightcurve is
            returned in magnification without normalization. Default is
            'magnitude'.
        :param num_lightcurves: Default is 1. If require multiple lightcurves for each image using the same magnification map, set
            this parameter to the number of lightcurves required.

        :return:

        lightcurves: numpy array of microlensing magnitudes
            with the shape (num_images, num_lightcurves, len(time)). The first dimension
            is the number of images of the source and the second
            dimension is the length of the time array.

        tracks: list of tracks for each image of the source.
            Each track is a list of tuples with the x and y positions
            of the source at each time step.

        time_arrays: list of time arrays for each image of the
            source. Each time array is a numpy array with the same
            length as the time array provided.

        :rtype: tuple
        """

        # generate magnification maps for each image of the source if they are not already generated and cached
        magmaps_images = self.generate_magnification_maps_from_microlensing_params()

        if (isinstance(time, np.ndarray) or isinstance(time, list)) and len(time) > 1:
            lightcurve_duration = time[-1] - time[0]
        elif (isinstance(time, np.ndarray) or isinstance(time, list)) and len(
            time
        ) == 1:
            lightcurve_duration = 0
        else:
            raise ValueError(
                "Time array not provided in the correct format. Supported formats are int, float, array, and list."
            )

        # obtain velocities and angles for each image
        eff_trv_vel_images, eff_trv_vel_angles_images = (
            self.effective_transverse_velocity_images
        )

        # obtain lightcurve starting position
        x_start_position, y_start_position = self.lc_start_position

        # generate lightcurves for each image of the source
        lightcurves = (
            []
        )  # a list which contains the [list of lightcurves] for each image of the source, depending on the num_lightcurves parameter.
        tracks = (
            []
        )  # a list which contains the [list of tracks] for each image of the source, depending on the num_lightcurves parameter.
        time_arrays = []  # corresponding to each lightcurve
        for i in range(len(self._kappa_star_images)):
            ml_lc = MicrolensingLightCurve(
                magnification_map=magmaps_images[i],
                time_duration=lightcurve_duration,
                point_source_morphology=self._point_source_morphology,
                kwargs_source_morphology=self._kwargs_source_morphology,
            )
            curr_lightcurves, curr_tracks, curr_time_arrays = (
                ml_lc.generate_lightcurves(
                    source_redshift=self._source_redshift,
                    cosmo=self._cosmology,
                    lightcurve_type=lightcurve_type,
                    effective_transverse_velocity=eff_trv_vel_images[i],
                    num_lightcurves=num_lightcurves,
                    x_start_position=x_start_position,
                    y_start_position=y_start_position,
                    phi_travel_direction=eff_trv_vel_angles_images[i],
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
    ):
        """Generate magnification maps for each image of the source based on
        the image positions and the lens model.

        Returns:
        magmaps_images: a list which contains the [magnification map for each image of the source].
        """
        # check if magnification maps are already generated
        if hasattr(self, "_magmaps_images"):
            return self._magmaps_images

        # generate magnification maps for each image of the source
        self._magmaps_images = []
        for i in range(len(self._kappa_star_images)):
            # generate magnification maps for each image of the source
            magmap = MagnificationMap(
                kappa_tot=self._kappa_tot_images[i],
                shear=self._shear_images[i],
                kappa_star=self._kappa_star_images[i],
                **self._kwargs_magnification_map,
            )
            self._magmaps_images.append(magmap)

        return self._magmaps_images

    @property
    def magmaps_images(self):
        """Returns the magnification maps for each image of the source."""
        if hasattr(self, "_magmaps_images"):
            return self._magmaps_images
        else:
            raise AttributeError(
                "Magnification maps are not set. Please run generate_magnification_maps_from_microlensing_params first."
            )

    def _interpolate_light_curve(self, lightcurve, time_array, time_array_new):
        """Interpolate the lightcurve to a new time array.

        :param lightcurve: Lightcurve to be interpolated.
        :param time_array: Original time array of the lightcurve.
        :param time_array_new: New time array to interpolate the
            lightcurve to.
        :return: Interpolated lightcurve.
        :rtype: numpy array
        """
        return np.interp(time_array_new, time_array, lightcurve)

    @property
    def effective_transverse_velocity_images(self):
        """Returns the effective transverse velocity in the source plane for
        each image position in the frame of the magnification map by using
        appropriate transformations. Once calculated, the values are cached for
        future use.

        :return: effective_velocities: list containing the effective
            transverse velocity in km/s for each image of the source.
        :return: effective_velocities_angles_deg: list containing the
            angle of the effective transverse velocity in degrees for
            each image of the source.
        :rtype: tuple
        """
        if hasattr(self, "_eff_trv_vel_images"):
            return self._eff_trv_vel_images
        else:
            self._eff_trv_vel_images = self._effective_transverse_velocity_images()
            return self._eff_trv_vel_images

    def _effective_transverse_velocity_images(
        self,
        random_seed=None,
        magmap_reference_frame=True,
    ):
        """Calculate the effective transverse velocity in the source plane for
        each image position. Eventually return the effective transverse
        velocity in frame of the magnification map by using appropriate
        transformations.

        This implementation is based on the works in the following papers [Credits (for suggestions): James Hung-Hsu Chan, Luke Weisenbach and Henry Best]:
        1. https://arxiv.org/pdf/2004.13189
        2. https://iopscience.iop.org/article/10.1088/0004-637X/712/1/658/pdf
        3. https://iopscience.iop.org/article/10.3847/0004-637X/832/1/46/pdf

        :param random_seed: Random seed for reproducibility. If None, a random seed will be generated.

        :return: effective_velocities: list containing the effective transverse velocity in km/s for each image of the source.
        :return: effective_velocities_angles_deg: list containing the angle of the effective transverse velocity in degrees for each image of the source.
        :rtype: tuple
        """

        # --- Lens Model Inputs ---
        z_s = self._source_redshift
        z_l = self._deflector_redshift

        if not isinstance(self._ra_lens, u.Quantity):
            ra_l = self._ra_lens * u.deg
        else:
            ra_l = self._ra_lens

        if not isinstance(self._dec_lens, u.Quantity):
            dec_l = self._dec_lens * u.deg
        else:
            dec_l = self._dec_lens

        # σ⋆
        if not isinstance(self._deflector_velocity_dispersion, u.Quantity):
            sig_star = self._deflector_velocity_dispersion * u.km / u.s
        else:
            sig_star = self._deflector_velocity_dispersion

        np.random.seed(random_seed)  # Set the random seed for reproducibility

        #############################################
        # Lightman & Schechter 1990, Hamilton 2001
        # f = Omega_m**(4./7.) + Omega_v*(1.+Omega_m/2.)/70.
        #############################################
        def f_GrowthRate(z):
            Omega_m = self._cosmology.Om(z)
            Omega_v = self._cosmology.Ode(z)
            return Omega_m ** (4.0 / 7.0) + Omega_v * (1.0 + Omega_m / 2.0) / 70.0

        #############################################

        #############################################
        # Kochanek04
        # sigma0 = 235 km/s
        #############################################
        sigma0 = 235 * (u.km / u.s)
        sig_l_pec = (
            sigma0 / (1 + z_l) ** 0.5 * f_GrowthRate(z_l) / f_GrowthRate(0)
        )  # σₗ,pec
        sig_s_pec = (
            sigma0 / (1 + z_s) ** 0.5 * f_GrowthRate(z_s) / f_GrowthRate(0)
        )  # σₛ,pec
        #############################################

        # angular‐diameter distances
        D_l = self._cosmology.angular_diameter_distance(z_l)
        D_s = self._cosmology.angular_diameter_distance(z_s)
        D_ls = self._cosmology.angular_diameter_distance_z1z2(z_l, z_s)

        # effective combined pec.‐velocity dispersion σ_g (Eq.5)
        sigma_g = np.sqrt(
            (sig_l_pec / (1 + z_l) * D_s / D_l) ** 2 + (sig_s_pec / (1 + z_s)) ** 2
        )

        # CMB dipole from Planck (2018)
        v_dipole = 369.8 * u.km / u.s
        dipole_apex = SkyCoord(ra=167.942 * u.deg, dec=-6.944 * u.deg, frame="icrs")
        u_dipole = dipole_apex.cartesian.xyz / dipole_apex.cartesian.norm()

        # line‐of‐sight unit vector to lens
        los = SkyCoord(ra=ra_l, dec=dec_l, frame="icrs")
        u_los = los.cartesian.xyz / los.cartesian.norm()

        # 1) observer’s transverse velocity (Eq.6)
        v_cmb_vec = v_dipole * u_dipole
        proj_along = np.dot(u_los.value, v_cmb_vec.value) * v_cmb_vec.unit
        v_o_vec = v_cmb_vec - proj_along * u_los  # already in km/s
        v_o_scaled = v_o_vec * (D_ls / D_l) / (1 + z_l)

        # 2) build two orthonormal basis vectors e1,e2 ⟂ u_los
        # (just any pair spanning the sky‐plane)
        e1 = np.cross(u_los.value, [0, 0, 1.0])
        if np.allclose(e1, 0):
            e1 = np.cross(u_los.value, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_los.value, e1)
        e2 /= np.linalg.norm(e2)

        # 3) combined random “Gaussian” component v_g
        ϕ = np.random.uniform(0, 2 * np.pi)
        v_g_mag = np.random.normal(0, sigma_g.value) * sigma_g.unit
        v_g_vec = (np.cos(ϕ) * e1 + np.sin(ϕ) * e2) * v_g_mag

        # 4) sum to get the effective transverse velocity v_e except the stellar part
        v_e_vec_no_star = v_o_scaled + v_g_vec  # (Eq.7), Same for all images

        effective_velocities = []  # km/s
        effective_velocities_angles_deg = []  # degrees

        for i in range(len(self._shear_phi_angle_images)):
            # 5) lens‐galaxy peculiar velocity v_* (Eq.4)
            θ = np.random.uniform(0, 2 * np.pi)
            v_star_mag = np.sqrt(2) * sig_star  # (Eq.3)
            v_star_vec = (
                (np.cos(θ) * e1 + np.sin(θ) * e2) * v_star_mag.value * v_star_mag.unit
            )
            v_star_scaled = v_star_vec * (D_s / D_l) / (1 + z_l)

            v_e_vec = v_e_vec_no_star - v_star_scaled  # (Eq.7)

            # 6) project onto sky-plane axes
            v_e_x = np.dot(v_e_vec.value, e1) * v_e_vec.unit  # component along e1
            v_e_y = np.dot(v_e_vec.value, e2) * v_e_vec.unit  # component along e2
            v_e_2d = np.array([v_e_x.value, v_e_y.value]) * v_e_x.unit

            # 7) Magnitude and angle of the effective velocity in the source plane
            v_e_mag = np.linalg.norm(v_e_2d.value)
            v_e_angle = np.arctan2(
                v_e_y.value, v_e_x.value
            )  # angle in radians, with respect to x axis in physical plane
            v_e_angle_deg = np.degrees(v_e_angle)  # convert to degrees

            effective_velocities.append(v_e_mag)

            # assuming the shear vector is in the x-direction of the magnification map
            # the returned angle is with respect to the x-axis of the magnification map
            if magmap_reference_frame:
                # convert the angle to the reference frame of the magnification map
                v_e_angle_deg = v_e_angle_deg - self._shear_phi_angle_images[i]
            effective_velocities_angles_deg.append(v_e_angle_deg)

        return (
            np.array(effective_velocities),
            np.array(effective_velocities_angles_deg),
        )

    def _get_default_kwargs_magnification_map(self, mean_microlens_mass=1):
        """Returns the default kwargs for the magnification map based on the
        source and deflector redshifts.

        Parameters
        ----------
        mean_microlens_mass : float
            The mean mass of the microlenses

        Returns
        -------
        dict
            The default kwargs for the magnification map
        """
        theta_star_arcsec, _, _ = theta_star_physical(
            z_lens=self._deflector_redshift,
            z_src=self._source_redshift,
            cosmo=self._cosmology,
            m=mean_microlens_mass,
        )

        theta_star_arcsec = theta_star_arcsec.to(u.arcsec).value

        return {
            "theta_star": theta_star_arcsec,  # arcsec
            "center_x": 0,  # arcsec
            "center_y": 0,  # arcsec
            "half_length_x": 2 * theta_star_arcsec,  # arcsec
            "half_length_y": 2 * theta_star_arcsec,  # arcsec
            "mass_function": "kroupa",
            "m_solar": 1.0,
            "m_lower": 0.08,
            "m_upper": 100,
            "num_pixels_x": 500,
            "num_pixels_y": 500,
        }

    def update_source_morphology(self, kwargs_source_morphology):
        """Updates the source morphology parameters (e.g., for a new band)
        without requiring re-initialization of the class or re-generation of
        magnification maps."""
        self._kwargs_source_morphology = kwargs_source_morphology

    @property
    def lc_start_position(self):
        """Chooses a random starting position for the lightcurve track on the
        magnification map. Once set, the starting position remains fixed for
        subsequent calls.

        :return: x_start_position: x-coordinate of the starting position
            on the magnification map (in arcsec).
        :return: y_start_position: y-coordinate of the starting position
            on the magnification map (in arcsec).
        :rtype: tuple
        """
        if hasattr(self, "_lc_start_position"):
            return self._lc_start_position
        else:
            half_length_x = self._kwargs_magnification_map["half_length_x"]
            half_length_y = self._kwargs_magnification_map["half_length_y"]

            x_start_position = np.random.uniform(
                -half_length_x,
                half_length_x,
            )
            y_start_position = np.random.uniform(
                -half_length_y,
                half_length_y,
            )
            self._lc_start_position = (x_start_position, y_start_position)
            return self._lc_start_position
