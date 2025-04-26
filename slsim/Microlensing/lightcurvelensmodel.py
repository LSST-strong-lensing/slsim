__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented

import numpy as np
# import astropy.constants as const
from astropy import units as u
from astropy.coordinates import Galactic,ICRS

from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.lightcurve import MicrolensingLightCurve

class MicrolensingLightCurveFromLensModel(object):
    """Class to generate microlensing lightcurves based on the microlensing
    parameters for each image of a source."""

    def generate_point_source_microlensing_magnitudes(
        self,
        time,
        source_redshift,
        kappa_star_images,
        kappa_tot_images,
        shear_images,
        cosmology,
        kwargs_MagnificationMap:dict,
        point_source_morphology:str,
        kwargs_source_morphology:dict,
    ):
        """Generate lightcurve magnitudes normalized to the mean magnification
        for gaussian point sources. For single source only, it produces the
        lightcurve magnitudes for all images of the source.

        Returns a numpy array of microlensing magnitudes (does not
        include macro-magnifications) with the shape (num_images,
        len(time)).

        :param time: Time array for which the lightcurve is needed.
        :param source_redshift: Redshift of the source
        :param kappa_star_images: list containing the kappa star for each image of the source.
        :param kappa_tot_images: list containing the kappa total for each image of the source.
        :param shear_images: list containing the shear for each image of the source.
        :param cosmology: Astropy cosmology object to use for the calculations.
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class. This can look like:
            
            kwargs_MagnificationMap = {
                "theta_star": theta_star,  # arcsec
                "rectangular": True,
                "center_x": 0,  # arcsec
                "center_y": 0,  # arcsec
                "half_length_x": 25 * theta_star,  # arcsec
                "half_length_y": 25 * theta_star,  # arcsec
                "mass_function": "kroupa",
                "m_solar": 1.0,
                "m_lower": 0.08,
                "m_upper": 100,
                "num_pixels_x": 500,
                "num_pixels_y": 500,
            }

            Note that theta_star needs be estimated based on the cosmology model and redshifts for the source and deflector.

        :param point_source_morphology: Morphology of the point source. Options are "gaussian", "agn" (Accretion Disk) or "supernovae".
        :param kwargs_source_morphology: Dictionary of keyword arguments for the source morphology class. (See slsim.Microlensing.source_morphology for more details)
            
            For example, for Gaussian source morphology, it will look like: kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmo, "source_size": source_size, }.
            
            For AGN source morphology, it will look like: kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmology, "r_out": r_out, "r_resolution": r_resolution, "smbh_mass_exp": smbh_mass_exp, "inclination_angle": inclination_angle, "black_hole_spin": black_hole_spin, "observer_frame_wavelength_in_nm": observer_frame_wavelength_in_nm, "eddington_ratio": eddington_ratio, }
        
        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'magnification'. If 'magnitude', the lightcurve is returned in magnitudes normalized to the macro magnification.
                                If 'magnification', the lightcurve is returned in magnification without normalization. Default is 'magnitude'.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.

        :return: lightcurves_single: numpy array of microlensing magnitudes with the shape (num_images, len(time)).
            

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

        lightcurves, __tracks, __time_arrays = self.generate_point_source_lightcurves(
            time_array,
            source_redshift,
            kappa_star_images,
            kappa_tot_images,
            shear_images,
            cosmology,
            kwargs_MagnificationMap=kwargs_MagnificationMap,
            point_source_morphology=point_source_morphology,
            kwargs_source_morphology=kwargs_source_morphology,
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

    def generate_point_source_lightcurves(
            self,
            time,
            source_redshift,
            kappa_star_images,
            kappa_tot_images,
            shear_images,
            cosmology,
            kwargs_MagnificationMap:dict,
            point_source_morphology:str,
            kwargs_source_morphology:dict,
            lightcurve_type="magnitude",  # 'magnitude' or 'magnification'
            num_lightcurves=1,  # Number of lightcurves to generate
    ):
        """
        Generate lightcurves for one single point source with certain size,
        but for all images of that source based on the lens model. The point
        source is simulated as a "gaussian", "agn" (Accretion Disk) or "supernovae".

        The lightcurves are generated based on the microlensing map convolved with the source morphology kernel.

        The generated lightcurves will have the same length of time as the "time" array provided.

        :param time: Time array for which the lightcurve is needed.
        :param source_redshift: Redshift of the source
        :param kappa_star_images: list containing the kappa star for each image of the source.
        :param kappa_tot_images: list containing the kappa total for each image of the source.
        :param shear_images: list containing the shear for each image of the source.
        :param kwargs_MagnificationMap: Keyword arguments for the MagnificationMap class.
        :param point_source_morphology: Morphology of the point source. Options are "gaussian", "agn" (Accretion Disk) or "supernovae".
        :param kwargs_source_morphology: Dictionary of keyword arguments for the source morphology class. (See slsim.Microlensing.source_morphology for more details)
            
            For example, for Gaussian source morphology, it will look like: kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmo, "source_size": source_size, }.

            For AGN source morphology, it will look like: kwargs_source_morphology = {"source_redshift": source_redshift, "cosmo": cosmology, "r_out": r_out, "r_resolution": r_resolution, "smbh_mass_exp": smbh_mass_exp, "inclination_angle": inclination_angle, "black_hole_spin": black_hole_spin, "observer_frame_wavelength_in_nm": observer_frame_wavelength_in_nm, "eddington_ratio": eddington_ratio, }

        :param lightcurve_type: Type of lightcurve to generate, either 'magnitude' or 'magnification'. If 'magnitude', the lightcurve is returned in magnitudes normalized to the macro magnification.
                                If 'magnification', the lightcurve is returned in magnification without normalization. Default is 'magnitude'.
        :param num_lightcurves: Number of lightcurves to generate. Default is 1.
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
                magnification_map=magmaps_images[i], time_duration=lightcurve_duration,
                point_source_morphology=point_source_morphology,
                kwargs_source_morphology=kwargs_source_morphology
            )
            curr_lightcurves, curr_tracks, curr_time_arrays = (
                ml_lc.generate_lightcurves(
                    source_redshift=source_redshift,
                    cosmo=cosmology,
                    lightcurve_type=lightcurve_type,
                    effective_transverse_velocity = 1000, #TODO: Needs to be set from the velocity model!
                    num_lightcurves = num_lightcurves,
                    x_start_position = None,
                    y_start_position = None,
                    phi_travel_direction = None,
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
                **kwargs_MagnificationMap,
            )
            magmaps_images.append(magmap)

        return magmaps_images

    def _interpolate_light_curve(self, lightcurve, time_array, time_array_new):
        """Interpolate the lightcurve to a new time array.

        Assuming "lightcurve" and "time_array" are 1D arrays of the same
        length. "time_array_new" is a 1D array of the new time array.
        """
        return np.interp(time_array_new, time_array, lightcurve)
    
    def effective_transverse_velocity_images(
        self,
        source_redshift,
        deflector_redshift,
        ra_images,
        dec_images,
        cosmo,
        shear_phi_angle_images,
        deflector_velocity_dispersion,
    ):
        """Calculate the effective transverse velocity in the source plane for each image position.
        Eventually return the effective transverse velocity in frame of the magnification map by using appropriate transformations.

        This implementation is based on the works in the following papers [Credits: James Hung-Hsu Chan (for the Code), Luke Weisenbach and Henry Best]:
        1. https://arxiv.org/pdf/2004.13189
        2. https://iopscience.iop.org/article/10.1088/0004-637X/712/1/658/pdf
        3. https://iopscience.iop.org/article/10.3847/0004-637X/832/1/46/pdf

        :param source_redshift: Redshift of the source
        :param deflector_redshift: Redshift of the deflector
        :param ra_images: Right Ascension of the image position in degrees.
        :param dec_images: Declination of the image position in degrees.
        :param cosmo: Astropy cosmology object to use for the calculations.
        :param shear_phi_angle_images: list containing the angle of the shear vector, w.r.t. the x-axis of the image plane, in degrees for each image of the source.

        :return: array containing the effective transverse velocity (in km/s) for each image of the source.
        :rtype: 
        """
        
        #############################################
        # Lightman & Schechter 1990, Hamilton 2001
        # f = Omega_m**(4./7.) + Omega_v*(1.+Omega_m/2.)/70.
        #############################################
        def f_GrowthRate(Z_Redshift):
            Omega_m = cosmo.Om( Z_Redshift)
            Omega_v = cosmo.Ode(Z_Redshift)
            return Omega_m**(4./7.) + Omega_v*(1.+Omega_m/2.)/70.

        #############################################
        ## CMB
        lon_CMB = 264.021*u.deg
        lat_CMB =  48.253*u.deg
        v_CMB = 369.82*u.km/u.s
        #ra_CMB  = 167.94190333*u.deg
        #dec_CMB = -6.94425998*u.deg
        ############################

        z_src = source_redshift
        z_len = deflector_redshift

        #############################################
        D_s = cosmo.angular_diameter_distance (z_src)
        D_l = cosmo.angular_diameter_distance (z_len)
        D_ls = (cosmo.comoving_distance(z_src)-cosmo.comoving_distance(z_len))/(1.+z_src)
        #############################################
        epsilon = 1.

        #############################################
        # Kochanek04
        # sigma0 = 235 km/s and sigma_str = 215 km/s
        #############################################
        sigma0 = 235*(u.km/u.s)
        sigma_len = sigma0/(1+z_len)**0.5 * f_GrowthRate(z_len)/f_GrowthRate(0)
        sigma_src = sigma0/(1+z_src)**0.5 * f_GrowthRate(z_src)/f_GrowthRate(0)
        sigma_str = 215*(u.km/u.s)
        print('---------------------------------------------')
        print('sigma_lens, sigma_src, sigma_star =',sigma_len, sigma_src, sigma_str)
        #############################################
        ## CMB velocity
        #############################################
        coord_CMB = Galactic(l=lon_CMB,b=lat_CMB,distance=D_l,
                            pm_l_cosb=0*u.mas/u.yr,pm_b=0*u.mas/u.yr,
                            radial_velocity=v_CMB)
        #print(coord_CMB.transform_to(ICRS))
        coord_CMB.representation_type = 'cartesian'

        eff_trv_vels = [] # magnitude of the effective transverse velocities
        eff_trv_vels_angles = [] # phi_travel_directions

        #TODO: finish the part below!
        for i in range(len(ra_images)):
            ra = ra_images[i]*u.deg
            dec = dec_images[i]*u.deg
            shear_phi_angle = shear_phi_angle_images[i]*u.deg
            
            #############################################
            ## Convert velocity to ICRS 
            #############################################
            coord = ICRS(ra=ra,dec=dec,distance=D_l)
            coord = coord.transform_to(Galactic())
            vel_data= coord.data.to_cartesian().with_differentials(coord_CMB.velocity)
            coord = coord.realize_frame(vel_data)
            coord = coord.transform_to(ICRS())
            vox, voy = coord.pm_ra_cosdec, coord.pm_dec
            vox = vox.to(''/u.s, equivalencies=u.dimensionless_angles())*D_l
            voy = voy.to(''/u.s, equivalencies=u.dimensionless_angles())*D_l
            vox = vox.to(u.km/u.s)
            voy = voy.to(u.km/u.s)
            print('vt_CMB: v_ra_cosdec, v_dec =',vox,voy)
            #############################################
            vox = vox/(1+z_len) * D_ls/D_l 
            voy = voy/(1+z_len) * D_ls/D_l 
            print('vox, voy =',vox,voy)
            #############################################
            ## Mediavilla et al. 2016
            #############################################
            sigma_eff = (         sigma_len /(1+z_len) * D_s/D_l )**2\
                    + (         sigma_src /(1+z_src)           )**2\
                    + ( epsilon*sigma_str/(1+z_len) * D_s/D_l )**2
            sigma_eff = np.sqrt(sigma_eff)
            print('sigma_eff =',sigma_eff)
            print('---------------------------------------------')
            #############################################
    
    # def effective_transverse_velocity_images(
    #     self,
    #     source_redshift,
    #     deflector_redshift,
    #     ra_image,
    #     dec_image,
    #     cosmo,
    #     shear_phi_angle,
    # ):
    #     """Calculate the effective transverse velocity in the source plane for one image position.

    #     :param source_redshift: Redshift of the source
    #     :param deflector_redshift: Redshift of the deflector
    #     :param ra_image: Right Ascension of the image position in degrees.
    #     :param dec_image: Declination of the image position in degrees.
    #     :param cosmo: Astropy cosmology object to use for the calculations.
    #     :param shear_phi_angle: the angle of the shear vector, w.r.t. the x-axis of the image plane, in degrees.

    #     :return: effective transverse velocity (in km/s).
    #     :rtype: 
    #     """


