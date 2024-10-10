import numpy as np
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from slsim.ParamDistributions.los_config import LOSConfig
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy

class FalsePositive(object):
    """Class to manage individual false positive."""

    def __init__(
        self,
        source_class,
        deflector_class,
        cosmo,
        test_area=4 * np.pi,
        los_config=None,
        los_dict=None,
    ):
        """
        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param cosmo: astropy.cosmology instance
        :param test_area: area of disk around one lensing galaxies to be investigated
            on (in arc-seconds^2).
        :param los_config: LOSConfig instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param los_dict: line of sight dictionary (optional, takes these values instead of drawing from distribution)
         Takes "gamma" = [gamma1, gamma2] and "kappa" = kappa as entries
        :type los_dict: dict
        """
        self.deflector = deflector_class
        self.source = source_class
        self.test_area = test_area
        self.cosmo = cosmo
        if isinstance(self.source, list):
            source_z=self.source[0].redshift
            self._source_type = self.source[0].source_type
            self.source_number = len(self.source)
            self.single_source_class = self.source[0] # to access some common kwargs.
        else:
            source_z=self.source.redshift
            self._source_type = self.source.source_type
            self.source_number = 1
            self.single_source_class = self.source
        self._lens_cosmo = LensCosmo(
            z_lens=float(self.deflector.redshift),
            z_source=float(source_z),
            cosmo=self.cosmo,
        )

        self._los_linear_distortions_cache = None
        self.los_config = los_config
        if self.los_config is None:
            if los_dict is None:
                los_dict = {}
            self.los_config = LOSConfig(**los_dict)


    @property
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self.deflector.deflector_center
    
    @property
    def deflector_redshift(self):
        """

        :return: lens redshift
        """
        return self.deflector.redshift

    @property
    def source_redshift(self):
        """

        :return: a source redshift or list of source redshift
        """
        if self.source_number==1:
            source_redshift = self.source.redshift
        else:
            source_redshift=[]
            for i in range(self.source_number):
                source_redshift.append(self.source[i].redshift)
        return source_redshift
            
    @property
    def external_convergence(self):
        """

        :return: external convergence
        """
        _, _, kappa_ext = self.los_linear_distortions
        return kappa_ext

    @property
    def external_shear(self):
        """

        :return: the absolute external shear
        """
        gamma1, gamma2, _ = self.los_linear_distortions
        return (gamma1**2 + gamma2**2) ** 0.5

    @property
    def einstein_radius(self):
        """Einstein radius, from SIS approximation (coming from velocity dispersion) +
        external convergence effect.

        :return: Einstein radius [arc seconds]
        """
        theta_E = self.einstein_radius_deflector
        _, _, kappa_ext = self.los_linear_distortions
        return theta_E / (1 - kappa_ext)

    def deflector_ellipticity(self):
        """

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        e1_light, e2_light = self.deflector.light_ellipticity
        e1_mass, e2_mass = self.deflector.mass_ellipticity
        return e1_light, e2_light, e1_mass, e2_mass

    def deflector_stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        return self.deflector.stellar_mass

    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        return self.deflector.velocity_dispersion(cosmo=self.cosmo)

    @property
    def los_linear_distortions(self):
        if self._los_linear_distortions_cache is None:
            self._los_linear_distortions_cache = (
                self._calculate_los_linear_distortions()
            )
        return self._los_linear_distortions_cache

    def _calculate_los_linear_distortions(self):
        """Line-of-sight distortions in shear and convergence.

        :return: kappa, gamma1, gamma2
        """
        if self.source_number==1:
            source_z = self.source_redshift
        else:
            source_z = self.source_redshift[0]
        return self.los_config.calculate_los_linear_distortions(
            source_redshift=source_z,
            deflector_redshift=self.deflector_redshift,
        )

    def deflector_magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        return self.deflector.magnitude(band=band)


    def extended_source_magnitude(self, band):
        """Unlensed apparent magnitude of the extended source for a given band (assumes
        that size is the same for different bands)

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        # band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = self.source.extended_source_magnitude(band)
        return source_mag

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_mass_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        (
            lens_light_model_list,
            kwargs_lens_light,
        ) = self.deflector.light_model_lenstronomy(band=band)

        sources, sources_kwargs = self.source_light_model_lenstronomy(band=band)
        combined_lens_light_model_list = lens_light_model_list + sources[
            "source_light_model_list"]
        combined_kwargs_lens_light = kwargs_lens_light + sources_kwargs["kwargs_source"]

        kwargs_model = {
            "lens_light_model_list": combined_lens_light_model_list,
            "lens_model_list": lens_mass_model_list,
        }

        kwargs_source = None
        kwargs_ps = sources_kwargs["kwargs_ps"]

        kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": combined_kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }

        return kwargs_model, kwargs_params

    def deflector_mass_model_lenstronomy(self):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :return: lens_model_list, kwargs_lens
        """
        if self.deflector.deflector_type in ["EPL", "NFW_HERNQUIST", "NFW_CLUSTER"]:
            lens_mass_model_list, kwargs_lens = self.deflector.mass_model_lenstronomy(
                lens_cosmo=self._lens_cosmo
            )
        else:
            raise ValueError(
                "Deflector model %s not supported for lenstronomy model"
                % self.deflector.deflector_type
            )
        # adding line-of-sight structure
        gamma1, gamma2, kappa_ext = self.los_linear_distortions
        gamma1_lenstronomy, gamma2_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=gamma1, e2_slsim=gamma2
        )
        kwargs_lens.append(
            {
                "gamma1": gamma1_lenstronomy,
                "gamma2": gamma2_lenstronomy,
                "ra_0": 0,
                "dec_0": 0,
            }
        )
        kwargs_lens.append({"kappa": kappa_ext, "ra_0": 0, "dec_0": 0})
        lens_mass_model_list.append("SHEAR")
        lens_mass_model_list.append("CONVERGENCE")

        return lens_mass_model_list, kwargs_lens

    def deflector_light_model_lenstronomy(self, band):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        return self.deflector.light_model_lenstronomy(band=band)

    def source_light_model_lenstronomy(self, band=None):
        """Returns source light model instance and parameters in lenstronomy
        conventions.

        :return: source_light_model_list, kwargs_source_light
        """
        source_models = {}
        all_source_kwarg_dict = {}
        if (
            self._source_type == "extended"
            or self._source_type == "point_plus_extended"
        ):
            if self.source_number==1:
                source_class=self.source
            else:
                source_class=self.source[0]
            if source_class.light_profile == "single_sersic":
                source_models["source_light_model_list"] = [
                    "SERSIC_ELLIPSE"]*self.source_number
            #In this case we will consider a single source with double sersic profile.
            else:
                source_models["source_light_model_list"] = [
                    "SERSIC_ELLIPSE",
                    "SERSIC_ELLIPSE",
                ]
            if self.source_number==1:
                kwargs_source = self.source.kwargs_extended_source_light(
                    draw_area=self.test_area,
                    center_lens=self.deflector_position,
                    band=band
                )
            else:
                kwargs_source=[]
                for i in range(self.source_number):
                    kwargs_source.append(self.source[i].kwargs_extended_source_light(
                    draw_area=self.test_area,
                    center_lens=self.deflector_position,
                    band=band
                )[0])
        else:
            # source_models['source_light_model_list'] = None
            kwargs_source = None

        if (
            self._source_type == "point_source"
            or self._source_type == "point_plus_extended"
        ):
            source_models["point_source_model_list"] = ["LENSED_POSITION"]
            img_x, img_y = self.point_source_image_positions()
            if band is None:
                image_magnitudes = np.abs(self.point_source_magnification())
            else:
                image_magnitudes = self.point_source_magnitude(band=band, lensed=False)
            kwargs_ps = [
                {"ra_image": img_x, "dec_image": img_y, "magnitude": image_magnitudes}
            ]
        else:
            # source_models['point_source_model'] = None
            kwargs_ps = None
        all_source_kwarg_dict["kwargs_source"] = kwargs_source
        all_source_kwarg_dict["kwargs_ps"] = kwargs_ps
        return source_models, all_source_kwarg_dict