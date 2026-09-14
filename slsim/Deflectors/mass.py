_SUPPORTED_DEFLECTORS = ["EPL", "NFW_HERNQUIST", "NFW", "HERNQUIST", "PJAFFE"]


class Mass(object):
    """"""

    def __init__(self, light, mass_type, **mass_dict):
        """

        :param light: Light model class instance
        :type light: ~slsim.Source.source object
        :param mass_type:
        :param mass_dict: mass model parameters
        :type mass_dict: dict
        """
        self._light = light
        self.mass_type = mass_type

        if mass_type in ["EPL"]:
            from slsim.Deflectors.MassTypes.epl import EPL

            self._mass = EPL(light=light, **mass_dict)
        elif mass_type in ["NFW_HERNQUIST"]:
            from slsim.Deflectors.MassTypes.nfw_hernquist import NFWHernquist

            self._mass = NFWHernquist(light=light, **mass_dict)
        elif mass_type in ["NFW"]:
            from slsim.Deflectors.MassTypes.nfw import NFW

            self._mass = NFW(light=light, **mass_dict)
        elif mass_type in ["HERNQUIST"]:
            from slsim.Deflectors.MassTypes.hernquist import Hernquist

            self._mass = Hernquist(light=light, **mass_dict)
        elif mass_type in ["PJAFFE"]:
            from slsim.Deflectors.MassTypes.pjaffe import PJAFFE

            self._mass = PJAFFE(light=light, **mass_dict)
        else:
            raise ValueError(
                "Deflector type %s not supported. Chose among %s."
                % (mass_type, _SUPPORTED_DEFLECTORS)
            )
        self.deflector_type = mass_type

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self._mass.velocity_dispersion(cosmo=cosmo)

    @property
    def mass_ellipticity(self):
        """Mass ellipticity
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.

        :return: e1_mass, e2_mass
        """
        return self._mass.ellipticity

    @property
    def mass_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return self._mass.mass_properties

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, removes ellipticity for simpler
            calculations
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        return self._mass.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo, spherical=spherical
        )

    @property
    def num_mass_models(self):
        """Number of mass models.

        :return: number of mass models
        """
        return self._mass.num_mass_models
