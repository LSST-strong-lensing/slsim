from slsim.Deflectors.DeflectorTypes.epl_sersic import EPLSersic
from slsim.Deflectors.DeflectorTypes.epl import EPL
from slsim.Deflectors.DeflectorTypes.nfw_hernquist import NFWHernquist
from slsim.Deflectors.DeflectorTypes.nfw_cluster import NFWCluster
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import data_util
from slsim.Util import param_util
import numpy as np
import lenstronomy.Util.constants as constants
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel

_SUPPORTED_DEFLECTORS = ["EPL", "EPL_SERSIC", "NFW_HERNQUIST", "NFW_CLUSTER"]
JAX_PROFILES = [
    "EPL",
    "NFW",
    "HERNQUIST",
    "NFW_ELLIPSE_CSE",
    "HERNQUIST_ELLIPSE_CSE",
]


class Deflector(object):
    """Class of a single deflector with quantities only related to the
    deflector (independent of the source)"""

    def __init__(self, deflector_type, **kwargs):
        """

        :param deflector_type: type of deflector, i.e. "EPL", "NFW_HERNQUIST", "NFW_CLUSTER"
        :type deflector_type: str
        :param deflector_dict: parameters of the deflector
        :type deflector_dict: dict
        # TODO: document magnitude inputs
        """
        self._name = "GAL"
        if deflector_type in ["EPL"]:
            self._deflector = EPL(**kwargs)
        elif deflector_type in ["EPL_SERSIC"]:
            self._deflector = EPLSersic(**kwargs)
        elif deflector_type in ["NFW_HERNQUIST"]:
            self._deflector = NFWHernquist(**kwargs)
        elif deflector_type in ["NFW_CLUSTER"]:
            self._deflector = NFWCluster(**kwargs)
            self._name = "CLUSTER"
            self.subhalo_redshifts = self._deflector.subhalo_redshifts
            self.cored_profile = self._deflector.cored_profile
        else:
            raise ValueError(
                "Deflector type %s not supported. Chose among %s."
                % (deflector_type, _SUPPORTED_DEFLECTORS)
            )
        self.deflector_type = deflector_type

    @property
    def name(self):
        """Meaningful name string of the deflector.

        :return: name string
        """
        return self._name

    @property
    def redshift(self):
        """Deflector redshift.

        :return: redshift
        """
        return float(self._deflector.redshift)

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self._deflector.velocity_dispersion(cosmo=cosmo)

    @property
    def deflector_center(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self._deflector.deflector_center

    def update_center(self, deflector_area):
        """Overwrites the deflector center position.

        :param deflector_area: area (in solid angle arcseconds^2) to
            dither the center of the deflector
        :return:
        """
        return self._deflector.update_center(deflector_area)

    @property
    def stellar_mass(self):
        """

        :return: stellar mass of deflector [M_sol]
        """
        return self._deflector.stellar_mass

    def magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        return self._deflector.magnitude(band=band)

    @property
    def light_ellipticity(self):
        """Light ellipticity.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.

        :return: e1_light, e2_light
        """
        return self._deflector.light_ellipticity

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
        return self._deflector.mass_ellipticity

    def mass_model_lenstronomy(self, lens_cosmo):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        return self._deflector.mass_model_lenstronomy(lens_cosmo=lens_cosmo)

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        return self._deflector.light_model_lenstronomy(band=band)

    @property
    def angular_size_light(self):
        """Angular size of the light component.

        :return: angular size [arcsec]
        """
        return self._deflector.angular_size_light

    @property
    def halo_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return self._deflector.halo_properties

    def surface_brightness(self, ra, dec, band=None):
        """Surface brightness at position ra/dec.

        :param ra: position RA
        :param dec: position DEC
        :param band: imaging band
        :type band: str
        :return: surface brightness at postion ra/dec [mag / arcsec^2]
        """
        _mag_zero_dummy = 0  # from mag to amp conversion we need a dummy mag zero point. Irrelevant for this routine.
        lens_light_model_list, kwargs_lens_light_mag = self.light_model_lenstronomy(
            band=band
        )
        lightModel = LightModel(light_model_list=lens_light_model_list)

        kwargs_lens_light_amp = data_util.magnitude2amplitude(
            lightModel, kwargs_lens_light_mag, magnitude_zero_point=_mag_zero_dummy
        )
        flux_lens_light_local = lightModel.surface_brightness(
            ra, dec, kwargs_lens_light_amp
        )
        mag_arcsec2 = param_util.amplitude_to_magnitude(
            flux_lens_light_local, mag_zero_point=_mag_zero_dummy
        )
        return mag_arcsec2

    def theta_e_infinity(self, cosmo, multi_plane=None, use_jax=True):
        """Einstein radius for a source at infinity (or well passed where
        galaxies exist.

        :param cosmo: astropy.cosmology instance
        :param use_jax: use JAX-accelerated lens models for lensing
            calculations, if available
        :type use_jax: bool
        :return: Einstein radius for source at infinite [arcsec]
        :type cosmo: ~astropy.cosmology class
        :param multi_plane: None for single-plane, 'Source' for multi-
            source plane, 'Deflector' for multi-deflector plane, or
            'Both' for both multi-deflector and multi-source plane
        :type multi_plane: None or str
        :return: Einstein radius [arcsec]
        """
        if hasattr(self, "_theta_e_infinity"):
            return self._theta_e_infinity
        if self.deflector_type in ["EPL", "EPL_SERSIC"]:
            v_sigma = self._deflector.velocity_dispersion(cosmo=cosmo)
            theta_E_infinity = (
                4 * np.pi * (v_sigma * 1000.0 / constants.c) ** 2 / constants.arcsec
            )
        else:
            _z_source_infty = 100
            lens_cosmo = LensCosmo(
                cosmo=cosmo, z_lens=self.redshift, z_source=_z_source_infty
            )
            lens_mass_model_list, kwargs_lens_mass = (
                self._deflector.mass_model_lenstronomy(
                    lens_cosmo=lens_cosmo, spherical=True
                )
            )

            if multi_plane:

                if self.deflector_type in ["NFW_CLUSTER"]:
                    num_main_lens_profiles = len(lens_mass_model_list) - len(
                        self.subhalo_redshifts
                    )
                    lens_redshift_list = [self.redshift] * num_main_lens_profiles
                    lens_redshift_list.extend(self.subhalo_redshifts)
                else:
                    num_main_lens_profiles = len(lens_mass_model_list)
                    lens_redshift_list = [self.redshift] * num_main_lens_profiles
                if use_jax is True:
                    _use_jax = True
                else:
                    _use_jax = False
            else:
                lens_redshift_list = None
                if use_jax is True:
                    _use_jax = []
                    for profile in lens_mass_model_list:
                        if profile in JAX_PROFILES:
                            _use_jax.append(True)
                        else:
                            _use_jax.append(False)
                else:
                    _use_jax = False

            lens_model = LensModel(
                lens_model_list=lens_mass_model_list,
                z_lens=self.redshift,
                lens_redshift_list=lens_redshift_list,
                z_source_convention=_z_source_infty,
                multi_plane=bool(multi_plane),
                z_source=_z_source_infty,
                cosmo=cosmo,
                use_jax=_use_jax,
            )

            lens_analysis = LensProfileAnalysis(lens_model=lens_model)

            theta_E_infinity = lens_analysis.effective_einstein_radius(
                kwargs_lens_mass,
                r_min=1e-3,
                r_max=5e1,
                num_points=40,
                spherical_model=True,
            )
            theta_E_infinity = np.nan_to_num(theta_E_infinity, nan=0)
        self._theta_e_infinity = theta_E_infinity
        return theta_E_infinity
