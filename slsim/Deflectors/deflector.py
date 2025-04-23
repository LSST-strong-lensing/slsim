from slsim.Deflectors.DeflectorTypes.epl_sersic import EPLSersic
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

_SUPPORTED_DEFLECTORS = ["EPL", "NFW_HERNQUIST", "NFW_CLUSTER"]


class Deflector(object):
    """Class of a single deflector with quantities only related to the
    deflector (independent of the source)"""

    def __init__(self, deflector_type, deflector_dict, **kwargs):
        """

        :param deflector_type: type of deflector, i.e. "EPL", "NFW_HERNQUIST", "NFW_CLUSTER"
        :type deflector_type: str
        :param deflector_dict: parameters of the deflector
        :type deflector_dict: dict
        # TODO: document magnitude inputs
        """
        if deflector_type in ["EPL"]:
            self._deflector = EPLSersic(deflector_dict=deflector_dict, **kwargs)
        elif deflector_type in ["NFW_HERNQUIST"]:
            self._deflector = NFWHernquist(deflector_dict=deflector_dict)
        elif deflector_type in ["NFW_CLUSTER"]:
            self._deflector = NFWCluster(deflector_dict=deflector_dict)
        else:
            raise ValueError(
                "Deflector type %s not supported. Chose among %s."
                % (deflector_type, _SUPPORTED_DEFLECTORS)
            )
        self.deflector_type = deflector_type

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

        :return: angular size [radian]
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

    def theta_e_infinity(self, cosmo):
        """Einstein radius for a source at infinity (or well passed where
        galaxies exist.

        :param cosmo: astropy.cosmology instance
        :return:
        """
        if self.deflector_type in ["EPL"]:
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
            lens_model = LensModel(
                lens_model_list=lens_mass_model_list,
                z_lens=self.redshift,
                z_source_convention=_z_source_infty,
                multi_plane=False,
                z_source=_z_source_infty,
                cosmo=cosmo,
            )

            lens_analysis = LensProfileAnalysis(lens_model=lens_model)

            theta_E_infinity = lens_analysis.effective_einstein_radius(
                kwargs_lens_mass, r_min=1e-3, r_max=5e1, num_points=40, spherical_model=True,
            )
            theta_E_infinity = np.nan_to_num(theta_E_infinity, nan=0)
        return theta_E_infinity
