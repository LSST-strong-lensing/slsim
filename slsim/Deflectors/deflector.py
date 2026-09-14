from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import data_util
from slsim.Util import param_util, lenstronomy_util
import numpy as np
import lenstronomy.Util.constants as constants
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from slsim.Sources.source import Source
from slsim.Deflectors.mass import Mass


class Deflector(object):
    """Class of a single deflector with quantities only related to the
    deflector (independent of the source)"""

    def __init__(
        self,
        z,
        deflector_area=0.01,
        center_x=None,
        center_y=None,
        kwargs_mass=None,
        kwargs_light=None,
    ):
        """

        :param z: redshift of deflector
        :param deflector_area: area [arcsec^2] in which to randomly place the center of the deflector,
         if center is not provided.
        :param center_x: center of deflector y-coordinate
        :param center_y: center of deflector x-coordinate
        :param kwargs_mass: dictionary as input to Mass() class
        :param kwargs_light: dictionary as input to Deflector() class
        """
        if center_x is None or center_y is None:

            center_x, center_y = param_util.draw_coord_in_circle(
                area=deflector_area, size=1
            )
        self._center_lens = np.array([center_x, center_y])
        # make a Source() instance with the joint redshift and center position
        if kwargs_light is None:
            kwargs_light = {}
        self.light = Source(
            z=z, lensed=False, center_x=center_x, center_y=center_y, **kwargs_light
        )
        self.mass = Mass(light=self.light, **kwargs_mass)

    @property
    def name(self):
        """Takes name of light model.

        :return: name of object
        """
        return self.light.name

    @property
    def deflector_type(self):
        """Type of the mass deflector.

        :return: mass type
        :rtype: string
        """
        return self.mass.mass_type

    @property
    def redshift(self):
        """Deflector redshift.

        :return: redshift
        """
        return float(self.light.redshift)

    @property
    def redshift_list(self):
        """

        :return: list of redshifts for all mass models
        """
        return [self.redshift] * self.num_mass_models

    @property
    def num_mass_models(self):
        """

        :return: number of mass models
        :rtype: int
        """
        return self.mass.num_mass_models

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self.mass.velocity_dispersion(cosmo=cosmo)

    @property
    def deflector_center(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        # TODO: this routine might not be needed
        return self._center_lens

    def update_center(
        self, area=None, reference_position=None, center_x=None, center_y=None
    ):
        """Overwrites the deflector center position.

        :param reference_position: [RA, DEC] in arc-seconds of the
            reference from where within a circle the source position is
            being drawn from
        :type reference_position: 2d numpy array
        :param area: area (in solid angle arc-seconds^2) to dither the
            center of the source
        :param center_x: RA position [arc-seconds] (optional, otherwise
            renders within area)
        :param center_y: DEC position [arc-seconds] (optional, otherwise
            renders within area)
        :return: Source() instance updated with new center position
        """
        self.light.update_center(
            area=area,
            center_x=center_x,
            center_y=center_y,
            reference_position=reference_position,
        )
        # TODO: these next lines of code might not be required if done properly
        center_x, center_y = self.light.extended_source_position
        self._center_lens = np.array([center_x, center_y])

    @property
    def stellar_mass(self):
        """

        :return: stellar mass of deflector [M_sol]
        """
        return self.light.stellar_mass

    def magnitude(self, band):
        """Apparent magnitude of the deflector for a given band (extended
        light).

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        return self.light.extended_source_magnitude(band=band)

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
        return self.light.ellipticity

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
        return self.mass.mass_ellipticity

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        if lens_cosmo.z_lens >= lens_cosmo.z_source:
            return [], []
        return self.mass.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo, spherical=spherical
        )

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        return self.light.kwargs_extended_light(band=band)

    @property
    def angular_size_light(self):
        """Angular size of the light component.

        :return: angular size [arcsec]
        """
        return self.light.angular_size

    @property
    def mass_properties(self):
        """Properties of the NFW halo.

        :return: dictionary of relevant mass properties, such as halo
            mass M200 [physical M_sol], concentration r200/rs
        """
        return self.mass.mass_properties

    def surface_brightness(self, ra, dec, band=None):
        """Surface brightness at position ra/dec.

        :param ra: position RA
        :param dec: position DEC
        :param band: imaging band
        :type band: str
        :return: surface brightness at position ra/dec [mag / arcsec^2]
        """
        lens_light_model_list, kwargs_lens_light_mag = self.light_model_lenstronomy(
            band=band
        )
        return surface_brightness(ra, dec, lens_light_model_list, kwargs_lens_light_mag)

    def theta_e_infinity(self, cosmo, use_jax=True):
        """Einstein radius for a source at infinity (or well passed where
        galaxies exist).

        :param cosmo: astropy.cosmology instance
        :param use_jax: use JAX-accelerated lens models for lensing
            calculations, if available
        :type use_jax: bool
        :return: Einstein radius for source at infinite [arcsec]
        :type cosmo: ~astropy.cosmology class
        :return: Einstein radius for source at infinite [arcsec]
        """
        if hasattr(self, "_theta_e_infinity"):
            return self._theta_e_infinity
        if self.mass.mass_type in ["EPL"]:
            v_sigma = self.mass.velocity_dispersion(cosmo=cosmo)
            theta_E_infinity = (
                4 * np.pi * (v_sigma * 1000.0 / constants.c) ** 2 / constants.arcsec
            )
        else:
            _z_source_infty = 100
            lens_cosmo = LensCosmo(
                cosmo=cosmo, z_lens=self.redshift, z_source=_z_source_infty
            )
            lens_mass_model_list, kwargs_lens_mass = self.mass.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=True
            )
            theta_E_infinity = lenstronomy_util.theta_E_numerical(
                lens_mass_model_list=lens_mass_model_list,
                kwargs_lens_mass=kwargs_lens_mass,
                use_jax=use_jax,
            )

        self._theta_e_infinity = theta_E_infinity
        return theta_E_infinity


def surface_brightness(ra, dec, lens_light_model_list, kwargs_lens_light_mag):
    """Surface brightness at position ra/dec.

    :param ra: position RA
    :param dec: position DEC
    :param lens_light_model_list: list of light models in lenstronomy
        conventions
    :param kwargs_lens_light_mag: list of light model dictionaries with
        magnitudes
    :return: surface brightness at position ra/dec [mag / arcsec^2]
    """
    _mag_zero_dummy = 0  # from mag to amp conversion we need a dummy mag zero point. Irrelevant for this routine.
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
