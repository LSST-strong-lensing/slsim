from slsim.Util import param_util, lenstronomy_util
import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from slsim.Deflectors.deflector import surface_brightness, Deflector


class DeflectorGroup(object):
    """Class to handle sets of deflectors that are bundled together.

    They have joint redshift, and a centroid
    """

    def __init__(
        self,
        z,
        kwargs_mass_list,
        kwargs_light_list,
        center_x_deflector_list,
        center_y_deflector_list,
        deflector_area=0.01,
        center_x=None,
        center_y=None,
    ):
        """

        :param z: redshift of deflector
        :param kwargs_mass_list: list of dictionary as input to Mass() class
        :param kwargs_light_list: list of dictionary as input to Source() class
        :param center_x_deflector_list: list of x-positions for all the deflectors (relative to global center_x)
        :param center_y_deflector_list: list of y-positions for all the deflectors (relative to global center_y)
        :param deflector_area: area [arcsec^2] in which to randomly place the center of the deflector,
         if center is not provided.
        :param center_x: global center of deflector y-coordinate
        :param center_y: global center of deflector x-coordinate

        """
        if center_x is None or center_y is None:

            center_x, center_y = param_util.draw_coord_in_circle(
                area=deflector_area, size=1
            )
        self._center_lens = np.array([center_x, center_y])
        # make a Source() instance with the joint redshift and center position
        if len(kwargs_mass_list) != len(kwargs_light_list):
            raise ValueError(
                "length of list of kwargs_mass = %s needs to be the same as the length of "
                "kwargs_light = %s" % (len(kwargs_mass_list), len(kwargs_light_list))
            )

        self._num_deflectors = len(kwargs_mass_list)
        if (
            len(center_x_deflector_list) != self._num_deflectors
            or len(center_y_deflector_list) != self._num_deflectors
        ):
            raise ValueError(
                "length of list of center_x_deflector_list = %s and center_y_deflector_list = %s "
                "needs to be the same as the length of "
                "kwargs_light = %s"
                % (
                    len(center_x_deflector_list),
                    len(center_y_deflector_list),
                    self._num_deflectors,
                )
            )
        self._deflector_list = []
        for i in range(self._num_deflectors):
            center_x_i = center_x + center_x_deflector_list[i]
            center_y_i = center_y + center_y_deflector_list[i]
            deflector = Deflector(
                z=z,
                kwargs_mass=kwargs_mass_list[i],
                kwargs_light=kwargs_light_list[i],
                center_x=center_x_i,
                center_y=center_y_i,
            )
            self._deflector_list.append(deflector)
        self._deflector_type = "group"
        self._z = float(z)
        self._center_x_deflector_list = center_x_deflector_list
        self._center_y_deflector_list = center_y_deflector_list

    @property
    def deflector_type(self):
        """Type of the mass deflector.

        :return: mass type
        :rtype: string
        """
        return self._deflector_type

    @property
    def redshift_list(self):
        """

        :return: list of redshifts for all mass models
        """
        return [self.redshift] * self._num_mass_models

    @property
    def _num_mass_models(self):
        """

        :return: number of mass models
        :rtype: int
        """
        num_models = 0
        for i, deflector in enumerate(self._deflector_list):
            num_models += deflector.num_mass_models
        return num_models

    def deflector(self, deflector_index=0):
        """Single Deflector() class.

        :param deflector_index: index of deflector
        :return: ~slsim.Deflector() type
        """
        return self._deflector_list[deflector_index]

    @property
    def redshift(self):
        """Deflector redshift.

        :return: redshift
        """
        return self._z

    def velocity_dispersion(self, cosmo=None, deflector_index=0):
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param deflector_index: index of deflector
        :return: velocity dispersion [km/s]
        """
        return self._deflector_list[deflector_index].velocity_dispersion(cosmo=cosmo)

    @property
    def deflector_center(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
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

        self._center_lens = param_util.update_center(
            area=area,
            reference_position=reference_position,
            center_x=center_x,
            center_y=center_y,
        )
        center_x, center_y = self._center_lens[0], self._center_lens[1]
        for i, deflector in enumerate(self._deflector_list):
            deflector.update_center(
                center_x=center_x + self._center_x_deflector_list[i],
                center_y=center_y + self._center_y_deflector_list[i],
            )

    @property
    def stellar_mass(self):
        """

        :return: stellar mass of deflector [M_sol]
        """
        stellar_mass = 0
        for deflector in self._deflector_list:
            stellar_mass_i = deflector.stellar_mass
            if stellar_mass_i is not None:
                stellar_mass += stellar_mass_i
        return stellar_mass

    def magnitude(self, band):
        """Apparent magnitude of the deflector for a given band (extended
        light).

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        flux = 0
        for deflector in self._deflector_list:
            mag_ = deflector.magnitude(band=band)
            if mag_ is not None:
                flux_i = 10 ** (-0.4 * mag_)
                flux += flux_i
        mag_tot = -2.5 * np.log10(flux)
        return mag_tot

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
        if lens_cosmo.z_lens >= lens_cosmo.z_source:
            return [], []
        lens_mass_model_list = []
        kwargs_lens_mass = []
        for deflect in self._deflector_list:
            model_list, kwargs_mass = deflect.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=spherical
            )
            lens_mass_model_list += model_list
            kwargs_lens_mass += kwargs_mass
        return lens_mass_model_list, kwargs_lens_mass

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        light_model_list = []
        kwargs_light_list = []
        for deflector in self._deflector_list:
            light_model, kwargs_light = deflector.light_model_lenstronomy(band=band)
            light_model_list += light_model
            kwargs_light_list += kwargs_light
        return light_model_list, kwargs_light_list

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
        :type cosmo: ~astropy.cosmology class
        :return: Einstein radius for source at infinite [arcsec]
        """
        _z_source_infty = 100
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.redshift, z_source=_z_source_infty
        )
        lens_mass_model_list, kwargs_lens_mass = self.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo, spherical=True
        )
        theta_E_infinity = lenstronomy_util.theta_E_numerical(
            lens_mass_model_list=lens_mass_model_list,
            kwargs_lens_mass=kwargs_lens_mass,
            use_jax=use_jax,
        )
        self._theta_e_infinity = theta_E_infinity
        return theta_E_infinity
