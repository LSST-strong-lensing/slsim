from abc import ABC
import numpy as np
from slsim.Util import param_util
from slsim.Sources.SourceVariability.variability import Variability


class SourceBase(ABC):
    """Class of a single source with quantities only related to the source
    (independent of the deflector)"""

    def __init__(
        self,
        z,
        model_type="SourceBase",
        lensed=True,
        point_source=False,
        extended_source=False,
        center_x=None,
        center_y=None,
        ra_off=None,
        dec_off=None,
        angular_size=None,
        e1=None,
        e2=None,
        cosmo=None,
        variability_model="NONE",
        kwargs_variability_model=None,
        **kwargs
    ):
        """

        :param z: redshift
        :param model_type: the model type, mostly used for error messages and information to the user
        :type model_type: str
        :param lensed: if True, regards the model be part of the lensed source,
         if False, it can be used as a deflector (unlensed) model
        :param point_source: whether a point source is involved
        :type point_source: bool
        :param extended_source: whether an extended source is involved
        :type extended_source: bool
        :param center_x: RA coordinate (relative arcseconds)
        :param center_y: DEC coordinate (relative arcseconds)
        :param ra_off: RA offset of point source from center of host (relative arc-seconds)
        :param dec_off: DEC offset of point source from center of host (relative arc-seconds)
        :param angular_size: angular size of the object [arcseconds]
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param variability_model: variability model supported in
         ~slsim.Sources.SourceVariability.variability.Variability() class
        :type variability_model: string
        :param kwargs_variability_model: Dictionary with bands as strings, each containing a dictionary for a
         Variability() class input configurations for point source variability
        :type kwargs_variability_model: dict of dict
        :param kwargs: ps_mag_<band> keyword arguments
        :type kwargs: dict
        :type source_dict: dict or astropy.table.Table
        """
        self.name = "NONE"
        self._z = float(z)
        self.lensed = lensed
        self._model_type = model_type
        self._point_source = point_source
        self._extended_source = extended_source
        self.update_center(
            area=None, reference_position=None, center_x=center_x, center_y=center_y
        )
        if ra_off is not None and dec_off is not None:
            if point_source is False:
                self._offset = np.array(
                    [0, 0]
                )  # without a point source, offsets have to be zero
            self._offset = np.array([float(ra_off), float(dec_off)])
        else:
            self._offset = np.array([0, 0])
        self.source_dict = kwargs

        self._variability_bands = (
            {}
        )  # empty dictionary to be filled with Variability() models for different bands
        self._variability_model = variability_model
        if kwargs_variability_model is None:
            kwargs_variability_model = {}
        self._kwargs_variability_model = kwargs_variability_model
        self._angular_size = angular_size
        if e1 is not None:
            e1 = float(e1)
        if e2 is not None:
            e2 = float(e2)
        self._e1, self._e2 = e1, e2
        self._cosmo = cosmo

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._z

    def update_center(
        self, area=None, reference_position=None, center_x=None, center_y=None
    ):
        """Overwrites the source center position.

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
        if center_x is not None and center_y is not None:
            self._center_source = np.array([float(center_x), float(center_y)])
        else:
            if reference_position is None:
                reference_position = np.array([0, 0])
            if area is None:
                x_, y_ = 0, 0
            else:
                x_, y_ = param_util.draw_coord_in_circle(area=area, size=1)
            self._center_source = np.array(
                [
                    reference_position[0] + x_,
                    reference_position[1] + y_,
                ]
            )

    @property
    def point_source_offset(self):
        """Provides point source offset from host center."""
        return self._offset

    @property
    def angular_size(self):
        """Returns angular size of the source for two component of the sersic
        profile.

        :return: angular size [arcseconds]
        """
        if self._extended_source is False:
            return 0
        if self._angular_size is not None:
            return self._angular_size
        else:
            raise ValueError("Angular size of the source is not specified.")

    def physical_size(self, cosmo):
        """

        :param cosmo: ~astropy.cosmology instance
        :return: physical size [kpc]
        """
        ang_dist = cosmo.angular_diameter_distance(self.redshift)
        physical_size = self.angular_size * 4.84814e-6 * ang_dist.value * 1000  # kPc
        return physical_size

    @property
    def ellipticity(self):
        """Returns ellipticity components of source for the both component of
        the light profile. first two ellipticity components are associated with the
        first sersic component and last two are associated with the second sersic component.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """
        if self._extended_source is False:
            return 0, 0
        if self._e1 is not None and self._e2 is not None:
            return self._e1, self._e2
        else:
            raise ValueError("ellipticity of extended object is not specified.")

    @property
    def extended_source_position(self):
        """Extended source position.

        :return: [x_pos, y_pos] [arc-seconds]
        """
        return self._center_source

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: list of extended source profiles (lenstronomy
            conventions), dictionary of keywords for the source light
            model(s)
        """
        if self._extended_source is False:
            return [], []
        else:
            raise NotImplementedError(
                "Model type %s requires an implementation of an extended surface brightness "
                "profile." % self._model_type
            )

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """
        if self._extended_source is False:
            return None
        band_string = "mag_" + band
        if band_string not in self.source_dict:
            raise ValueError(
                "required parameter %s is missing in the source dictionary to evaluate extended source "
                "magnitude in band %s." % (band_string, band)
            )
        return self.source_dict[band_string]

    @property
    def point_source_position(self):
        """Point source position.

        :return: [x_pos, y_pos]
        """
        if not hasattr(self, "_center_point_source"):
            self._center_point_source = self.extended_source_position + self._offset
        return self._center_point_source

    def point_source_magnitude(
        self, band, image_observation_times=None, at_maximum=False
    ):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :param at_maximum: returns maximum brightness (minimum
            magnitude) of a light curve. I.e. peak magnitude of a SNe.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        if self._point_source is False:
            return []
        if image_observation_times is None or self._variability_model == "NONE":
            band_string = "ps_mag_" + band
            if band_string not in self.source_dict:
                raise ValueError(
                    "required parameter %s is missing in the source dictionary to provide point source "
                    "magnitude without a image_observation_time or without variability model."
                    % band_string
                )
            return np.mean(self.source_dict[band_string])
        else:
            if band not in self._variability_bands:
                if band not in self._kwargs_variability_model:
                    raise ValueError(
                        "kwargs_variability_model requires information about band %s"
                        % band
                    )
                kwargs_variab_band = self._kwargs_variability_model[band]
                self._variability_bands[band] = Variability(
                    variability_model=self._variability_model, **kwargs_variab_band
                )
            return self._variability_bands[band].variability_at_time(
                image_observation_times
            )

    def point_source_type(self, image_positions=False):
        """Type of point source model.

        :param image_positions:
        :return: point source model string, or None
        """
        if self._point_source is False:
            return None
        if self.lensed:
            if image_positions:
                return "LENSED_POSITION"
            else:
                return "SOURCE_POSITION"
        else:
            return "UNLENSED"

    def kwargs_point_source(
        self,
        band,
        image_observation_times=None,
        image_pos_x=None,
        image_pos_y=None,
        ps_mag=None,
    ):
        """

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an image.
        :param image_pos_x: pre-calculated image positions (solutions of the lens equation) RA [arcseconds]
        :param image_pos_y: pre-calculated image positions (solutions of the lens equation) DEC [arcseconds]
        :param ps_mag: magnitudes of images (or source)
        :return: source type, list of dictionary in lenstronomy convention
        """
        if image_pos_x is None or image_pos_y is None:
            image_positions = False
        else:
            image_positions = True
        ps_type = self.point_source_type(image_positions=image_positions)
        if ps_type is None:
            return None, []
        # get point source position
        if not image_positions:
            ps_position = self.point_source_position
            image_pos_x, image_pos_y = ps_position[0], ps_position[1]
        # get magnitude
        if image_observation_times is not None:
            if len(image_pos_x) != len(image_observation_times):
                raise ValueError(
                    "length of image positions needs to be the length of the observation times"
                )
        if ps_mag is None:
            ps_mag = self.point_source_magnitude(
                band=band, image_observation_times=image_observation_times
            )
        if np.shape(np.asarray(ps_mag)) != np.shape(np.asarray(image_pos_x)):
            raise ValueError(
                "length of image positions %s needs to be the length of the ps_mag %s"
                % (np.shape(image_pos_x), np.shape(ps_mag))
            )

        # set keyword arguments for lenstronomy
        if ps_type in ["LENSED_POSITION", "UNLENSED"]:
            kwargs_ps = {
                "ra_image": image_pos_x,
                "dec_image": image_pos_y,
                "magnitude": ps_mag,
            }
        else:
            kwargs_ps = {
                "ra_source": image_pos_x,
                "dec_source": image_pos_y,
                "magnitude": ps_mag,
            }
        return ps_type, kwargs_ps

    def update_microlensing_kwargs_source_morphology(self, kwargs_source_morphology):
        """Update the kwargs_source_morphology dictionary with parameters from
        the respective source type class. This method will be overwritten in
        child classes for specific source types.

        :param kwargs_source_morphology: Dictionary of source morphology
            parameters.
        :return: Updated dictionary of source morphology parameters.
        """
        return kwargs_source_morphology
