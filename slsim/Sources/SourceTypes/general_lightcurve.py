from slsim.Sources.SourceVariability.variability import Variability
from slsim.Sources.SourceTypes.source_base import SourceBase


class GeneralLightCurve(SourceBase):
    """A class to manage a source with given lightcurve."""

    def __init__(self, source_dict, **kwargs):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This table or dict should contain atleast redshift of a supernova, offset from
         the host if host galaxy is available, and lightcurve in a desired band.
         eg: data={"z": 0.8, "MJD":
         np.array([1,2,3,4,5,6,7,8,9]),
         "ps_mag_i": np.array([15, 16, 17, 18, 19, 20, 21, 22, 23]), "ra_off": 0.001,
           "dec_off": 0.002}
        :type source_dict: dict or astropy.table.Table
        :param kwargs: dictionary of keyword arguments for a source of lightcurve. It sould contain
          following keywords:
            :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
            :type variability_model: str
        """

        super().__init__(source_dict=source_dict)
        # These are the keywords that kwargs dict should contain
        self.variability_model = kwargs.get("variability_model")

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """

        column_names = self.source_dict.colnames
        if "ps_mag_" + band not in column_names:
            raise ValueError(
                "%s band magnitudes are missing in the source dictionary." % band
            )
        else:
            band_string = "ps_mag_" + band
            kwargs_variab_band = {
                "MJD": self.source_dict["MJD"],
                band_string: self.source_dict[band_string],
            }
            self.variability_class = Variability(
                self.variability_model, **kwargs_variab_band
            )
        if image_observation_times is not None:
            variable_mag = self.variability_class.variability_at_time(
                image_observation_times
            )
            return variable_mag
        return kwargs_variab_band[band_string]
