from slsim.Sources.SourceVariability.variability import Variability
from slsim.Sources.SourceTypes.source_base import SourceBase
import numpy as np


class GeneralLightCurve(SourceBase):
    """A class to manage a source with given lightcurve."""

    def __init__(self, MJD, variability_model="light_curve", **kwargs):
        """


        :param MJD: list of times of the recorded magnitudes
        :type MJD: array of same length as ps_mag_<band>
        :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
        :type variability_model: str
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This table or dict should contain atleast redshift of a supernova, offset from
         the host if host galaxy is available, and lightcurve in a desired band.
         eg: data={"z": 0.8, "MJD":
         np.array([1,2,3,4,5,6,7,8,9]),
         "ps_mag_i": np.array([15, 16, 17, 18, 19, 20, 21, 22, 23]), "ra_off": 0.001,
           "dec_off": 0.002}
        :type source_dict: dict or astropy.table.Table
        :param kwargs: dictionary of keyword arguments for a source of lightcurve.

        """

        super().__init__(
            variability_model=variability_model,
            point_source=True,
            extended_source=False,
            **kwargs
        )
        self.name = kwargs.get("name", "LC")
        # These are the keywords that kwargs dict should contain
        self._MJD = MJD

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """

        band_string = "ps_mag_" + band
        if band_string not in self.source_dict:
            raise ValueError(
                "%s band magnitudes are missing in the source dictionary." % band
            )

        if image_observation_times is None or self._variability_model == "NONE":
            band_string = "ps_mag_" + band
            return np.mean(self.source_dict[band_string])
        else:
            if band not in self._variability_bands:
                kwargs_variab_band = {
                    "MJD": self._MJD,
                    band_string: self.source_dict[band_string],
                }
                self._variability_bands[band] = Variability(
                    variability_model=self._variability_model, **kwargs_variab_band
                )
            return self._variability_bands[band].variability_at_time(
                image_observation_times
            )
