from slsim.Sources.SourceTypes.point_source import PointSource
from slsim.Sources.SourceTypes.extended_source import ExtendedSource


class PointPlusExtendedSource(PointSource, ExtendedSource):
    """Class to manage a single point source and a single extended source
    (host)"""

    def __init__(
        self,
        extended_source_type,
        point_source_type,
        cosmo=None,
        **source_dict,
    ):
        """
        :param source_dict: Source properties. Can be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, Interpolated classes, Supernova, and Quasar class.
        :type source_dict: dict or astropy.table.Table .
         eg of a supernova plus host galaxy dict: {"z": 0.8, "mag_i": 22, "n_sersic": 1,
           "angular_size": 0.10, "e1": 0.002, "e2": 0.001, "ra_off": 0.001, "dec_off": 0.005}
        :param extended_source_type: keyword for specifying light profile model.
        :type extended_source_type: str. supported types are "single_sersic",
         "double_sersic", "interpolated".
        :param point_source_type: keyword for specifying point source type.
        :type point_source_type: str. supported types are "supernova", "quasar", "general_lightcurve".
        """
        # Initialize the extended source. Here, source_dict will contain both host
        # galaxy and point source information but only extended source properties will
        # be read in the ExtendedSource class and only point source properties will be
        # read in the point source class.

        # Initialize the point source first.
        PointSource.__init__(
            self,
            source_type=point_source_type,
            cosmo=cosmo,
            **source_dict,
        )

        ExtendedSource.__init__(
            self,
            source_type=extended_source_type,
            cosmo=cosmo,
            **source_dict,
        )

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
        self._point_source.update_center(
            area=area,
            reference_position=reference_position,
            center_x=center_x,
            center_y=center_y,
        )
        center = self._point_source.extended_source_position
        self._extended_source.update_center(center_x=center[0], center_y=center[1])

    def update_microlensing_kwargs_source_morphology(self, kwargs_source_morphology):
        """Update the microlensing kwargs_source_morphology for the point
        source.

        :param kwargs_source_morphology: Dictionary of source morphology
            parameters. See Microlensing.source_morphology for details.
        :return: Updated dictionary of source morphology parameters.
        """
        return self._point_source.update_microlensing_kwargs_source_morphology(
            kwargs_source_morphology=kwargs_source_morphology
        )
