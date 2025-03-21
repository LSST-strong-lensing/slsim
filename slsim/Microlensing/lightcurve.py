__author__ = "Paras Sharma"

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented
from Microlensing import HENRYS_AMOEBA_PATH
import sys

sys.path.append(HENRYS_AMOEBA_PATH)
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.magnification_map import MagnificationMap as AmoebaMagnificationMap


class MicrolensingLightCurve(object):
    """Class to generate lightcurves based on the magnification maps and the
    source."""

    def __init__(self, magnification_map, source, image_observing_time):
        """
        :param magnification_map: MagnificationMap object
        :param source: Source object
        """
        self.magnification_map = magnification_map
        self.source = source

    def generate_lightcurve(self):
        """Generate lightcurve based on the source type."""
        pass

    def _generate_agn_lightcurve(self):
        """Generate lightcurve for a quasar(AGN) with the accretion disk model
        from amoeba."""
        pass

    def _generate_supernova_lightcurve(self):
        """Generate lightcurve for a supernova."""
        pass
