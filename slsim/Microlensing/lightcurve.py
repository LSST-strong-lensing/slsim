__author__ = 'Paras Sharma'

# here we generate the lightcurve from the microlensing map
# this process can be different depending on the source type
# currently only Quasar is implemented
# from Microlensing import HENRYS_AMOEBA_PATH
# import sys
# sys.path.append(HENRYS_AMOEBA_PATH)
from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Classes.magnification_map import MagnificationMap as AmoebaMagnificationMap
import amoeba.Util.util as util

from magmap import MagnificationMap

import numpy as np
import astropy.constants as const
from astropy import units as u

class MicrolensingLightCurve(object):
    """Class to generate lightcurves based on the magnification maps and the source."""

    def __init__(self, magnification_map:MagnificationMap, ):
        """
        :param magnification_map: MagnificationMap object
        :param image_observing_time: time at which the image is observed
        """
        self.magnification_map = magnification_map
        self.image_observing_time = image_observing_time

    def generate_lightcurve_magnitude(self, time_array, lens_class):
        """
        Generate lightcurve based on the source type
        """
        pass

    def _generate_agn_lightcurve(self, source_class, lens_class):
        """
        Generate lightcurve for a quasar(AGN) with the accretion disk model from amoeba
        """
        mag_map_2d = self.magnification_map.magnifications
        z_l = lens_class.deflector.redshift

        # Disk parameters
        lamp_height = 10
        x = np.linspace(-2000, 2000, 2001)
        y = np.linspace(-2000, 2000, 2001)
        X, Y = np.meshgrid(x, y)
        radii, phi_array = util.convert_cartesian_to_polar(X, Y)
        mass_kg = 10**8.0 * const.M_sun.to(u.kg)
        mass_sm = 10**8.0
        grav_rad = util.calculate_gravitational_radius(mass_sm)
        temp_map = util.accretion_disk_temperature(
            radii * grav_rad, 6 * grav_rad, mass_sm, 0.15
        )

        Disk = AccretionDisk(
            smbh_mass_exp=8.0,
            redshift_source=source_class.redshift,
            inclination_angle=0,
            corona_height=lamp_height,
            temp_array=temp_map,
            phi_array=phi_array,
            g_array=np.ones(np.shape(temp_map)),
            radii_array=radii,
            height_array=np.zeros(np.shape(radii)),
        )

        disk_projection = Disk.calculate_surface_intensity_map(600)

        MagMap = MagnificationMap(
            z_s,
            z_l,
            mag_map_2d,
            0.6,
            0.7,
        )


        convolution = MagMap.convolve_with_flux_projection(disk_projection)
        # convolved_flux_map = convolution.magnification_array
        # convolutions_each_quasar.append(convolution)

    def _generate_supernova_lightcurve(self):
        """
        Generate lightcurve for a supernova
        """
        pass