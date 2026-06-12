import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

class SLSimLogo(object):
        """Generates the SLSim logo.

        :param rubin_color: color of the logo text and marker dots
        :param arc_color: color of the strong-lensing arcs
        :param galaxy_color: color of the central galaxy
        :param symbol_scale: overall scaling applied to the lensing symbol
        :type symbol_scale: float
        :param symbol_rotation: rotation angle of the lensing symbol in degrees
        :type symbol_rotation: float
        :param flip_x: mirror the lensing symbol about the y-axis
        :type flip_x: bool
        :param fontsize: font size of the logo text
        :type fontsize: float
        :param galaxy_ellipticity: ellipticity of the source galaxy
        :type galaxy_ellipticity: float
        :param galaxy_angle: major-axis position angle of the galaxy in degrees
        :type galaxy_angle: float
        :param galaxy_seed: random seed used for galaxy realization
        :type galaxy_seed: int
        """

        def __init__(
            self,
            rubin_color="#58B4B8",
            arc_color="#00D9FF",
            galaxy_color="#4B2E83",
            symbol_scale=1.1,
            symbol_rotation=-30,
            flip_x=False,
            fontsize=170,
            galaxy_ellipticity=0.4,
            galaxy_angle=150,
            galaxy_seed=3,
        ):

            self.rubin_color = rubin_color
            self.arc_color = arc_color
            self.galaxy_color = galaxy_color

            self.symbol_scale = symbol_scale
            self.symbol_rotation = symbol_rotation
            self.flip_x = flip_x

            self.fontsize = fontsize

            self.galaxy_ellipticity = galaxy_ellipticity
            self.galaxy_angle = galaxy_angle
            self.galaxy_seed = galaxy_seed

        def _transform(
            self,
            x,
            y,
            scale=1.0,
            rotation_deg=0.0,
            flip_x=False,
        ):
            """Apply scaling, reflection, and rotation.

            :param x: x coordinates
            :param y: y coordinates
            :param scale: scaling factor
            :type scale: float
            :param rotation_deg: rotation angle in degrees
            :type rotation_deg: float
            :param flip_x: mirror coordinates about the y-axis
            :type flip_x: bool
            :return: transformed coordinates
            """

            x = np.asarray(x) * scale
            y = np.asarray(y) * scale

            if flip_x:
                x = -x

            theta = np.deg2rad(rotation_deg)

            xr = x * np.cos(theta) - y * np.sin(theta)
            yr = x * np.sin(theta) + y * np.cos(theta)

            return xr, yr