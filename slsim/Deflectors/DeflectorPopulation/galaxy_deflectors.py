import numpy as np
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from slsim.Util.color_gradient import attach_foreground_deflector_color_gradient
from astropy.table import vstack


class GalaxyDeflectors(DeflectorsBase):
    """Class describing all-type galaxies.

    Abundance matching of velocity dispersion only being applied to the
    red galaxies, but retrieved scaling between stellar mass and
    velocity dispersion is being applied to all galaxies.
    """

    def __init__(
        self,
        red_galaxy_list,
        kwargs_mass2light,
        cosmo,
        sky_area,
        blue_galaxy_list=None,
        kwargs_cut=None,
        gamma_pl=None,
        catalog_type="skypy",
        mass_type="EPL",
        light_type="single_sersic",
        foreground_color_gradient=None,
        foreground_component_weights=(0.4, 0.6),
    ):
        """
        :param red_galaxy_list: list of dictionary with elliptical galaxy
            parameters (supporting skypy pipelines)
        :type red_galaxy_list: astropy.Table
        :param blue_galaxy_list: list of dictionary with spiral galaxy
            parameters (supporting skypy pipelines)
        :type blue_galaxy_list: astropy.Table
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :param mass_type: type of Deflector() mass model class
        :type mass_type: string
        :param light_type: type of Source() model class for the light distribution
        :type light_type: string
        :param foreground_color_gradient: Optional configuration for a
         band-dependent two-component deflector light profile. When provided,
         the light is handled by the standard ``Source``/``DoubleSersic`` path.
        :type foreground_color_gradient: dict or None
        :param foreground_component_weights: Reference-band flux weights of the
         two Sersic components.
        :type foreground_component_weights: tuple or list
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        :type catalog_type: str. "skypy" or None.
        """
        if foreground_color_gradient is not None:
            if light_type not in ("single_sersic", "double_sersic"):
                raise ValueError(
                    "foreground_color_gradient requires a single_sersic or "
                    "double_sersic light_type."
                )
            light_type = "double_sersic"
            red_galaxy_list = red_galaxy_list.copy()
            attach_foreground_deflector_color_gradient(
                red_galaxy_list,
                foreground_color_gradient,
                component_weights=foreground_component_weights,
            )
            if blue_galaxy_list is not None:
                blue_galaxy_list = blue_galaxy_list.copy()
                attach_foreground_deflector_color_gradient(
                    blue_galaxy_list,
                    foreground_color_gradient,
                    component_weights=foreground_component_weights,
                )

        red_column_names = red_galaxy_list.colnames
        if "galaxy_type" not in red_column_names:
            red_galaxy_list["galaxy_type"] = "red"
        if blue_galaxy_list is not None:
            blue_column_names = blue_galaxy_list.colnames
            if "galaxy_type" not in blue_column_names:
                blue_galaxy_list["galaxy_type"] = "blue"

            galaxy_list = vstack([red_galaxy_list, blue_galaxy_list])
        else:
            galaxy_list = red_galaxy_list
        # Abundance matching with the SDSS velocity dispersion function matching the red galaxies
        if "vel_disp" not in red_column_names:
            self._f_vel_disp = vel_disp_abundance_matching(
                red_galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
            )
            galaxy_list["vel_disp"] = self._f_vel_disp(
                np.log10(galaxy_list["stellar_mass"])
            )

        super().__init__(
            deflector_table=galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=gamma_pl,
            kwargs_mass2light=kwargs_mass2light,
            catalog_type=catalog_type,
            mass_type=mass_type,
            light_type=light_type,
        )
