import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Deflectors.velocity_dispersion import vel_disp_composite_model
from slsim.Util import param_util
from slsim.Deflectors.deflectors_base import DeflectorsBase
from lenstronomy.Util import constants

class CompoundLensHalosGalaxies(DeflectorsBase):
    """Class describing compound lens model based on the halo model."""

    def __init__(self, halo_galaxy_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area):
        """

        :param halo_galaxy_list: list of dictionary with lens parameters of
            elliptical dark matte haloes and galaxies (currently supporting SL-Hammocks pipelines)
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        # :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        ## MEMO: DeflectorsBase's inputs are deflector_table, kwargs_cut, cosmo, sky_area
        """
        super().__init__(
            deflector_table=halo_galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )

        n = len(halo_galaxy_list)
        column_names = halo_galaxy_list.columns
        if "vel_disp" not in column_names:
            halo_galaxy_list["vel_disp"] = -np.ones(n)
        if "mag_g" not in column_names:
            halo_galaxy_list["mag_g"] = -np.ones(n)
        if "mag_r" not in column_names:
            halo_galaxy_list["mag_r"] = -np.ones(n)
        if "mag_i" not in column_names:
            halo_galaxy_list["mag_i"] = -np.ones(n)
        if "mag_z" not in column_names:
            halo_galaxy_list["mag_z"] = -np.ones(n)
        if "mag_Y" not in column_names:
            halo_galaxy_list["mag_Y"] = -np.ones(n)
        if "e1_light" not in column_names or "e2_light" not in column_names:
            halo_galaxy_list["e1_light"] = -np.ones(n)
            halo_galaxy_list["e2_light"] = -np.ones(n)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            halo_galaxy_list["e1_mass"] = -np.ones(n)
            halo_galaxy_list["e2_mass"] = -np.ones(n)

        self._galaxy_select = deflector_cut(halo_galaxy_list, **kwargs_cut)
        # Currently only supporting redshift cut
        self._num_select = len(self._galaxy_select)

        self._cosmo = cosmo

        # TODO: random reshuffle of matched list

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self):

        """
        :return: dictionary of complete parameterization of deflector
        """

        cosmo = self._cosmo
        index = random.randint(0, self._num_select - 1)
        deflector = self._galaxy_select[index]
        if deflector["vel_disp"] == -1:
            theta_eff = deflector['tb']/0.551 # [arcsec]
            reff = (theta_eff*cosmo.angular_diameter_distance(deflector['z'])*constants.arcsec).value  # physical Mpc
            hubble = cosmo.H0.value/100.
            vel_disp = vel_disp_composite_model(
                theta_eff, deflector['stellar_mass']/(hubble), reff, max(deflector['halo_mass'],deflector['halo_mass_acc'])/(hubble),deflector['concentration'], cosmo,  deflector['z'])
            deflector["vel_disp"] = vel_disp
        if deflector["mag_g"] == -1 or deflector["mag_r"] or  deflector["mag_i"] == -1 or deflector["mag_z"] or deflector["mag_Y"] == -1:
            mag_g, mag_r, mag_i, mag_z, mag_Y = 0,0,0,0,0 # TODO: make function if needed
        if deflector["e1_light"] == -1 or deflector["e2_light"] == -1:
            e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity_galaxy(
                **deflector
            ) # TODO: check
            deflector["e1_light"] = e1_light
            deflector["e2_light"] = e2_light
            deflector["e1_mass"] = e1_mass
            deflector["e2_mass"] = e2_mass
        return deflector


def elliptical_projected_eccentricity_galaxy(e_g, **kwargs):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param e_g: eccentricity amplitude of galaxy
    :type e_g: float [0,1)
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    ellipticity = e_g
    e_light = param_util.epsilon2e(ellipticity)
    phi_light = np.random.uniform(0, np.pi)
    e1_light = e_light * np.cos(phi_light)
    e2_light = e_light * np.sin(phi_light)
    e_mass = 0.5 * ellipticity + np.random.normal(loc=0, scale=0.1)
    phi_mass = phi_light + np.random.normal(loc=0, scale=0.1)
    e1_mass = e_mass * np.cos(phi_mass)
    e2_mass = e_mass * np.sin(phi_mass)
    return e1_light, e2_light, e1_mass, e2_mass

