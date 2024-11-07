import numpy as np
import numpy.random as random
from slsim.selection import object_cut
from slsim.Util import param_util
from slsim.Deflectors.deflectors_base import DeflectorsBase
from slsim.Deflectors.velocity_dispersion import vel_disp_abundance_matching


class EllipticalLensGalaxies(DeflectorsBase):
    """Class describing elliptical galaxies."""

    def __init__(
        self,
        galaxy_list,
        kwargs_cut,
        kwargs_mass2light,
        cosmo,
        sky_area,
        gamma_pl=None,
        catalog_type="skypy",
    ):
        """

        :param galaxy_list: list of dictionary with galaxy parameters of
            elliptical galaxies (currently supporting skypy pipelines)
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation 
         of a density slope for gaussian distribution or minimum and maximum values of 
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :type catalog_type: str. "skypy" or None.
        """
        galaxy_list = param_util.catalog_with_angular_size_in_arcsec(
            galaxy_catalog=galaxy_list, input_catalog_type=catalog_type
        )
        super().__init__(
            deflector_table=galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=gamma_pl
        )

        n = len(galaxy_list)
        column_names = galaxy_list.colnames
        if "vel_disp" not in column_names:
            galaxy_list["vel_disp"] = -np.ones(n)
        if "e1_light" not in column_names or "e2_light" not in column_names:
            galaxy_list["e1_light"] = -np.ones(n)
            galaxy_list["e2_light"] = -np.ones(n)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            galaxy_list["e1_mass"] = -np.ones(n)
            galaxy_list["e2_mass"] = -np.ones(n)
        if "n_sersic" not in column_names:
            galaxy_list["n_sersic"] = -np.ones(n)

        self._f_vel_disp = vel_disp_abundance_matching(
            galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
        )

        self._galaxy_select = object_cut(galaxy_list, **kwargs_cut)
        self._num_select = len(self._galaxy_select)
        self._galaxy_select["vel_disp"] = self._f_vel_disp(
            np.log10(self._galaxy_select["stellar_mass"])
        )

        self._kwargs_mass2light = kwargs_mass2light

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

        index = random.randint(0, self._num_select - 1)
        deflector = self._galaxy_select[index]
        if deflector["vel_disp"] == -1:
            stellar_mass = deflector["stellar_mass"]
            vel_disp = vel_disp_from_m_star(stellar_mass)
            deflector["vel_disp"] = vel_disp
        if deflector["e1_light"] == -1 or deflector["e2_light"] == -1:
            e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity(
                **deflector, **self._kwargs_mass2light
            )
            deflector["e1_light"] = e1_light
            deflector["e2_light"] = e2_light
            deflector["e1_mass"] = e1_mass
            deflector["e2_mass"] = e2_mass
        if deflector["n_sersic"] == -1:
            deflector["n_sersic"] = 4  # TODO make a better estimate with scatter
        return deflector


def elliptical_projected_eccentricity(
    ellipticity,
    light2mass_e_scaling=1,
    light2mass_e_scatter=0.1,
    light2mass_angle_scatter=0.1,
    **kwargs
):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude (1-q^2)/(1+q^2)
    :type ellipticity: float [0,1)
    :param light2mass_e_scaling: scaling factor of mass eccentricity / light
        eccentricity
    :param light2mass_e_scatter: scatter in light and mass eccentricities from the
        scaling relation
    :param light2mass_angle_scatter: scatter in orientation angle between light and mass
        eccentricity
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    e_light = param_util.epsilon2e(ellipticity)
    phi_light = np.random.uniform(0, np.pi)
    e1_light = e_light * np.cos(2 * phi_light)
    e2_light = e_light * np.sin(2 * phi_light)
    e_mass = light2mass_e_scaling * ellipticity + np.random.normal(
        loc=0, scale=light2mass_e_scatter
    )
    phi_mass = phi_light + np.random.normal(loc=0, scale=light2mass_angle_scatter)
    e1_mass = e_mass * np.cos(2 * phi_mass)
    e2_mass = e_mass * np.sin(2 * phi_mass)
    return e1_light, e2_light, e1_mass, e2_mass


def vel_disp_from_m_star(m_star):
    """Function to calculate the velocity dispersion from the staller mass using
    empirical relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::

         V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
         M_\\odot} \\right)^{0.24}

    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    :param m_star: stellar mass in the unit of solar mass
    :return: the velocity dispersion ("km/s")
    """
    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)
    return v_disp
