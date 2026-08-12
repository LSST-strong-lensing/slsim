from slsim.Sources.SourcePopulation.galaxies import convert_catalog_to_source
from slsim.Sources.SourcePopulation.galaxies import galaxy_projected_eccentricity
from slsim.Deflectors.MassLightConnection.richness2mass import mass_richness_relation
from slsim.Halos.halo_population import concent_m_w_scatter
from slsim.Deflectors.deflector import Deflector
from slsim.Util import param_util
import numpy as np
from colossus.cosmology import cosmology as colossus_cosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


def deflector_from_table(table, mass_type, extended_source_type, cosmo=None):
    """Create a Deflector() instance from a single table combining the light
    and mass arguments.

    :param table: table of all the parameters of mass and light
    :param mass_type: type of mass model for Mass() class
    :param extended_source_type: type of light model for Source() class
    :return: Deflector() instance
    :rtype: ~slsim.Deflectors.deflector.Deflector() instance
    """
    z, center_x, center_y, kwargs_mass, kwargs_light = deflector_dict_from_table(
        table=table,
        mass_type=mass_type,
        extended_source_type=extended_source_type,
        cosmo=cosmo,
    )
    deflector = Deflector(
        z=z,
        kwargs_mass=kwargs_mass,
        kwargs_light=kwargs_light,
        center_x=center_x,
        center_y=center_y,
    )
    return deflector


def deflector_dict_from_table(
    table, mass_type, extended_source_type, cosmo=None, **kwargs_mass2light
):
    """Dictionaries to create a Deflector() instance from a single table
    combining the light and mass arguments.

    :param table: table of all the parameters of mass and light
    :param mass_type: type of mass model for Mass() class
    :param cosmo: astropy cosmology
    :param extended_source_type: type of light model for Source() class
    :return: z, center_x, center_y, kwargs_mass, kwargs_light
    """
    kwargs_light = convert_catalog_to_source(
        table,
        extended_source_type,
        catalog_type=None,
        size_model=None,
        cosmo=cosmo,
        include_all_keywords=False,
    )

    kwargs_mass = light2mass(
        kwargs_light,
        mass_type,
        **kwargs_mass2light,
        halo_dict=table,
    )
    if isinstance(table, dict):
        colnames = list(table.keys())
    else:
        colnames = table.colnames

    if "center_x" in colnames and "center_y" in colnames:
        center_x = table["center_x"]
        center_y = table["center_y"]
    else:
        center_x, center_y = 0, 0
    z = kwargs_light.pop("z")
    return z, center_x, center_y, kwargs_mass, kwargs_light


def light2mass(
    kwargs_source,
    mass_type,
    light2mass_e_scaling=1,
    light2mass_e_scatter=0.1,
    halo_dict=None,
    m_star_v_disp_scaling=False,
    richness_fn="Abdullah2022",
):
    """

    :param kwargs_source: dictionary for Source() class
    :param light2mass_e_scaling: scaling factor of mass eccentricity /
    light eccentricity
    :param mass_type: type of mass profile
    :param light2mass_e_scatter: scatter in light and mass
        eccentricities from the scaling relation
    :param light2mass_e_scatter: scatter in eccentricity strength
        between light and mass eccentricity
    :param halo_dict: dictionary of halo
    :type halo_dict: entry of object to act as deflector
    :param m_star_v_disp_scaling: applying M_star - velocity dispersion scaling relation to get velocity dispersion
     from stellar mass. Is used when stellar_mass is provided but no vel_disp
    :param richness_fn: richness-mass relation to assign a mass to each cluster
    :type richness_fn: str
    :return: dictionary for Mass() class
    """
    if halo_dict is not None:
        if isinstance(halo_dict, dict):
            halo_columns = list(halo_dict.keys())
        else:
            halo_columns = halo_dict.colnames
    else:
        halo_columns = []
    kwargs_mass = {"mass_type": mass_type}
    if mass_type in ["EPL", "NFW", "NFW_HERNQUIST"]:

        if "e1_mass" in halo_columns and "e2_mass" in halo_columns:
            e1_mass, e2_mass = float(halo_dict["e1_mass"]), float(halo_dict["e2_mass"])
        elif "e_h" in halo_columns and "p_h" in halo_columns:
            # SL_hammock conventions
            e1_mass, e2_mass = galaxy_projected_eccentricity(
                float(halo_dict["e_h"]), rotation_angle=np.deg2rad(halo_dict["p_h"])
            )
        else:
            # scale light to mass ellipticity
            e1_light, e2_light = kwargs_source["e1"], kwargs_source["e2"]
            e1_mass, e2_mass = (
                light2mass_e_scaling * e1_light,
                light2mass_e_scaling * e2_light,
            )
            # add scatter in mass
            e_mass_scatter = np.random.normal(loc=0, scale=light2mass_e_scatter)
            phi_scatter = np.random.uniform(0, np.pi)
            e1_mass += e_mass_scatter * np.cos(2 * phi_scatter)
            e2_mass += e_mass_scatter * np.sin(2 * phi_scatter)
        kwargs_mass["e1"] = e1_mass
        kwargs_mass["e2"] = e2_mass

    if mass_type in ["EPL"]:
        if "gamma_pl" in halo_columns:
            kwargs_mass["gamma_pl"] = float(halo_dict["gamma_pl"])
        else:
            kwargs_mass["gamma_pl"] = 2
        if "theta_E" in halo_columns:
            kwargs_mass["theta_E"] = halo_dict["theta_E"]
    if "vel_disp" in kwargs_source:
        kwargs_mass["vel_disp"] = kwargs_source["vel_disp"]
    elif m_star_v_disp_scaling and "stellar_mass" in kwargs_source:
        kwargs_mass["vel_disp"] = param_util.vel_disp_from_m_star(
            kwargs_source["stellar_mass"]
        )
    if mass_type in ["NFW", "NFW_HERNQUIST"]:
        if halo_dict is None:
            raise ValueError(
                "halo_dict needs to be provided for mass_type %s" % mass_type
            )
        if "halo_mass" not in halo_columns:
            if "richness" not in halo_columns:
                raise ValueError(
                    "Either 'halo_mass' or 'richness' needs to be in 'halo_dict' for "
                    "mass_type %s." % mass_type
                )
            halo_mass = mass_richness_relation(
                float(halo_dict["richness"]), richness_fn
            )
        else:
            halo_mass = float(halo_dict["halo_mass"])
        kwargs_mass["halo_mass"] = halo_mass
        if "concentration" not in halo_columns:
            concentration = concent_m_w_scatter(
                np.array([halo_mass]), kwargs_source["z"], sig=0.33
            )[0]
        else:
            concentration = float(halo_dict["concentration"])
        kwargs_mass["concentration"] = concentration

    return kwargs_mass


def set_colossus_cosmo(cosmo):
    """Set the cosmology in colossus to match the astropy.cosmology instance.

    :param cosmo: astropy cosmology instance
    """
    params = dict(
        flat=(cosmo.Ok0 == 0.0),
        H0=cosmo.H0.value,
        Om0=cosmo.Om0,
        Ode0=cosmo.Ode0,
        Ob0=(cosmo.Ob0 if (cosmo.Ob0 is not None) and (cosmo.Ob0 != 0) else 0.04897),
        Tcmb0=cosmo.Tcmb0.value if cosmo.Tcmb0.value > 0 else 2.7255,
        Neff=cosmo.Neff,
        sigma8=0.8102,
        ns=0.9660499,
    )
    return colossus_cosmo.setCosmology(cosmo_name="halo_cosmo", **params)


def critical_curves_caustics_list(
    deflector, z_source, cosmo, kwargs_critical_curve_caustics=None
):
    """Returns list of critical curves and caustics for a source at `z_source`

    :param deflector: `DeflectorGroup` or `Deflector` object
    :param z_source: redshift at which to compute curves
    :param cosmo: astropy.cosmology instance
    :param kwargs_critical_curve_caustics: arguments passed into the `critical_curve_caustics` function from LensModelExtensions. Keys:
    - compute_window: window size in arcsec where the critical curve is computed
    - grid_scale: numerical grid spacing of the computation of the critical curves
    - center_x: float, center of the window to compute critical curves and caustics
    - center_y: float, center of the window to compute critical curves and caustics
    - kwargs_lens: lens model kwargs
    """

    lens_cosmo = LensCosmo(
        z_lens=deflector.redshift,
        z_source=z_source,
        cosmo=cosmo,
    )

    lens_mass_model_list, model_params = deflector.mass_model_lenstronomy(lens_cosmo)

    lens_model = LensModel(
        lens_model_list=lens_mass_model_list,
        cosmo=cosmo,
        z_lens=deflector.redshift,
        z_source=z_source,
        multi_plane=False,
    )

    lens_model_ext = LensModelExtensions(lens_model)
    ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = (
        lens_model_ext.critical_curve_caustics(
            model_params, **kwargs_critical_curve_caustics
        )
    )

    return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list
