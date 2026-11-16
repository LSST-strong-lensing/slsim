import numpy as np
from astropy import units as u
from astropy.table import Table
from lenstronomy.Util import constants

from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourcePopulation.galaxies import convert_catalog_to_source
from slsim.Deflectors.deflector import Deflector
from slsim.Deflectors import deflector_util
from slsim.Util.cosmo_util import get_cosmo_cache
from slsim.Util.lenstronomy_util import theta_E_spherical_nfw_hernquist
from slsim.Util.param_util import (
    galaxy_size_redshift_evolution,
    vel_disp_from_m_star,
)


class DeflectorsBase(Galaxies):
    """Abstract Base Class to create a class that accesses a set of deflectors.

    All object that inherit from Lensed System must contain the methods
    it contains.
    """

    def __init__(
        self,
        deflector_table,
        kwargs_cut,
        cosmo,
        sky_area,
        gamma_pl=None,
        mass_type="EPL",
        light_type="single_sersic",
        kwargs_mass2light=None,
        catalog_type=None,
        sis_convention=True,
    ):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :param mass_type: type of Deflector() mass model class
        :type mass_type: string
        :param light_type: type of Source() model class for the light distribution
        :type light_type: string
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :param kwargs_mass2light: mass-to-light relation
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        :param sis_convention: if True, the Einstein radius of an EPL deflector is
         normalized with the SIS convention. Only relevant for mass_type="EPL" with
         gamma_pl != 2. The non-SIS normalization solves the spherical Jeans equation
         for every deflector and is orders of magnitude slower.
        :type sis_convention: bool
        """
        super().__init__(
            galaxy_list=deflector_table,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut=kwargs_cut,
            catalog_type=catalog_type,
            size_model=None,
            extended_source_type=light_type,
            downsample_to_dc2=False,
            extended_source_kwargs=None,
        )

        self.mass_type = mass_type
        if kwargs_mass2light is None:
            kwargs_mass2light = {}
        self._kwargs_mass2light = kwargs_mass2light
        self._gamma_pl = gamma_pl
        self._sis_convention = sis_convention
        self._vel_disp_from_stellar_mass = None
        # Will be overwriten by interpolation function deriving velocity dispersion from stellar mass

    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        return self.source_number_selected

    def draw_deflector(self, z_max=None, z_min=None, deflector_index=None):
        """

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param deflector_index: index of galaxy to pic (if provided)
        :return: dictionary of complete parameterization of a deflector
        """

        halo_gal = self.draw_object(
            z_max=z_max, z_min=z_min, galaxy_index=deflector_index
        )
        return self._deflector_from_object(halo_gal)

    def _deflector_from_object(self, halo_gal):
        """Instantiate a ``Deflector`` from a single catalog entry."""
        kwargs_source = convert_catalog_to_source(
            galaxy=halo_gal,
            extended_source_type=self._extended_source_type,
            catalog_type=self._catalog_type,
            size_model=self._size_model,
            cosmo=self._cosmo,
            include_all_keywords=False,
        )
        kwargs_mass = deflector_util.light2mass(
            kwargs_source,
            halo_dict=halo_gal,
            mass_type=self.mass_type,
            **self._kwargs_mass2light,
        )
        kwargs_mass = self._update_mass(
            kwargs_mass=kwargs_mass, kwargs_source=kwargs_source
        )
        z = kwargs_source.pop("z")
        return Deflector(z=z, kwargs_mass=kwargs_mass, kwargs_light=kwargs_source)

    def _draw_deflector_at_selected(self, selected_index):
        """Instantiate a ``Deflector`` from an index in ``_objects_select``."""
        return self._deflector_from_object(self._draw_selected_object(selected_index))

    def _epl_theta_e_infinity(self, selected_indices):
        """Compute vectorized EPL theta_e_infinity, or return None on
        fallback."""
        colnames = self._objects_select.colnames
        if "vel_disp" in colnames:
            v_sigma = np.asarray(
                self._objects_select["vel_disp"][selected_indices], dtype=float
            )
        else:
            use_interp = self._vel_disp_from_stellar_mass is not None
            use_scaling = bool(
                self._kwargs_mass2light.get("m_star_v_disp_scaling", False)
            )
            if not (use_interp or use_scaling) or "stellar_mass" not in colnames:
                return None
            stellar_mass = self._extract_positive_array(
                "stellar_mass", selected_indices
            )
            if stellar_mass is None:
                return None
            v_sigma = (
                np.asarray(
                    self._vel_disp_from_stellar_mass(np.log10(stellar_mass)),
                    dtype=float,
                )
                if use_interp
                else np.asarray(vel_disp_from_m_star(stellar_mass), dtype=float)
            )
        if not np.all(np.isfinite(v_sigma)) or np.any(v_sigma < 0.0):
            return None
        return 4.0 * np.pi * (v_sigma * 1000.0 / constants.c) ** 2 / constants.arcsec

    def _extract_positive_array(self, colname, selected_indices, unit=None):
        """Extract column slice as positive finite float array, else None."""
        if colname not in self._objects_select.colnames:
            return None
        col = self._objects_select[colname][selected_indices]
        arr = (
            col.to_value(unit)
            if unit is not None and isinstance(col, u.Quantity)
            else np.asarray(col, dtype=float)
        )
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            return None
        return arr

    def _nfw_angular_params(
        self, selected_indices, cosmo_cache, z_arr, sigma_crit, dd_l
    ):
        """Return vectorized (Rs_nfw, alpha_Rs) at z_s=100, or None on
        fallback."""
        halo_mass = self._extract_positive_array("halo_mass", selected_indices)
        concentration = self._extract_positive_array("concentration", selected_indices)
        if halo_mass is None or concentration is None:
            return None
        h = float(cosmo_cache.cosmo.H(0).value) / 100.0
        efunc_l = cosmo_cache.efunc(z_arr)
        rhoc_z = 2.77536627e11 * efunc_l**2
        r200 = (3.0 * halo_mass * h / (4.0 * np.pi * rhoc_z * 200.0)) ** (1.0 / 3.0) / h
        rho0 = (
            200.0
            / 3.0
            * rhoc_z
            * concentration**3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
            * h**2
        )
        rs_phys = r200 / concentration
        rs_nfw = rs_phys / dd_l / constants.arcsec
        alpha_rs = (
            rho0
            * (4.0 * rs_phys**2 * (1.0 + np.log(0.5)))
            / (sigma_crit * dd_l * constants.arcsec)
        )
        return rs_nfw, alpha_rs

    def _hernquist_angular_params(self, selected_indices, z_arr, sigma_crit, dd_l):
        """Return vectorized (Rs_hern, sigma0) at z_s=100, or None on
        fallback."""
        if self._extended_source_type not in (
            "single_sersic",
            "hernquist",
            "catalog_source",
        ) or (self._size_model == "Bernardi" and self._catalog_type == "skypy"):
            return None
        stellar_mass = self._extract_positive_array("stellar_mass", selected_indices)
        if stellar_mass is None:
            return None

        ang_size = self._extract_positive_array(
            "angular_size", selected_indices, unit=u.arcsec
        )
        if ang_size is None:
            phys_kpc = self._extract_positive_array(
                "physical_size", selected_indices, unit=u.kpc
            )
            if phys_kpc is None:
                return None
            ang_size = (phys_kpc / (dd_l * 1000.0)) / constants.arcsec
        if self._catalog_type == "skypy":
            ang_size = ang_size * (galaxy_size_redshift_evolution(z_arr) / 2.5)
        rs_hern = ang_size / 1.815
        rs_phys = dd_l * rs_hern * constants.arcsec
        sigma0 = (stellar_mass / (2.0 * np.pi * rs_phys**2)) / sigma_crit
        return rs_hern, sigma0

    def _theta_e_infinity_for_selected(self, selected_indices, cosmo):
        """Compute vectorized theta_e_infinity [arcsec] for selected indices.

        Returns None if the deflector mass model or table columns
        require the per-deflector generic path.
        """
        if not getattr(self, "_astropy_table", False) or not isinstance(
            self._objects_select, Table
        ):
            return None
        if self.mass_type not in ("EPL", "NFW", "HERNQUIST", "NFW_HERNQUIST"):
            return None
        idx = np.asarray(selected_indices, dtype=int)
        if idx.size == 0:
            return np.empty(0, dtype=float)
        z_arr = self._extract_positive_array("z", idx)
        if z_arr is None:
            return None
        if self.mass_type == "EPL":
            return self._epl_theta_e_infinity(idx)

        cosmo_cache = get_cosmo_cache(cosmo)
        dd_l, ds_100, dds_100 = cosmo_cache.dd_ds_dds_at_z100(z_arr)
        const_factor = (constants.c**2 / (4.0 * np.pi * constants.G)) * (
            constants.Mpc / constants.M_sun
        )
        sigma_crit = (ds_100 / (dd_l * dds_100)) * const_factor

        rs_nfw = alpha_rs = None
        if self.mass_type in ("NFW", "NFW_HERNQUIST"):
            nfw_params = self._nfw_angular_params(
                idx, cosmo_cache, z_arr, sigma_crit, dd_l
            )
            if nfw_params is None:
                return None
            rs_nfw, alpha_rs = nfw_params

        rs_hern = sigma0 = None
        if self.mass_type in ("HERNQUIST", "NFW_HERNQUIST"):
            hern_params = self._hernquist_angular_params(idx, z_arr, sigma_crit, dd_l)
            if hern_params is None:
                return None
            rs_hern, sigma0 = hern_params

        return theta_E_spherical_nfw_hernquist(
            rs_nfw=rs_nfw, alpha_rs=alpha_rs, rs_hern=rs_hern, sigma0=sigma0
        )

    def _update_mass(self, kwargs_mass, kwargs_source):
        """Additional updates on mass.

        :param kwargs_mass:
        :param kwargs_source: dictionary matching the Source() input
            class
        :return:
        """
        if self.mass_type in ["EPL"]:
            if self._gamma_pl is not None:
                kwargs_mass["gamma_pl"] = _gamma_pl(self._gamma_pl)
            kwargs_mass["sis_convention"] = self._sis_convention
        if (
            "vel_disp" not in kwargs_mass
            and "stellar_mass" in kwargs_source
            and self._vel_disp_from_stellar_mass is not None
        ):
            kwargs_mass["vel_disp"] = self._vel_disp_from_stellar_mass(
                np.log10(kwargs_source["stellar_mass"])
            )
        return kwargs_mass


def _gamma_pl(gamma_pl):
    """Madd density power-law slope (2 = isothermal)

    :param gamma_pl:
    :return:
    """
    if gamma_pl is not None:
        if isinstance(gamma_pl, float):
            return gamma_pl
        elif isinstance(gamma_pl, dict):
            parameters = gamma_pl.keys()
            if "mean" in parameters and "std_dev" in parameters:
                return np.random.normal(loc=gamma_pl["mean"], scale=gamma_pl["std_dev"])
            elif "gamma_min" in parameters and "gamma_max" in parameters:
                return np.random.uniform(
                    low=gamma_pl["gamma_min"],
                    high=gamma_pl["gamma_max"],
                )
            else:
                raise ValueError(
                    "The given quantities in gamma_pl are not recognized."
                    " Please provide the mean and standard deviation for a"
                    " gaussian distribution, or specify the gamma_min and gamma_max "
                    " for a uniform distribution."
                )
        else:
            raise ValueError(
                "The given format of the gamma_pl is not supported."
                " Please provide a float or dictionary. See the documentation"
                " in DeflectorsBase class"
            )
    return 2
