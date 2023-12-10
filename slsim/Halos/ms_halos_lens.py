from lenstronomy.Util.util import make_grid
import numpy as np
import astropy.units as u
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from slsim.Halos.halos_lens import concentration_from_mass, cone_radius_angle_to_physical_area, deg2_to_cone_angle
import warnings
from tqdm.notebook import tqdm
import math
import time
import multiprocessing
from collections.abc import Iterable
import matplotlib.pyplot as plt


class HalosMSLens(object):
    """Manage lensing properties of Halos from Millennium Simulation.
    """

    def __init__(
            self,
            halos_list,
            cosmo=None,
            sky_area=0.00082,
            z_source=5,
    ):
        """
        """
        self.z_source = z_source
        self.halos_list = halos_list[halos_list['z'] <= z_source]
        self.n_halos = len(self.halos_list)
        self.sky_area = sky_area
        self.halos_redshift_list = self.halos_list["z"]
        self.mass_list = self.halos_list["mass"]
        self._z_source_convention = (
            5  # if this need to be changed
        )
        if cosmo is None:
            warnings.warn(
                "No cosmology provided, instead uses astropy.cosmology import default_cosmology"
            )
            import astropy.cosmology as default_cosmology
            # todo: find the cosmology for Millennium Simulation
            self.cosmo = default_cosmology.get()
        else:
            self.cosmo = cosmo

        self._lens_cosmo = None  # place-holder for lazy load
        self._lens_model = None  # same as above
        c_200 = [concentration_from_mass(z=zi, mass=mi)
                 for zi, mi in zip(self.halos_redshift_list, self.mass_list)]
        self.halos_list['c_200'] = c_200

        # TODO: Note that the los_correction is under construction

    @property
    def lens_cosmo(self):
        """Lazy-load lens_cosmo."""
        if self._lens_cosmo is None:
            self._lens_cosmo = [
                LensCosmo(
                    z_lens=self.halos_redshift_list[h],
                    z_source=self.z_source,
                    cosmo=self.cosmo,
                )
                for h in range(self.n_halos)
            ]
        return self._lens_cosmo

    @property
    def lens_model(self):
        """Lazy-load lens_model."""
        if self._lens_model is None:  # Only compute if not already done
            self._lens_model = self.get_lens_model()
        return self._lens_model

    def get_lens_model(self):
        """
        """
        lens_model = LensModel(
            lens_model_list=["NFW"] * self.n_halos,
            lens_redshift_list=self.halos_redshift_list,
            cosmo=self.cosmo,
            observed_convention_index=[],
            multi_plane=True,
            z_source=self.z_source,
            z_source_convention=self._z_source_convention,
        )
        return lens_model

    def get_nfw_kwargs(self, z=None, mass=None, n_halos=None, lens_cosmo=None, c=None):
        """Returns the angle at scale radius, observed bending angle at the
        scale radius, and positions of the Halos in the lens plane from
        physical mass and concentration parameter of an NFW profile.

        Returns
        -------
        Rs_angle, alpha_Rs, px, py : np.array
            Rs_angle (angle at scale radius) (in units of arcsec)
            alpha_Rs (observed bending angle at the scale radius) (in units of arcsec)
            Arrays containing Rs_angle, alpha_Rs, and x and y positions of all the Halos.
        """
        if n_halos is None:
            n_halos = self.n_halos
        Rs_angle = []
        alpha_Rs = []
        if z is None:
            z = self.halos_redshift_list
        if mass is None:
            mass = self.mass_list
        assert len(z) == len(mass)
        if lens_cosmo is None:
            lens_cosmo = self.lens_cosmo
        if c is None:
            c = self.halos_list['c_200']
        for h in range(n_halos):
            Rs_angle_h, alpha_Rs_h = lens_cosmo[h].nfw_physical2angle(
                M=mass[h], c=c[h]
            )
            if isinstance(Rs_angle_h, Iterable):
                Rs_angle.extend(Rs_angle_h)
            else:
                Rs_angle.append(Rs_angle_h)

            if isinstance(alpha_Rs_h, Iterable):
                alpha_Rs.extend(alpha_Rs_h)
            else:
                alpha_Rs.append(alpha_Rs_h)

        Rs_angle = np.array(Rs_angle)
        Rs_angle = np.array(Rs_angle)
        return Rs_angle, alpha_Rs

    def get_halos_lens_kwargs(self):
        Rs_angle, alpha_Rs = self.get_nfw_kwargs()

        kwargs_lens = [
            {
                "Rs": Rs_angle[h],
                "alpha_Rs": alpha_Rs[h],
                "center_x": self.halos_list['px'][h],
                "center_y": self.halos_list['py'][h],
            }
            for h in range(self.n_halos)
        ]

        return kwargs_lens

    def get_convergence_shear(
            self,
            gamma12=False,
            diff=1.0,
            diff_method="square",
            kwargs=None,
            lens_model=None,
            zdzs=None,
    ):
        """Calculates and returns the convergence and shear at the origin due
        to all the Halos.

        Parameters
        ----------
        gamma12 : bool, optional
            If True, the function will return gamma1 and gamma2 instead of gamma. Default is False.
        diff : float, optional
            Differential used in the computation of the Hessian matrix. Default is 0.0001.
        diff_method : str, optional
            The method to compute differential. Default is 'square'.

        Returns
        -------
        kappa : float
            The computed convergence at the origin.
        gamma1, gamma2 : float
            The computed two components of the shear at the origin if gamma12 is True.
        gamma : float
            The computed shear at the origin if gamma12 is False.
        """
        if kwargs is None:
            kwargs = self.get_halos_lens_kwargs()
        if lens_model is None:
            lens_model = self.lens_model
        if zdzs is not None:
            f_xx, f_xy, f_yx, f_yy = lens_model.hessian_z1z2(
                z1=zdzs[0],
                z2=zdzs[1],
                theta_x=0,
                theta_y=0,
                kwargs_lens=kwargs,
                diff=1.0,
            )
        else:
            f_xx, f_xy, f_yx, f_yy = lens_model.hessian(
                x=0.0, y=0.0, kwargs=kwargs, diff=diff, diff_method=diff_method
            )
        kappa = 1 / 2.0 * (f_xx + f_yy)
        if gamma12:
            gamma1 = 1.0 / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            return kappa, gamma1, gamma2
        else:
            gamma = np.sqrt(f_xy ** 2 + 0.25 * (f_xx - f_yy) ** 2)
            return kappa, gamma

    def compute_kappa_gamma(self, i, gamma_tot, diff, diff_method):

        if gamma_tot:
            kappa, gamma = self.get_convergence_shear(
                gamma12=False, diff=diff, diff_method=diff_method
            )
            return [kappa, gamma]
        else:
            kappa, gamma1, gamma2 = self.get_convergence_shear(
                gamma12=True, diff=diff, diff_method=diff_method
            )
            return [kappa, gamma1, gamma2]

    def filter_halos_by_redshift(self, zd, zs):
        """Filters halos and mass corrections by redshift conditions and
        constructs lens data.

        Parameters
        ----------
        - zd (float): Deflector redshift.
        - zs (float): Source redshift. It should be greater than zd; otherwise, a ValueError is raised.

        Returns
        ----------
        - tuple: Contains lens data for three different conditions:
            1. Between deflector and source redshift (ds).
            2. From zero to deflector redshift (od).
            3. From zero to source redshift (os).

        Raises
        ----------
        - ValueError: If the source redshift (zs) is less than the deflector redshift (zd).

        Internal Methods
        ----------
        - Uses `_filter_halos_by_condition` to filter halos based on redshift conditions.
        - Uses `_filter_mass_correction_by_condition` to filter mass corrections based on redshift conditions.
        - Uses `_build_lens_data` to construct lens data for each condition.
        """
        halos_od, halos_ds, halos_os = self._filter_halos_by_condition(zd, zs)
        if zs < zd:
            raise ValueError(
                f"Source redshift {zs} cannot be less than deflector redshift {zd}."
            )

        return (
            self._build_lens_data(halos_ds, zd=zd, zs=zs),
            self._build_lens_data(halos_od, zd=0, zs=zd),
            self._build_lens_data(halos_os, zd=0, zs=zs),
        )

    def _filter_halos_by_condition(self, zd, zs):
        """Filters the halos based on redshift conditions relative to deflector
        and source redshifts.

        This internal method is designed to segregate halos into three categories:
        1. Between the deflector and source redshifts (ds).
        2. From zero redshift up to the deflector redshift (od).
        3. From zero redshift up to the source redshift (os).

        Parameters
        ----------
        - zd (float): Deflector redshift.
        - zs (float): Source redshift.

        Returns
        ----------
        - tuple:
            * halos_od (DataFrame): Halos with redshift less than the deflector redshift.
            * halos_ds (DataFrame): Halos with redshift greater than or equal to the deflector redshift and less than the source redshift.
            * halos_os (DataFrame): Halos with redshift less than the source redshift.

        Note
        ----------
        This method assumes `self.halos_list` is a DataFrame containing a 'z' column that represents the redshift of each halo.
        """
        halos_ds = self.halos_list[
            (self.halos_list["z"] >= zd) & (self.halos_list["z"] < zs)
            ]
        halos_od = self.halos_list[self.halos_list["z"] < zd]
        halos_os = self.halos_list[self.halos_list["z"] < zs]
        return halos_od, halos_ds, halos_os

    def _build_lens_data(self, halos, zd, zs):
        n_halos = len(halos)
        z_halo = halos["z"]
        mass_halo = halos["mass"]
        px_halo = halos["px"]
        py_halo = halos["py"]
        c_200_halos = halos["c_200"]

        if not z_halo.size:
            warnings.warn(
                f"No halos OR mass correction in the given redshift range from zd={zd} to zs={zs}."
            )
            return None, None, None
        if zs < zd:
            raise ValueError(
                f"Source redshift {zs} cannot be less than deflector redshift {zd}."
            )
        if min(z_halo) < zd:
            raise ValueError(
                f"Redshift of the farthest {min(z_halo)}"
                f" halo cannot be smaller than deflector redshift{zd}."
            )
        if max(z_halo) > zs:
            raise ValueError(
                f"Redshift of the closet halo {max(z_halo)} "
                f"cannot be larger than source redshift {zs}."
            )

        lens_cosmo_dict = self._build_lens_cosmo_dict(z_halo, zs)
        lens_model, lens_model_list = self._build_lens_model(
            z_halo, zs, n_halos
        )

        lens_cosmo_list = list(lens_cosmo_dict.values())
        kwargs_lens = self._build_kwargs_lens(
            n_halos,
            z_halo,
            mass_halo,
            px_halo,
            py_halo,
            c_200_halos,
            lens_model_list,
            lens_cosmo_list,
        )
        # Note: If MASS_MOMENT (moment),this need to be change
        return lens_model, lens_cosmo_list, kwargs_lens

    def _build_lens_cosmo_dict(self, combined_redshift_list, z_source):
        """Constructs a dictionary mapping each redshift to its corresponding LensCosmo instance.

        Parameters
        ----------
        combined_redshift_list : list or array-like
            List of redshifts representing the halos and mass corrections combined.

        z_source : float
            Source redshift.

        Returns
        -------
        lens_cosmo_dict : dict
            Dictionary mapping each redshift to its corresponding LensCosmo instance.
        """
        return {
            z: LensCosmo(z_lens=z, z_source=z_source, cosmo=self.cosmo)
            for z in combined_redshift_list
        }

    def _build_lens_model(self, combined_redshift_list, z_source, n_halos):
        """Construct a lens model based on the provided combined redshift list,
        source redshift, and number of halos.

        The method generates a lens model list using 'NFW' for halos and 'CONVERGENCE' for any additional mass
        corrections present in the combined redshift list. The method ensures that the number of redshifts in the
        combined list matches the provided number of halos, and raises an error otherwise.

        Parameters
        ----------
        combined_redshift_list : list or array-like
            List of redshifts combining both halos and any additional mass corrections.

        z_source : float
            The redshift of the source.

        n_halos : int
            The number of halos present in the combined redshift list.

        Returns
        -------
        lens_model : lenstronomy.LensModel
            The constructed lens model based on the provided parameters.

        lens_model_list : list of str
            List containing the lens model type ('NFW' or 'CONVERGENCE') for each redshift in the combined list.

        Raises
        ------
        ValueError:
            If the length of the combined redshift list does not match the specified number of halos.

        Notes
        -----
        The order of the lens model list is constructed as:
        ['NFW', 'NFW', ..., 'CONVERGENCE', 'CONVERGENCE', ...],
        where the number of 'NFW' entries matches `n_halos` and the number of 'CONVERGENCE' entries corresponds
        to any additional redshifts present in `halos_redshift_list`.
        """

        lens_model_list = ["NFW"] * n_halos

        lens_model = LensModel(
            lens_model_list=lens_model_list,
            lens_redshift_list=combined_redshift_list,
            cosmo=self.cosmo,
            multi_plane=True,
            z_source=z_source,
            z_source_convention=self._z_source_convention,
        )

        return lens_model, lens_model_list

    def _build_kwargs_lens(
            self,
            n_halos,
            z_halo,
            mass_halo,
            px_halo,
            py_halo,
            c_200_halos,
            lens_model_list,
            lens_cosmo_list=None,
    ):
        if n_halos == 0:
            return None

        Rs_angle, alpha_Rs = self.get_nfw_kwargs(
            z=z_halo,
            mass=mass_halo,
            n_halos=n_halos,
            lens_cosmo=lens_cosmo_list[:n_halos],
            c=c_200_halos
        )

        return [
            {
                "Rs": Rs_angle[i],
                "alpha_Rs": alpha_Rs[i],
                "center_x": px_halo[i],
                "center_y": py_halo[i],
            }
            for i in range(n_halos)
        ]

    def get_lens_data_by_redshift(self, zd, zs):
        """Retrieves lens data filtered by the specified redshift range.

        Given a range of redshifts defined by zd and zs, this function filters halos
        and returns the corresponding lens models, lens cosmologies, and lens keyword arguments
        for three categories: 'ds', 'od', and 'os'. ('ds' stands for deflector-source, 'od' stands for
        observer-deflector, and 'os' stands for observer-source.)

        Parameters
        ----------
        zd : float
            The deflector redshift. It defines the lower bound of the redshift range.

        zs : float
            The source redshift. It defines the upper bound of the redshift range.

        Returns
        -------
        dict
            A dictionary with three keys: 'ds', 'od', and 'os'. Each key maps to a sub-dictionary containing:
            - 'lens_model': The lens model for the corresponding category.
            - 'lens_cosmo': The lens cosmology for the corresponding category.
            - 'kwargs_lens': The lens keyword arguments for the corresponding category.

        Note
        ----
            lens_model_ds = lens_data['ds']['lens_model']
            lens_cosmo_ds = lens_data['ds']['lens_cosmo']
            kwargs_lens_ds = lens_data['ds']['kwargs_lens']
             ... and similarly for 'od' and 'os' data
        """
        ds_data, od_data, os_data = self.filter_halos_by_redshift(zd, zs)

        lens_model_ds, lens_cosmo_ds, kwargs_lens_ds = ds_data
        lens_model_od, lens_cosmo_od, kwargs_lens_od = od_data
        lens_model_os, lens_cosmo_os, kwargs_lens_os = os_data

        return {
            "ds": {
                "lens_model": lens_model_ds,
                "lens_cosmo": lens_cosmo_ds,
                "kwargs_lens": kwargs_lens_ds,
            },
            "od": {
                "lens_model": lens_model_od,
                "lens_cosmo": lens_cosmo_od,
                "kwargs_lens": kwargs_lens_od,
            },
            "os": {
                "lens_model": lens_model_os,
                "lens_cosmo": lens_cosmo_os,
                "kwargs_lens": kwargs_lens_os,
            },
        }

    def compute_various_kappa_gamma_values(self, zd, zs):
        """Computes various kappa (convergence) and gamma (shear) values for
        given deflector and source redshifts.

        This function retrieves the lens data based on the input redshifts and computes the convergence
        and shear for three categories: 'od', 'os', and 'ds'. The gamma values are computed for
        both components, gamma1 and gamma2.

        Parameters
        ----------
        zd : float
            The deflector redshift.

        zs : float
            The source redshift.

        Returns
        -------
        kappa_od : float
            The convergence for the 'od' category.

        kappa_os : float
            The convergence for the 'os' category.

        gamma_od1 : float
            The gamma1 shear component for the 'od' category.

        gamma_od2 : float
            The gamma2 shear component for the 'od' category.

        gamma_os1 : float
            The gamma1 shear component for the 'os' category.

        gamma_os2 : float
            The gamma2 shear component for the 'os' category.

        kappa_ds : float
            The convergence for the 'ds' category.

        gamma_ds1 : float
            The gamma1 shear component for the 'ds' category.

        gamma_ds2 : float
            The gamma2 shear component for the 'ds' category.
        """
        # Obtain the lens data for each redshift using the get_lens_data_by_redshift function
        lens_data = self.get_lens_data_by_redshift(zd, zs)

        # Extracting lens model and kwargs for 'od' and 'os'
        lens_model_od = lens_data["od"]["lens_model"]
        kwargs_lens_od = lens_data["od"]["kwargs_lens"]

        lens_model_os = lens_data["os"]["lens_model"]
        kwargs_lens_os = lens_data["os"]["kwargs_lens"]

        lens_model_ds = lens_data["ds"]["lens_model"]
        kwargs_lens_ds = lens_data["ds"]["kwargs_lens"]
        kappa_od, gamma_od1, gamma_od2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_od, lens_model=lens_model_od
        )

        kappa_os, gamma_os1, gamma_os2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_os, lens_model=lens_model_os
        )
        kappa_ds, gamma_ds1, gamma_ds2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_ds, lens_model=lens_model_ds, zdzs=(zd, zs)
        )

        return (
            kappa_od,
            kappa_os,
            gamma_od1,
            gamma_od2,
            gamma_os1,
            gamma_os2,
            kappa_ds,
            gamma_ds1,
            gamma_ds2,
        )

    def get_kext_gext_values(self, zd, zs):
        r"""Computes the external convergence (kappa_ext) and external shear
        (gamma_ext) for given deflector and source redshifts.

        Parameters
        ----------
        zd : float
            The deflector redshift.
        zs : float
            The source redshift.

        Returns
        -------
        kext : float
            The computed external convergence value.
        gext : float
            The computed external shear magnitude.

        Notes
        -----
        The function implements the following formula:

        .. math::
            1 - \kappa_{\text{ext}} = \frac{(1-\kappa_{\text{od}})(1-\kappa_{\text{os}})}{1-\kappa_{\text{ds}}}

        and

        .. math::
            \gamma_{\text{ext}} = \sqrt{(\gamma_{\text{od}1}+\gamma_{\text{os}1}-\gamma_{\text{ds}1})^2+(\gamma_{\text{od}2}+\gamma_{\text{os}2}-\gamma_{\text{ds}2})^2}
        """
        (
            kappa_od,
            kappa_os,
            gamma_od1,
            gamma_od2,
            gamma_os1,
            gamma_os2,
            kappa_ds,
            gamma_ds1,
            gamma_ds2,
        ) = self.compute_various_kappa_gamma_values(zd, zs)

        kext = 1 - (1 - kappa_od) * (1 - kappa_os) / (1 - kappa_ds)
        gext = math.sqrt(
            (gamma_od1 + gamma_os1 - gamma_ds1) ** 2
            + (gamma_od2 + gamma_os2 - gamma_ds2) ** 2
        )

        return kext, gext

    def get_kappaext_gammaext_distib_zdzs(self, zd, zs, listmean=False):
        """Computes the distribution of external convergence (kappa_ext) and
        external shear (gamma_ext) for given deflector and source redshifts.

        Parameters
        ----------
        zd : float
            The deflector redshift.
        zs : float
            The source redshift.

        Returns
        -------
        kappa_gamma_distribution : numpy.ndarray
            An array of shape (samples_number, 2) containing the computed kappa_ext and gamma_ext values
            for the given deflector and source redshifts. Each row corresponds to a sample,
            with the first column being kappa_ext and the second column being gamma_ext.

        Notes
        -----
        The progress is shown with a tqdm progress bar if the number of samples exceeds 999.
        The total elapsed time for computing weak-lensing maps is printed at the end.
        """

        kappa_gamma_distribution = np.empty((self.samples_number, 2))

        loop = range(self.samples_number)
        if self.samples_number > 999:
            loop = tqdm(loop)

        start_time = time.time()

        for i in loop:
            kappa, gamma = self.get_kext_gext_values(zd=zd, zs=zs)
            kappa_gamma_distribution[i] = [kappa, gamma]

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this halos lists, zd,zs, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )
        if listmean:
            kappa_mean = np.mean(kappa_gamma_distribution[:, 0])
            kappa_gamma_distribution[:, 0] -= kappa_mean

        return kappa_gamma_distribution

    def generate_distributions_0to5(self, output_format="dict", listmean=False):
        """Generates distributions of external convergence (kappa_ext) and
        external shear (gamma_ext) for a range of deflector and source
        redshifts from 0 to 5.

        Parameters
        ----------
        listmean
        output_format : str, optional
            The format of the output data. Options are:
            'dict' - A list of dictionaries.
            'vector' - A list of vectors [zd, zs, kappa, gamma].
            Default is 'dict'.

        Returns
        -------
        distributions : list
            If output_format='dict', a list of dictionaries. Each dictionary contains:
            - zd (float) : The deflector redshift.
            - zs (float) : The source redshift.
            - kappa (float) : The computed external convergence value for the given zd and zs.
            - gamma (float) : The computed external shear magnitude for the given zd and zs.
            If output_format='vector', a list of vectors with elements [zd, zs, kappa, gamma].

        Notes
        -----
        The function iterates through possible values of zs from 0 to 5 in steps of 0.1.
        For each zs, it considers zd values from 0 to zs-0.1 in steps of 0.1. For each zd, zs pair,
        it computes the kappa_ext and gamma_ext distributions and appends them to the result list.
        """
        distributions = []
        zs_values = np.linspace(0, 5, int(5 / 0.1 + 1))
        for zs in zs_values:
            zd_values = np.linspace(0, zs - 0.1, int(zs / 0.1))
            for zd in zd_values:
                kappa_gamma_dist = self.get_kappaext_gammaext_distib_zdzs(zd=zd, zs=zs, listmean=listmean)
                for kappa_gamma in kappa_gamma_dist:
                    kappa, gamma = kappa_gamma

                    if output_format == "dict":
                        distributions.append(
                            {
                                "zd": round(zd, 3),
                                "zs": round(zs, 3),
                                "kappa": kappa,
                                "gamma": gamma,
                            }
                        )
                    elif output_format == "vector":
                        distributions.append([round(zd, 3), round(zs, 3), kappa, gamma])
                    else:
                        raise ValueError(
                            "Invalid output_format. Choose either 'dict' or 'vector'."
                        )

        return distributions

    def compute_various_kappa_gamma_values_new(self, zd, zs):
        r"""
        Computes various convergence (kappa) and shear (gamma) values for given deflector and source redshifts.

        This function extracts the lens model and its keyword arguments for different redshift combinations
        ('od', 'os', and 'ds'). It then computes the convergence and shear values for each of these combinations.

        Parameters
        ----------
        zd : float
            The deflector redshift.
        zs : float
            The source redshift.

        Returns
        -------
        tuple
            A tuple containing:
                - A tuple of computed values for kappa and gamma for the different redshift combinations and the
                  external convergence and shear.
                - A tuple containing the lens model and its keyword arguments for the 'os' redshift combination.

        Notes
        -----
        This function is utilized by the `self.get_alot_distib_()` method. The mathematical formulations behind
        the calculations, especially for `kext` and `gext`, can be referenced from the documentation of
        `get_kext_gext_values`, is applied with the line of sight non-linear correction.

        The function implements the following formulae for the external convergence and shear with LOS correction:

        .. math::
            1 - \kappa_{\text{ext}} = \frac{(1-\kappa_{\text{od}})(1-\kappa_{\text{os}})}{1-\kappa_{\text{ds}}}

        and

        .. math::
            \gamma_{\text{ext}} = \sqrt{(\gamma_{\text{od}1}+\gamma_{\text{os}1}-\gamma_{\text{ds}1})^2+(\gamma_{\text{od}2}+\gamma_{\text{os}2}-\gamma_{\text{ds}2})^2}

        """
        # Obtain the lens data for each redshift using the get_lens_data_by_redshift function
        lens_data = self.get_lens_data_by_redshift(zd, zs)

        # Extracting lens model and kwargs for 'od' and 'os'
        lens_model_od = lens_data["od"]["lens_model"]
        kwargs_lens_od = lens_data["od"]["kwargs_lens"]

        lens_model_os = lens_data["os"]["lens_model"]
        kwargs_lens_os = lens_data["os"]["kwargs_lens"]

        lens_model_ds = lens_data["ds"]["lens_model"]
        kwargs_lens_ds = lens_data["ds"]["kwargs_lens"]

        kappa_od, gamma_od1, gamma_od2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_od, lens_model=lens_model_od
        )

        kappa_os, gamma_os1, gamma_os2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_os, lens_model=lens_model_os
        )

        kappa_os2, gamma_os12, gamma_os22 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_os, lens_model=lens_model_os, zdzs=(0, zs)
        )

        kappa_ds, gamma_ds1, gamma_ds2 = self.get_convergence_shear(
            gamma12=True, kwargs=kwargs_lens_ds, lens_model=lens_model_ds, zdzs=(zd, zs)
        )

        kext = 1 - (1 - kappa_od) * (1 - kappa_os) / (1 - kappa_ds)
        gext = math.sqrt(
            (gamma_od1 + gamma_os1 - gamma_ds1) ** 2
            + (gamma_od2 + gamma_os2 - gamma_ds2) ** 2
        )

        return (
            kappa_od,
            kappa_os,
            gamma_od1,
            gamma_od2,
            gamma_os1,
            gamma_os2,
            kappa_ds,
            gamma_ds1,
            gamma_ds2,
            kappa_os2,
            gamma_os12,
            gamma_os22,
            kext,
            gext
        ), (kwargs_lens_os, lens_model_os)

    def get_alot_distib_(self, zd, zs):
        """Computes the distribution of external convergence (kappa_ext) and
        external shear (gamma_ext) for given deflector and source redshifts.

        Parameters
        ----------
        zd : float
            The deflector redshift.
        zs : float
            The source redshift.

        Returns
        -------
        kappa_gamma_distribution : numpy.ndarray
            An array of shape (samples_number, 2) containing the computed kappa_ext and gamma_ext values
            for the given deflector and source redshifts. Each row corresponds to a sample,
            with the first column being kappa_ext and the second column being gamma_ext.

        Notes
        -----
        The progress is shown with a tqdm progress bar if the number of samples exceeds 999.
        The total elapsed time for computing weak-lensing maps is printed at the end.
        """

        kappa_gamma_distribution = np.empty((self.samples_number, 14))
        lens_instance = np.empty((self.samples_number, 2), dtype=object)

        loop = range(self.samples_number)
        if self.samples_number > 999:
            loop = tqdm(loop)

        start_time = time.time()

        for i in loop:
            (kappa_od, kappa_os, gamma_od1, gamma_od2, gamma_os1, gamma_os2, kappa_ds, gamma_ds1, gamma_ds2, kappa_os2,
             gamma_os12, gamma_os22, kext, gext), (
                kwargs_lens_os, lens_model_os) = self.compute_various_kappa_gamma_values_new(
                zd=zd, zs=zs)
            kappa_gamma_distribution[i] = [kappa_od, kappa_os, gamma_od1, gamma_od2, gamma_os1, gamma_os2, kappa_ds,
                                           gamma_ds1, gamma_ds2, kappa_os2, gamma_os12, gamma_os22, kext, gext]
            lens_instance[i] = [kwargs_lens_os, lens_model_os]
        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this halos lists, zd,zs, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution, lens_instance

    def compute_kappa_in_bins(self):
        """Computes the kappa values for each redshift bin."""
        # todo: different zd,zs
        bins = np.arange(0, 5.025, 0.05)
        bin_centers = [round((z1 + z2) / 2, 3) for z1, z2 in zip(bins[:-1], bins[1:])]
        all_kappa_dicts = []
        for _ in range(self.samples_number):
            # Iterate over the bins
            kappa_dict = {}
            for i in range(len(bins) - 1):
                # Filter halos in the current redshift bin
                _, halos_ds, _ = self._filter_halos_by_condition(bins[i], bins[i + 1])

                # Since we want halos between zd and zs (in this case, the current bin upper limit)
                # we will consider halos from halos_ds as those are the ones between zd and zs
                if len(halos_ds) > 0:
                    lens_model, lens_cosmo_list, kwargs_lens = self._build_lens_data(
                        halos_ds, None, zd=0, zs=5)
                    kappa, _ = self.get_convergence_shear(lens_model=lens_model, kwargs=kwargs_lens, gamma12=False,
                                                          zdzs=(0, 5))
                    kappa_dict[bin_centers[i]] = kappa
                else:
                    kappa_dict[bin_centers[i]] = 0
            all_kappa_dicts.append(kappa_dict)
        return all_kappa_dicts

    def xy_convergence(
            self,
            x,
            y,
            diff=0.0000001,
            diff_method="square",
            kwargs=None,
            lens_model=None,
            zdzs=None
    ):
        r"""
        Computes the convergence (kappa) at given (x, y) coordinates using either the hessian of the lens model or the
        hessian based on redshifts z1 and z2 (if provided).

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        diff : float, optional
            The differentiation value used for computing the hessian. Default is 1e-7.
        diff_method : str, optional
            The method to use for differentiation when computing the hessian. Default is "square".
        kwargs : dict, optional
            Keyword arguments for the lens model.
        lens_model : lenstronomy.LensModel instance, optional
            The lens model to use. If not provided, the function will utilize the lens model from the class instance.
        zdzs : tuple of float, optional
            A tuple containing two redshift values (z1, z2). If provided, the hessian will be computed based on these
            redshifts.

        Returns
        -------
        kappa : float
            The computed convergence value at the given (x, y) coordinates.

        Notes
        -----
        The convergence, :math:`\kappa`, is computed using the hessian of the lens model as:

        .. math::
            \kappa = \frac{1}{2} (f_{xx} + f_{yy})
        """
        if zdzs is not None:
            f_xx, _, _, f_yy = lens_model.hessian_z1z2(
                z1=zdzs[0],
                z2=zdzs[1],
                theta_x=x,
                theta_y=y,
                kwargs_lens=kwargs,
                diff=0.0000001,
            )
        else:
            f_xx, _, _, f_yy = lens_model.hessian(
                x=x, y=y, kwargs=kwargs, diff=diff, diff_method=diff_method
            )
        kappa = 1 / 2.0 * (f_xx + f_yy)

        return kappa

    def compute_kappa(self,
                      diff=0.0000001,
                      num_points=500,
                      diff_method="square",
                      kwargs=None,
                      lens_model=None,
                      mass_sheet=None,
                      radial_interpolate=None,
                      enhance_pos=False,
                      ):
        if mass_sheet is not None:
            self.mass_sheet = mass_sheet
            if mass_sheet is True:
                if radial_interpolate is not None:
                    self.radial_interpolate = radial_interpolate
                    radial = True

        if kwargs is None:
            kwargs = self.get_halos_lens_kwargs()
        if lens_model is None:
            lens_model = self.lens_model

        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806
        x = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        y = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        X, Y = np.meshgrid(x, y)
        mask_2D = X ** 2 + Y ** 2 <= radius_arcsec ** 2
        mask_1D = mask_2D.ravel()

        # Use lenstronomy utility to make grid
        x_grid, y_grid = make_grid(numPix=num_points, deltapix=2 * radius_arcsec / num_points)
        x_grid, y_grid = x_grid[mask_1D], y_grid[mask_1D]

        # Calculate the kappa values
        kappa_values = lens_model.kappa(x_grid, y_grid, kwargs, diff=diff, diff_method=diff_method)
        kappa_image = np.ones((num_points, num_points)) * np.nan
        kappa_image[mask_2D] = kappa_values
        if enhance_pos:
            self.enhance_halos_table_random_pos()
        return kappa_image, kappa_values

    def plot_convergence(self,
                         diff=0.0000001,
                         num_points=500,
                         diff_method="square",
                         kwargs=None,
                         lens_model=None,
                         mass_sheet=None,
                         radial_interpolate=None,
                         enhance_pos=True,
                         ):
        r"""
        Plots the convergence (:math:`\kappa`) across the lensed sky area.

        Parameters
        ----------
        diff : float, optional
            The differentiation value used for computing the hessian. Default is 1e-7.
        num_points : int, optional
            Number of points along each axis for which convergence is computed. Default is 500.
        diff_method : str, optional
            The method to use for differentiation when computing the hessian. Default is "square".
        kwargs : dict, optional
            Keyword arguments for the lens model. If not provided, the halos lens kwargs of the instance are used.
        lens_model : LensModel instance, optional
            The lens model to use. If not provided, the lens model from the class instance is utilized.
        zdzs : tuple of float, optional
            A tuple containing two redshift values (z1, z2). If provided, the hessian will be computed based on these redshifts.
        mass_sheet : bool, optional
            Whether to utilize a mass sheet for the plot. If set, it will temporarily overwrite the instance's mass sheet.
        radial_interpolate : bool, optional
            If set along with `mass_sheet=True`, this will temporarily overwrite the instance's radial interpolate setting.
        enhance_pos : bool, optional
            Enhances halo positions randomly after plotting. Default is True.

        Returns
        -------
        None
            The function will display a plot of the computed convergence with plot.

        Notes
        -----
        The function computes the convergence for a grid defined by `num_points` and plots the result using matplotlib.
        The computed sky area is determined by the instance's sky area, converted from square degrees to arcseconds.
        Overlaying on the convergence plot are positions of halos represented by yellow 'x' markers.

        The computation is parallelized for efficiency, using the number of available CPU cores.
        Temporary changes made to the instance (like `mass_sheet` and `radial_interpolate`) are reverted at the end of the function.
        """
        import matplotlib.pyplot as plt

        original_mass_sheet = self.mass_sheet
        original_radial_interpolate = self.radial_interpolate
        radial = False
        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806

        try:
            kappa_image, _ = self.compute_kappa(diff=diff,
                                                num_points=num_points,
                                                diff_method=diff_method,
                                                kwargs=kwargs,
                                                lens_model=lens_model,
                                                mass_sheet=mass_sheet,
                                                radial_interpolate=radial_interpolate,
                                                enhance_pos=False, )

            plt.imshow(kappa_image, extent=[-radius_arcsec, radius_arcsec, -radius_arcsec, radius_arcsec])
            plt.colorbar(label=r'$\kappa$')

            halos_x = [k.get('center_x', None) for k in kwargs]
            halos_y = [-k.get('center_y') if k.get('center_y') is not None else None for k in kwargs]
            plt.scatter(halos_x, halos_y, color='yellow', marker='x', label='Halos')
            plt.title(f'Convergence Plot, radius is {radius_arcsec} arcsec')
            plt.xlabel('x-coordinate (arcsec)')
            plt.ylabel('y-coordinate (arcsec)')
            plt.legend()
            plt.show()

        finally:
            self.mass_sheet = original_mass_sheet
            if radial is True:
                self.radial_interpolate = original_radial_interpolate
            if enhance_pos:
                self.enhance_halos_table_random_pos()

    def azimuthal_average_kappa_dict(self,
                                     diff=0.0000001,
                                     diff_method="square",
                                     kwargs=None,
                                     lens_model=None,
                                     zdzs=None):
        r"""
        Computes the azimuthal average of convergence (:math:`\kappa`) values over a set of radii.

        Parameters
        ----------
        diff : float, optional
            The differentiation value used in computing convergence. Default is 1e-7.
        diff_method : str, optional
            The method used for differentiation in convergence computation. Default is "square".
        kwargs : dict, optional
            Keyword arguments for the lens model. If not provided, the halos lens kwargs of the instance are used.
        lens_model : LensModel instance, optional
            The lens model to use. If not provided, the lens model from the class instance is utilized.
        zdzs : tuple of float, optional
            A tuple containing two redshift values (z1, z2). If provided, convergence will be computed based on these redshifts.

        Returns
        -------
        all_kappa_dicts : list of dict
            A list of dictionaries. Each dictionary maps a radius (rounded to 4 decimal places) to its azimuthally averaged convergence value.

        Notes
        -----
        The function calculates convergence values for a series of radii, taken as a linear space over the entire sky area (converted to arcseconds). For each radius, a series of x-values are considered and the convergence is computed for two y-values (using a circle's equation).

        After computation, the results are averaged and stored in a dictionary, which is then added to a list that encompasses all sample runs.
        """
        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806
        radii = np.linspace(0, radius_arcsec, 25)

        all_kappa_dicts = []
        if lens_model is None:
            lens_model = self.lens_model

        for _ in range(self.samples_number):
            self.enhance_halos_table_random_pos()
            if kwargs is None:
                kwargs = self.get_halos_lens_kwargs()

            kappa_dict = {}

            for r in radii:
                x_values = np.linspace(-r, r, 25)
                kappas = []

                for x in x_values:
                    y1 = np.sqrt(r ** 2 - x ** 2)
                    y2 = -y1

                    kappas.append(self.xy_convergence(x, y1, diff, diff_method, kwargs, lens_model, zdzs))
                    kappas.append(self.xy_convergence(x, y2, diff, diff_method, kwargs, lens_model, zdzs))

                kappa_avg = np.mean(kappas)
                kappa_dict[round(r, 4)] = kappa_avg

            all_kappa_dicts.append(kappa_dict)

        return all_kappa_dicts

    def compute_azimuthal_kappa_in_bins(self):
        r"""
        Computes the azimuthal average of convergence values binned by redshift.

        Returns
        -------
        all_kappa_dicts : list of dict
            A list of dictionaries. Each dictionary maps a bin center redshift to its
            corresponding azimuthally averaged convergence value or 0 if no halos are present in the bin.

        Notes
        -----
        The function is designed for calculating the azimuthal mass sheet value.
        """

        bins = np.arange(0, 5.025, 0.05)  # todo: different zd,zs
        bin_centers = [round((z1 + z2) / 2, 3) for z1, z2 in zip(bins[:-1], bins[1:])]
        all_kappa_dicts = []
        for _ in range(self.samples_number):
            self.enhance_halos_table_random_pos()
            # Iterate over the bins
            kappa_dict = {}
            for i in range(len(bins) - 1):
                # Filter halos in the current redshift bin
                _, halos_ds, _ = self._filter_halos_by_condition(bins[i], bins[i + 1])

                # Since we want halos between zd and zs (in this case, the current bin upper limit)
                # we will consider halos from halos_ds as those are the ones between zd and zs
                if len(halos_ds) > 0:
                    lens_model, lens_cosmo_list, kwargs_lens = self._build_lens_data(
                        halos_ds, None, zd=0, zs=5)
                    kappa = self.azimuthal_average_kappa_dict(diff=0.0000001,
                                                              diff_method="square",
                                                              kwargs=kwargs_lens,
                                                              lens_model=lens_model,
                                                              zdzs=None)
                    kappa_dict[bin_centers[i]] = kappa
                else:
                    kappa_dict[bin_centers[i]] = 0
            all_kappa_dicts.append(kappa_dict)
        return all_kappa_dicts

    def compare_plot_convergence(self,
                                 diff=0.0000001,
                                 diff_method="square",
                                 kwargs=None,
                                 lens_model=None,
                                 ):
        r"""
        Compares and plots the convergence for different configurations of the mass sheet and radial
        interpolation parameters.

        This function invokes the `plot_convergence` method three times, with different
        configurations for the `mass_sheet` and `radial_interpolate` arguments,
        allowing users to visually compare the effects of these configurations on the convergence plot.

        Parameters
        ----------
        diff : float, optional
            The differentiation value used in computing convergence. Default is 1e-7.
        diff_method : str, optional
            The method used for differentiation in convergence computation. Default is "square".
        kwargs : dict, optional
            Keyword arguments for the lens model. If not provided,
            the lens model parameters from the class instance are used.
        lens_model : LensModel instance, optional
            The lens model to use. If not provided, the lens model from the class instance is utilized.
        zdzs : tuple of float, optional
            A tuple containing two redshift values (z1, z2). If provided,
            convergence will be computed based on these redshifts.

        Notes
        -----
        The configurations for the convergence plots are as follows:

        1. `mass_sheet` set to `False`.
        2. `mass_sheet` set to `True` and `radial_interpolate` set to `True`.
        3. `mass_sheet` set to `True` and `radial_interpolate` set to `False`.

        In all cases, the `enhance_pos` parameter for `plot_convergence` is set to `False`.

        This function is currently under development!
        """

        # TODO：debug, this is currently not working as expected
        print('mass_sheet=False')
        # mass_sheet=False
        self.plot_convergence(diff=diff,
                              diff_method=diff_method,
                              kwargs=kwargs,
                              lens_model=lens_model,
                              mass_sheet=False,
                              radial_interpolate=False,
                              enhance_pos=False
                              )

        # mass_sheet=True, radial_interpolate=True
        print('mass_sheet=True, radial_interpolate=True')
        self.plot_convergence(diff=diff,
                              diff_method=diff_method,
                              kwargs=kwargs,
                              lens_model=lens_model,
                              mass_sheet=True,
                              radial_interpolate=True,
                              enhance_pos=False)

        # mass_sheet=True, radial_interpolate=False
        print('mass_sheet=True, radial_interpolate=False')
        self.plot_convergence(diff=diff,
                              diff_method=diff_method,
                              kwargs=kwargs,
                              lens_model=lens_model,
                              mass_sheet=True,
                              radial_interpolate=False,
                              enhance_pos=False
                              )

    def total_halo_mass(self):
        """
        Calculates the total mass of all halos.

        This function sums up all the masses from the mass list associated with the class instance to give the total mass of all halos.

        Returns
        -------
        float
            Total mass of all halos.

        Notes
        -----
        The total mass is computed by summing up all the entries in the `mass_list` attribute of the class instance.
        """

        mass_list = self.mass_list
        return np.sum(mass_list)

    def total_critical_mass(self, method='differential_comoving_volume'):
        """
        Computes the total critical mass within a given sky area up to a redshift of 5
        using either the total comoving volume or differential comoving volume method.

        The function computes the critical mass using either the 'comoving_volume'
        method or the 'differential_comoving_volume' method.

        Parameters
        ----------
        method : str, optional
            The method to use for computing the critical mass. Options are:
            - 'comoving_volume': Computes the total critical mass using the comoving volume method.
            - 'differential_comoving_volume': Computes the total critical mass using the differential comoving volume method.
            Default is 'differential_comoving_volume'.

        Returns
        -------
        float
            The computed total critical mass up to redshift 5.

        """
        v_ratio = self.sky_area / 41252.96

        z = np.linspace(0, 5, 2000)
        total_mass = 0.0
        if method == 'comoving_volume':
            for i in range(1, len(z)):
                critical_density_z = self.cosmo.critical_density(z[i]).to('Msun/Mpc^3').value
                # Compute differential comoving volume for this redshift slice
                dVc = ((self.cosmo.comoving_volume(z[i]) - self.cosmo.comoving_volume(z[i - 1])) * v_ratio).value
                total_mass += dVc * critical_density_z
        if method == 'differential_comoving_volume':
            dV_dz = (self.cosmo.differential_comoving_volume(z) * (self.sky_area * (u.deg ** 2))).to_value("Mpc3")
            dV = dV_dz * ((5 - 0) / len(z))
            total_mass = np.sum(dV * self.cosmo.critical_density(z).to_value("Msun/Mpc3"))
        return total_mass  # In Msun

    def get_kappa_mean_range(
            self, diff=1.0, diff_method="square"
    ):
        kwargs = self.get_halos_lens_kwargs()
        lens_model = self.lens_model
        num_points = 50
        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806
        x = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        y = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        X, Y = np.meshgrid(x, y)
        mask_2D = X ** 2 + Y ** 2 <= radius_arcsec ** 2
        mask_1D = mask_2D.ravel()

        # Use lenstronomy utility to make grid
        x_grid, y_grid = make_grid(numPix=num_points, deltapix=2 * radius_arcsec / num_points)
        x_grid, y_grid = x_grid[mask_1D], y_grid[mask_1D]

        # Calculate the kappa values
        kappa_values = lens_model.kappa(x_grid, y_grid, kwargs, diff=diff, diff_method=diff_method)

        kappa_mean = np.mean(kappa_values)
        kappa_std = np.std(kappa_values)
        kappa_2sigma = 2 * kappa_std
        return kappa_mean, kappa_2sigma