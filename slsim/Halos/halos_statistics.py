from slsim.Halos.halos_lens_base import HalosLensBase
from lenstronomy.Util.util import make_grid
import numpy as np
import astropy.units as u
import math
import time
import multiprocessing
from slsim.Util.astro_util import cone_radius_angle_to_physical_area
from slsim.Util.param_util import deg2_to_cone_angle
from slsim.Halos.halos_util import convergence_mean_0


class HalosStatistics(HalosLensBase):
    """A class for computing statistical distributions of lensing properties
    such as external convergence (kappa_ext) and external shear (gamma_ext)
    across multiple samples of halo configurations.

    This class extends `HalosLensBase` to include statistical methods that allow for the computation of lensing effects over a range of redshifts and configurations, providing tools to analyze the statistical properties of lensing in cosmological simulations.

    Attributes:
        samples_number (int): Number of samples for statistical calculations.
        halos_list (astropy.Table): Table containing details of halos, including their redshifts and masses.
        mass_correction_list (astropy.Table, optional): Table for mass correction, containing details like redshifts and external convergences.
        cosmo (astropy.Cosmology): Cosmology used for lensing computations.
        sky_area (float): Total sky area (in steradians) over which Halos are distributed. Defaults to full sky (4pi steradians). Optional.
        mass_sheet (bool): Flag to decide whether to use the mass_sheet correction.

    Methods:
        get_kappaext_gammaext_distib_zdzs: Computes the distribution of external convergence and shear for given deflector and source redshifts.
        generate_distributions_0to5: Generates distributions of external convergence and shear for a range of deflector and source redshifts from 0 to 5.
        compute_various_k_g_lens_values: Computes various convergence and shear values for given deflector and source redshifts.
        get_all_pars_distib: Computes the distribution of various lensing parameters for given deflector and source redshifts.
        compute_kappa_in_bins: Computes the kappa values for each redshift bin for mass sheet correction.
        total_halo_mass: Calculates the total mass of all halos.
        total_critical_mass: Computes the total critical mass within a given sky area up to a redshift of 5.
        mass_divide_kcrit: Computes the external convergence by dividing the mass of each halo by the critical surface density.
        kappa_divergence: Calculates the divergence of kappa values across a specified sky area.
        compute_kappa_gamma: Compute the convergence and shear values for a given index, designed for use with multiprocessing.
        get_kappa_gamma_distib: Computes and returns the distribution of convergence and shear values using multiprocessing.
        get_kappa_gamma_distib_without_multiprocessing: Computes and returns the distribution of convergence and shear values without using multiprocessing.
    """

    def __init__(
        self,
        halos_list,
        mass_correction_list=None,
        cosmo=None,
        sky_area=0.004 * np.pi,
        samples_number=1000,
        mass_sheet=True,
        z_source=5,
    ):
        super().__init__(
            halos_list,
            mass_correction_list,
            cosmo,
            sky_area,
            mass_sheet,
            z_source,
        )
        self.samples_number = samples_number

    def get_kappaext_gammaext_distib_zdzs(self, zd, zs, listmean=False):
        """Computes the distribution of external convergence (kappa_ext) and
        external shear (gamma_ext) for given deflector and source redshifts.

        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :param listmean: the boolean if average convergence (kappa) to 0
        :type listmean: bool
        :return: An array of shape (samples_number, 2) containing the computed kappa_ext and gamma_ext values for the given deflector and source redshifts. Each row corresponds to a sample, with the first column being kappa_ext and the second column being gamma_ext.
        :rtype: numpy.ndarray

        .. note::
            The total elapsed time for computing weak-lensing maps is printed at the end.
        """

        kappa_gamma_distribution = np.empty((self.samples_number, 2))

        loop = range(self.samples_number)

        start_time = time.time()

        for i in loop:
            self.enhance_halos_table_random_pos()
            kappa, gamma = self.halos_get_kext_gext_values(zd=zd, zs=zs)
            kappa_gamma_distribution[i] = [kappa, gamma]

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this halos lists, zd,zs, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )
        if listmean:
            kappa_gamma_distribution[:, 0] = convergence_mean_0(
                kappa_gamma_distribution[:, 0]
            )

        return kappa_gamma_distribution

    def generate_distributions_0to5(self, output_format="dict", listmean=False):
        """Generates distributions of external convergence (kappa_ext) and
        external shear (gamma_ext) for a range of deflector and source
        redshifts from 0 to 5.

        :param listmean: the boolean if average convergence (kappa) to 0
        :type listmean: bool
        :param output_format: The format of the output data. Options are `dict` - A list of dictionaries, `vector` - A list of vectors [zd, zs, kappa, gamma]. Default is `dict`.
        :type output_format: str, optional
        :return: If output_format='dict`, a list of dictionaries. Each dictionary contains: zd (float) - The deflector redshift, zs (float) - The source redshift, kappa (float) - The computed external convergence value for the given zd and zs, gamma (float) - The computed external shear magnitude for the given zd and zs. If output_format='vector`, a list of vectors with elements [zd, zs, kappa, gamma].
        :rtype: list

        .. note::
            The function iterates through possible values of zs from 0 to 5 in steps of 0.1. For each zs, it considers zd values from 0 to zs-0.1 in steps of 0.1. For each zd, zs pair, it computes the kappa_ext and gamma_ext distributions and appends them to the result list.
        """

        distributions = []
        zs_values = np.linspace(0, 5, int(5 / 0.1 + 1))
        for zs in zs_values:
            zd_values = np.linspace(0, zs - 0.1, int(zs / 0.1))
            for zd in zd_values:
                kappa_gamma_dist = self.get_kappaext_gammaext_distib_zdzs(
                    zd=zd, zs=zs, listmean=listmean
                )
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

    def compute_various_k_g_lens_values(self, zd, zs):
        r"""Computes various convergence (kappa) and shear (gamma) values for
        given deflector and source redshifts and the lens kwargs and lens
        model.

        This function extracts the lens model and its keyword arguments for different redshift combinations
        ('od`, `os`, and `ds`). It then computes the convergence and shear values for each of these combinations.

        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :returns:
            A tuple containing:
                - A tuple of computed values for kappa and gamma for the different redshift combinations and the external convergence and shear.
                - A tuple containing the lens model and its keyword arguments for the `os` redshift combination.
        :rtype: tuple

        .. note::
            This function is utilized by the `self.get_all_pars_distib()` method. The mathematical formulations behind
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

        # Extracting lens model and lens_kwargs for 'od' and 'os'
        lens_model_od = lens_data["od"]["param_lens_model"]
        kwargs_lens_od = lens_data["od"]["kwargs_lens"]

        lens_model_os = lens_data["os"]["param_lens_model"]
        kwargs_lens_os = lens_data["os"]["kwargs_lens"]

        lens_model_ds = lens_data["ds"]["param_lens_model"]
        kwargs_lens_ds = lens_data["ds"]["kwargs_lens"]

        kappa_od, gamma_od1, gamma_od2 = self.halos_get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_od,
            lens_model=lens_model_od,
            same_from_class=False,
        )

        kappa_os, gamma_os1, gamma_os2 = self.halos_get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_os,
            lens_model=lens_model_os,
            same_from_class=False,
        )

        kappa_os2, gamma_os12, gamma_os22 = self.halos_get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_os,
            lens_model=lens_model_os,
            zdzs=(0, zs),
            same_from_class=False,
        )

        kappa_ds, gamma_ds1, gamma_ds2 = self.halos_get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_ds,
            lens_model=lens_model_ds,
            zdzs=(zd, zs),
            same_from_class=False,
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
            gext,
        ), (kwargs_lens_os, lens_model_os)

    def get_all_pars_distib(self, zd, zs):
        """Computes the distribution of external convergence (kappa_ext) and
        external shear (gamma_ext) for given deflector and source redshifts.

        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :return: An array of shape (samples_number, 2) containing the computed kappa_ext and gamma_ext values for the given deflector and source redshifts. Each row corresponds to a sample, with the first column being kappa_ext and the second column being gamma_ext.
        :rtype: numpy.ndarray

        .. note::
            The total elapsed time for computing weak-lensing maps is printed at the end.
        """

        kappa_gamma_distribution = np.empty((self.samples_number, 14))
        lens_instance = np.empty((self.samples_number, 2), dtype=object)

        loop = range(self.samples_number)

        start_time = time.time()

        for i in loop:
            self.enhance_halos_table_random_pos()
            results, lens_model_data = self.halos_various_halos_data(zd=zd, zs=zs)
            kappa_od = results["kappa_od"]
            kappa_os = results["kappa_os"]
            gamma_od1 = results["gamma_od1"]
            gamma_od2 = results["gamma_od2"]
            gamma_os1 = results["gamma_os1"]
            gamma_os2 = results["gamma_os2"]
            kappa_ds = results["kappa_ds"]
            gamma_ds1 = results["gamma_ds1"]
            gamma_ds2 = results["gamma_ds2"]
            kappa_os2 = results["kappa_os2"]
            gamma_os12 = results["gamma_os12"]
            gamma_os22 = results["gamma_os22"]
            kext = results["kext"]
            gext = results["gext"]

            kwargs_lens_os = lens_model_data["kwargs_lens_os"]
            lens_model_os = lens_model_data["lens_model_os"]
            kappa_gamma_distribution[i] = [
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
                gext,
            ]
            lens_instance[i] = [kwargs_lens_os, lens_model_os]
        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this halos lists, zd,zs, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution, lens_instance

    def compute_kappa_in_bins(self):
        """For computing a mass sheet correction. Computes the kappa values for
        each redshift bin.

        :returns: A list of kappa values for each redshift bin.
        :rtype: list[float]
        """

        # todo: different zd,zs
        bins = np.arange(0, 5.025, 0.05)
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
                        halos_ds, None, z1=0, z2=5
                    )
                    kappa, _ = self.halos_get_convergence_shear(
                        lens_model=lens_model,
                        kwargs=kwargs_lens,
                        same_from_class=False,
                        gamma12=False,
                        zdzs=(0, 5),
                    )
                    kappa_dict[bin_centers[i]] = kappa
                else:
                    kappa_dict[bin_centers[i]] = 0
            all_kappa_dicts.append(kappa_dict)
        return all_kappa_dicts

    def total_halo_mass(self):
        """Calculates the total mass of all halos.

        :returns: Total mass of all halos.
        :rtype: float

        .. note::
            The total mass is computed by summing up all the entries in the `mass_list` attribute of the class instance.
        """

        if self.n_halos == 0:
            return 0.0
        else:
            mass_list = self.mass_list
            return np.sum(mass_list)

    def total_critical_mass(self, method="differential_comoving_volume"):
        """Computes the total critical mass within a given sky area up to a
        redshift of 5 using either the total comoving volume or differential
        comoving volume method.

        The function computes the critical mass using either the `comoving_volume`
        method or the `differential_comoving_volume` method.

        :param method: The method to use for computing the critical mass. Options are:
                       - `comoving_volume`: Computes the total critical mass using the comoving volume method.
                       - `differential_comoving_volume`: Computes the total critical mass using the differential comoving volume method.
                       Default is `differential_comoving_volume`.
        :type method: str, optional
        :return: The computed total critical mass up to redshift 5.
        :rtype: float
        """

        v_ratio = self.sky_area / 41252.96

        z = np.linspace(0, 5, 2000)
        total_mass = 0.0
        if method == "comoving_volume":
            for i in range(1, len(z)):
                critical_density_z = (
                    self.cosmo.critical_density(z[i]).to("Msun/Mpc^3").value
                )
                # Compute differential comoving volume for this redshift slice
                dVc = (
                    (
                        self.cosmo.comoving_volume(z[i])
                        - self.cosmo.comoving_volume(z[i - 1])
                    )
                    * v_ratio
                ).value
                total_mass += dVc * critical_density_z
        if method == "differential_comoving_volume":
            dV_dz = (
                self.cosmo.differential_comoving_volume(z)
                * (self.sky_area * (u.deg**2))
            ).to_value("Mpc3")
            dV = dV_dz * ((5 - 0) / len(z))
            total_mass = np.sum(
                dV * self.cosmo.critical_density(z).to_value("Msun/Mpc3")
            )
        return total_mass  # In Msun

    def mass_divide_kcrit(self):
        """Computes the external convergence (kappa_ext) by dividing the mass
        of each halo by the critical surface density and the physical area
        corresponding to the cone opening angle at each halo's redshift.

        :returns: An array of kappa_ext values for each halo.
        :rtype: numpy.ndarray
        """
        mass_list = self.mass_list
        z = self.halos_redshift_list
        cone_opening_angle = deg2_to_cone_angle(self.sky_area)
        area = []
        sigma_crit = []
        lens_cosmo = self.param_lens_cosmo[: self.n_halos]
        for i in range(len(lens_cosmo)):
            sigma_crit.append(lens_cosmo[i].sigma_crit)
        for z_val in z:
            area_val = cone_radius_angle_to_physical_area(
                cone_opening_angle, z_val, self.cosmo
            )
            area.append(area_val)
        area_values = [a.value for a in area]
        mass_list_values = np.array(mass_list).flatten()
        mass_d_area = np.divide(np.array(mass_list_values), np.array(area_values))
        kappa_ext = np.divide(mass_d_area, sigma_crit)
        assert kappa_ext.ndim == 1
        return kappa_ext

    def kappa_divergence(
        self,
        diff=0.0000001,
        num_points=500,
        diff_method="square",
        kwargs=None,
        lens_model=None,
        mass_sheet=None,
    ):
        """Calculates the divergence of kappa values across a specified sky
        area.

        :param diff: The differential used in the computation of kappa. Defaults to 0.0000001.
        :type diff: float, optional
        :param num_points: The number of points along each axis in the grid used for calculations. Defaults to 500.
        :type num_points: int, optional
        :param diff_method: The method used to compute the differential. Defaults to "square".
        :type diff_method: str, optional
        :param kwargs: Keyword arguments for the lens model. If None, the method will use `get_halos_lens_kwargs` to generate them. Defaults to None.
        :type kwargs: dict, optional
        :param lens_model: The lens model to use for calculations. If None, the method will use the class's lens model. Defaults to None.
        :type lens_model: LensModel, optional
        :param mass_sheet: Specifies whether to use the mass sheet correction. If not None, it will update the class's `mass_sheet` attribute. Defaults to None.
        :type mass_sheet: bool, optional
        :returns: The calculated two-sigma divergence of kappa values.
        :rtype: float
        """

        if mass_sheet is not None:
            self.mass_sheet = mass_sheet

        if kwargs is None:
            kwargs = self.get_halos_lens_kwargs()
        if lens_model is None:
            lens_model = self.param_lens_model

        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806
        x = np.linspace(-radius_arcsec / 2.0, radius_arcsec / 2.0, num_points)
        y = np.linspace(-radius_arcsec / 2.0, radius_arcsec / 2.0, num_points)
        X, Y = np.meshgrid(x, y)
        mask_2D = X**2 + Y**2 <= radius_arcsec**2
        mask_1D = mask_2D.ravel()

        # Use lenstronomy utility to make grid
        x_grid, y_grid = make_grid(
            numPix=num_points, deltapix=2 * (radius_arcsec / 2.0) / num_points
        )
        x_grid, y_grid = x_grid[mask_1D], y_grid[mask_1D]

        # Calculate the kappa values
        kappa_values = lens_model.kappa(
            x_grid, y_grid, kwargs, diff=diff, diff_method=diff_method
        )
        std_dev = np.std(kappa_values)

        # Calculate the kappa divergence:
        twosig = 2 * std_dev

        return twosig

    def compute_kappa_gamma(self, i, gamma_tot, diff, diff_method):
        """Compute the convergence and shear values for a given index.

        This method is designed to be used with multiprocessing to speed up the process.

        :param i: Index of the sample for which the computation will be done.
        :type i: int
        :param gamma_tot: If True, the function will return total shear gamma. If False, it will return gamma1 and gamma2.
        :type gamma_tot: bool
        :param diff: Differential used in the computation of the Hessian matrix.
        :type diff: float
        :param diff_method: Method used to compute differential.
        :type diff_method: str
        :return: A list containing kappa and either gamma or gamma1 and gamma2, based on the value of `gamma_tot`.
        :rtype: list

        .. note::
            This function is designed to work in conjunction with `get_kappa_gamma_distib` which uses multiprocessing
            to compute the kappa and gamma values for multiple samples in parallel.
        """

        self.enhance_halos_table_random_pos()

        if gamma_tot:
            kappa, gamma = self.halos_get_convergence_shear(
                gamma12=False, diff=diff, diff_method=diff_method
            )
            return [kappa, gamma]
        else:
            kappa, gamma1, gamma2 = self.halos_get_convergence_shear(
                gamma12=True, diff=diff, diff_method=diff_method
            )
            return [kappa, gamma1, gamma2]

    def get_kappa_gamma_distib(
        self, gamma_tot=False, diff=1.0, diff_method="square", listmean=False
    ):
        """Computes and returns the distribution of convergence and shear
        values.

        This method uses multiprocessing to compute the convergence and shear values for multiple samples in parallel.

        :param listmean: The boolean if average convergence (kappa) to 0.
        :type listmean: bool
        :param gamma_tot: If True, the function will return total shear gamma values. If False, it will return gamma1 and gamma2 values.
                          Default is False.
        :type gamma_tot: bool, optional
        :param diff: Differential used in the computation of the Hessian matrix. Default is 0.0001.
        :type diff: float, optional
        :param diff_method: Method used to compute differential. Default is `square`.
        :type diff_method: str, optional
        :returns: A 2D array containing kappa and either gamma or gamma1 and gamma2 for each sample, based on the value of `gamma_tot`.
        :rtype: numpy.ndarray

        .. note::
            The method uses the `compute_kappa_gamma` static method to compute the values for each sample.
            If the number of samples exceeds 2000, a print statement will indicate the elapsed time for computation.
        """

        kappa_gamma_distribution = np.empty(
            (self.samples_number, 2 if gamma_tot else 3)
        )
        start_time = time.time()

        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                self.compute_kappa_gamma,
                [(i, gamma_tot, diff, diff_method) for i in range(self.samples_number)],
            )

        for i, result in enumerate(results):
            kappa_gamma_distribution[i] = result

        if self.samples_number > 2000:
            elapsed_time = time.time() - start_time
            print(
                f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )
        if listmean:
            kappa_gamma_distribution[:, 0] = convergence_mean_0(
                kappa_gamma_distribution[:, 0]
            )
        return kappa_gamma_distribution

    def get_kappa_gamma_distib_without_multiprocessing(
        self, gamma_tot=False, diff=1.0, diff_method="square", listmean=False
    ):
        """Runs the method get_convergence_shear() a specific number of times
        and stores the results for kappa, gamma1, and gamma2 in separate lists.

        :param listmean: the boolean if average convergence (kappa) to 0
        :type listmean: bool
        :param diff_method: The method to compute differential. Default is `square`.
        :type diff_method: str, optional
        :param diff: Differential used in the computation of the Hessian matrix. Default is 1.0.
        :type diff: float, optional
        :param gamma_tot: If True, the function will return gamma values in place of gamma1 and gamma2 values. Default is False.
        :type gamma_tot: bool, optional
        :return: If gamma_tot is False, the returned list contains three lists with kappa, gamma1, and gamma2 values for each sample, respectively. If gamma_tot is True, the returned list contains two lists with kappa and gamma values for each sample, respectively.
        :rtype: list of lists
        :notes: This function assumes the method get_convergence_shear() is implemented, and it returns a 4-tuple: (kappa, gamma1, gamma2, gamma). If gamma_tot parameter is False, gamma1 and gamma2 are stored, otherwise gamma is stored. All returned values from get_convergence_shear() are assumed to be floats.
        """

        kappa_gamma_distribution = np.empty(
            (self.samples_number, 2 if gamma_tot else 3)
        )

        loop = range(self.samples_number)

        start_time = time.time()

        if gamma_tot:
            for i in loop:
                self.enhance_halos_table_random_pos()
                kappa, gamma = self.halos_get_convergence_shear(
                    gamma12=False, diff=diff, diff_method=diff_method
                )
                kappa_gamma_distribution[i] = [kappa, gamma]
            if listmean:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0])
                kappa_gamma_distribution[:, 0] -= kappa_mean
        else:
            for i in loop:
                self.enhance_halos_table_random_pos()
                kappa, gamma1, gamma2 = self.halos_get_convergence_shear(
                    gamma12=True, diff=diff, diff_method=diff_method
                )
                kappa_gamma_distribution[i] = [kappa, gamma1, gamma2]
            if listmean:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0])
                kappa_gamma_distribution[:, 0] -= kappa_mean

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution
