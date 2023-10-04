import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import warnings
from tqdm.notebook import tqdm
import math
import time
import multiprocessing
from collections.abc import Iterable


def concentration_from_mass(z, mass, A=75.4, d=-0.422, m=-0.089):
    """Get the halo concentration from halo masses using the fit in Childs et
    al. 2018 Eq(19), Table 2 for all individual Halos, both relaxed and
    unrelaxed.

    Parameters
    ----------
    z : float
        Redshift of the halo.
    mass : float or array_like
        The mass of the halo in solar masses.
    A : float, optional
        The pre-factor in the concentration-mass relation. Default value is 75.4.
    d : float, optional
        The exponent for the (1+z) term in the concentration-mass relation. Default value is -0.422.
    m : float, optional
        The exponent for the mass term in the concentration-mass relation. Default value is -0.089.

    Returns
    -------
    c_200 : float or array_like
        The concentration parameter of the halo(s).

    Notes
    -----
    The function implements the following formula:

    . math::
        C_{200c} = A(1+z)^d M^m;

    Here, A=75.4, d=-0.422, and m=-0.089 by default. The concentration parameter cannot be less than 1,
    hence the function returns the maximum of the calculated concentration and 1.

    References
    ----------
    . [1] Childs et al. 2018, arXiv:1804.10199, doi:10.3847/1538-4357/aabf95
    """
    c_200 = A * ((1 + z) ** d) * (mass ** m)
    c_200 = np.maximum(c_200, 1)
    return c_200
    # TODO: Make this able for list


class HalosLens(object):
    """Manage lensing properties of Halos.

    Provides methods to compute lensing properties of Halos, such as their convergence and shear.

    Parameters
    ----------
    halos_list : table
    cosmo : astropy.Cosmology instance, optional
        Cosmology used for lensing computations. If not provided, default astropy cosmology is used.
    sky_area : float, optional
        Total sky area (in steradians) over which Halos are distributed. Defaults to full sky (4pi steradians).
    samples_number : int, optional
        Number of samples for statistical calculations. Defaults to 1000.

    Attributes
    ----------
    halos_list : table
        Table of Halos.
    n_halos : int
        Number of Halos in `halos_list`.
    sky_area : float
        Total sky area in square degrees.
    halos_redshift_list : array_like
        Redshifts of the Halos.
    mass_list : array_like
        Masses of the Halos in solar masses.
    cosmo : astropy.Cosmology instance
        Cosmology used for computations.
    lens_model : lenstronomy.LensModel instance
        LensModel with a NFW profile for each halo.


    Methods
    -------
    random_position() :
        Generate random x and y coordinates in the sky.
    get_nfw_kwargs() :
        Get scale radius, observed bending angle, and positions of Halos in lens plane.
    get_halos_lens_kwargs() :
        Get list of keyword arguments for each halo in lens model.
    get_convergence_shear(gamma12=False, diff=1.0, diff_method='square') :
        Compute convergence and shear at origin due to all Halos.
    get_kappa_gamma_distib(gamma_tot=False, diff=1.0, diff_method='square') :
        Get distribution of convergence and shear values by repeatedly sampling with multiprocessing.
    get_kappa_gamma_distib_without_multiprocessing(gamma_tot=False, diff=1.0, diff_method='square') :
        Compute and store the results for kappa, gamma1, and gamma2 in separate lists without using multiprocessing.
    filter_halos_by_redshift(zd, zs) :
        Get lens data for three different conditions based on deflector and source redshifts.
    _filter_halos_by_condition(zd, zs) :
        Filters halos and mass corrections by redshift conditions and constructs lens data.
    _filter_mass_correction_by_condition(zd, zs) :
        Filter mass corrections based on redshift conditions.
    _build_lens_data(halos, mass_correction, zd, zs) :
        Constructs lens data based on the provided halos, mass corrections, and redshifts.
    _build_lens_cosmo_list(combined_redshift_list, z_source) :
        Constructs a list of LensCosmo instances based on the provided combined redshift list and source redshift.
    _build_lens_model(combined_redshift_list, z_source, n_halos) :
        Constructs a lens model based on the provided combined redshift list, source redshift, and number of halos.
    _build_kwargs_lens(n_halos, n_mass_correction, z_halo, mass_halo, lens_model_list, kappa_ext_list, lens_cosmo_list) :
        Constructs a list of keyword arguments to define the lens model.
    get_lens_data_by_redshift(zd, zs) :
        Get lens data for three different conditions based on deflector and source redshifts.
    compute_various_kappa_gamma_values(zd, zs, gamma_tot=False, diff=1.0, diff_method='square') :
        Computes various kappa (convergence) and gamma (shear) values for given deflector and source redshifts.
    get_kext_gext_values(zd, zs) :
        Computes kappa_ext (external convergence) and gamma_ext (external shear) values for given deflector and source redshifts.
    get_kappaext_gammaext_distib_zdzs(zd, zs):
        Computes kappa_ext (external convergence) and gamma_ext (external shear) distributions for given deflector and source redshifts.
    generate_distributions_0to5() :
        Generates kappa_ext, gamma_ext distributions for a range of deflector and source redshifts from 0 to 5 for this
        given halos list.
    Notes
    -----
    This class need external libraries such as lenstronomy for its computations.
    """

    # TODO: ADD test functions
    # TODO: Add documentation for all methods, CHANGE the documentation for all methods
    def __init__(
            self,
            halos_list,
            mass_correction_list=None,
            cosmo=None,
            sky_area=4 * np.pi,
            samples_number=1000,
            mass_sheet=True,
    ):
        """Initialize the HalosLens class.

        Parameters
        ----------
        halos_list : table
            Table containing details of halos, including their redshifts and masses.
        mass_correction_list : table, optional
            Table for mass correction, containing details like redshifts and external convergences.
            Defaults to None. Ignored if `mass_sheet` is set to False.
        cosmo : astropy.Cosmology instance, optional
            Instance specifying the cosmological parameters for lensing computations.
            If not provided, the default astropy cosmology will be used.
        sky_area : float, optional
            Total sky area in steradians over which halos are distributed. Defaults to full sky (4π steradians).
        samples_number : int, optional
            Number of samples for statistical calculations. Defaults to 1000.
        mass_sheet : bool, optional
            Flag to decide whether to use the mass_sheet correction. If set to False, the mass_correction_list is ignored.
            Defaults to True.
        Parameters
        ----------
        halos_list
        mass_correction_list
        cosmo
        sky_area
        samples_number
        mass_sheet
        """
        if not mass_sheet:
            mass_correction_list = None
        if mass_sheet and mass_correction_list is None:
            warnings.warn("Mass sheet correction is not applied")

        self.halos_list = halos_list
        self.mass_correction_list = mass_correction_list
        self.mass_sheet = mass_sheet
        self.n_halos = len(self.halos_list)
        self.n_correction = len(self.mass_correction_list)
        self.sky_area = sky_area
        self.halos_redshift_list = halos_list["z"]
        self.mass_list = halos_list["mass"]
        self.mass_sheet_correction_redshift = mass_correction_list["z"]
        self.kappa_ext_list = mass_correction_list["kappa_ext"]
        self.samples_number = samples_number
        self._z_source_convention = (
            10  # if this need to be changed, change it in the halos.py too
        )
        if cosmo is None:
            warnings.warn(
                "No cosmology provided, instead uses astropy.cosmology import default_cosmology"
            )
            import astropy.cosmology as default_cosmology

            self.cosmo = default_cosmology
        else:
            self.cosmo = cosmo

        self.combined_redshift_list = np.concatenate(
            (self.halos_redshift_list, self.mass_sheet_correction_redshift)
        )

        self._lens_cosmo = None  # place-holder for lazy load
        self._lens_model = None  # same as above
        c_200 = [concentration_from_mass(z=zi, mass=mi)[0]
                 for zi, mi in zip(self.halos_redshift_list, self.mass_list)]
        self.halos_list['c_200'] = c_200

        self.enhance_halos_table_random_pos()
        # TODO: Set z_source as an input parameter or other way

    @property
    def lens_cosmo(self):
        """Lazy-load lens_cosmo."""
        if self._lens_cosmo is None:
            self._lens_cosmo = [
                LensCosmo(
                    z_lens=self.combined_redshift_list[h],
                    z_source=self._z_source_convention,
                    cosmo=self.cosmo,
                )
                for h in range(self.n_halos + self.n_correction)
            ]
        return self._lens_cosmo

    @property
    def lens_model(self):
        """Lazy-load lens_model."""
        if self._lens_model is None:  # Only compute if not already done
            self._lens_model = self.get_lens_model()
        return self._lens_model

    def enhance_halos_table_random_pos(self):
        n_halos = self.n_halos
        px, py = np.array([self.random_position() for _ in range(n_halos)]).T

        # Adding the computed attributes to the halos table
        self.halos_list['px'] = px
        self.halos_list['py'] = py

    def get_lens_model(self):
        """Create a lens model using provided halos and optional mass sheet
        correction.

        This method constructs a lens model based on the halos and (if specified)
        the mass sheet correction. The halos are modeled with the NFW profile,
        and the mass sheet correction is modeled using the CONVERGENCE profile.

        Returns
        -------
        lenstronomy.LensModel
            The lens model constructed from the provided halos and optional mass sheet correction.

        Notes
        -----
        If `mass_sheet` attribute of the class is set to True, the lens model will incorporate
        both the halos' NFW profile and the mass sheet's CONVERGENCE profile. If set to False,
        only the halos' NFW profile is used.
        """
        if self.mass_sheet:
            lens_model = LensModel(
                lens_model_list=["NFW"] * self.n_halos
                                + ["CONVERGENCE"] * self.n_correction,
                lens_redshift_list=self.combined_redshift_list,
                cosmo=self.cosmo,
                observed_convention_index=[],
                multi_plane=True,
                z_source=5,
                z_source_convention=self._z_source_convention,
            )
        else:
            lens_model = LensModel(
                lens_model_list=["NFW"] * self.n_halos,
                lens_redshift_list=self.halos_redshift_list,
                cosmo=self.cosmo,
                observed_convention_index=[],
                multi_plane=True,
                z_source=5,
                z_source_convention=self._z_source_convention,
            )
        return lens_model

    def random_position(self):
        """Generates and returns random positions in the sky using a uniform
        distribution.

        Returns
        -------
        px, py : float
            The generated random x and y coordinates inside the skyarea in arcsec.
        """
        phi = 2 * np.pi * np.random.random()
        upper_bound = np.sqrt(self.sky_area / np.pi)
        random_radius = 3600 * np.sqrt(np.random.random()) * upper_bound
        px = random_radius * np.cos(2 * phi)
        py = random_radius * np.sin(2 * phi)
        return px, py

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
        # TODO: make for divided
        # TODO: make only computed one
        # TODO: docstring

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
                M=mass[h], c=c
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
        """Constructs and returns the list of keyword arguments for each halo
        to be used in the lens model for lenstronomy.

        Returns
        -------
        kwargs_halos : list of dicts
            The list of dictionaries containing the keyword arguments for each halo.!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        if self.mass_sheet:
            Rs_angle, alpha_Rs = self.get_nfw_kwargs()
            kappa = self.kappa_ext_list
            ra_0 = [0] * self.n_correction
            dec_0 = [0] * self.n_correction

            kwargs_lens = [
                              {
                                  "Rs": Rs_angle[h],
                                  "alpha_Rs": alpha_Rs[h],
                                  "center_x": self.halos_list['px'][h],
                                  "center_y": self.halos_list['py'][h],
                              }
                              for h in range(self.n_halos)
                          ] + [
                              {"kappa": kappa[h], "ra_0": ra_0[h], "dec_0": dec_0[h]}
                              for h in range(self.n_correction)
                          ]

        else:
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
        #    print(f'zd{zdzs[0]}zs{zdzs[1]}', f_xx, f_xy, f_yx, f_yy)
        #    print('nonezdzs', lens_model.hessian(
        #        x=0.0, y=0.0, kwargs=kwargs, diff=1.0, diff_method='square'
        #    ))
        #    print('------------------------------')
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
        """Compute the convergence and shear values for a given index.

        This method is designed to be used with multiprocessing to speed up the process.

        Parameters
        ----------
        i : int
            Index of the sample for which the computation will be done.
        obj : HalosLens object
            Instance of the HalosLens class.
        gamma_tot : bool
            If True, the function will return total shear gamma. If False, it will return gamma1 and gamma2.
        diff : float
            Differential used in the computation of the Hessian matrix.
        diff_method : str
            Method used to compute differential.

        Returns
        -------
        list
            A list containing kappa and either gamma or gamma1 and gamma2, based on the value of `gamma_tot`.

        Notes
        -----
        This function is designed to work in conjunction with `get_kappa_gamma_distib` which uses multiprocessing
        to compute the kappa and gamma values for multiple samples in parallel.
        """
        self.enhance_halos_table_random_pos()

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

    def get_kappa_gamma_distib(self, gamma_tot=False, diff=1.0, diff_method="square"):
        """Computes and returns the distribution of convergence and shear
        values.

        This method uses multiprocessing to compute the convergence and shear values for multiple samples in parallel.

        Parameters
        ----------
        gamma_tot : bool, optional
            If True, the function will return total shear gamma values. If False, it will return gamma1 and gamma2 values.
            Default is False.
        diff : float, optional
            Differential used in the computation of the Hessian matrix. Default is 0.0001.
        diff_method : str, optional
            Method used to compute differential. Default is 'square'.

        Returns
        -------
        numpy.ndarray
            A 2D array containing kappa and either gamma or gamma1 and gamma2 for each sample, based on the value
            of `gamma_tot`.

        Notes
        -----
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
                [
                    (i, gamma_tot, diff, diff_method)
                    for i in range(self.samples_number)
                ],
            )

        for i, result in enumerate(results):
            kappa_gamma_distribution[i] = result

        if self.samples_number > 2000:
            elapsed_time = time.time() - start_time
            print(
                f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution
        # TODO: Maybe considering a choice between multiprocessing and not multiprocessing.

    def get_kappa_gamma_distib_without_multiprocessing(
            self, gamma_tot=False, diff=1.0, diff_method="square"
    ):
        """Runs the method get_convergence_shear() a specific number of times
        and stores the results for kappa, gamma1, and gamma2 in separate lists.

        Parameters
        ----------
        diff_method : str, optional
            The method to compute differential. Default is 'square'.
        diff: float, optional
            Differential used in the computation of the Hessian matrix. Default is 1.0.
        gamma_tot : bool, optional
            If True, the function will return gamma values in place of gamma1 and gamma2 values.
            Default is False.

        Returns
        -------
        kappa_gamma_distribution: list of lists If gamma is False, the returned list contains three
        lists with kappa, gamma1, and gamma2 values for each sample, respectively. If gamma is True, the returned
        list contains two lists with kappa and gamma values for each sample, respectively.

        Notes
        -----
        This function assumes the method get_convergence_shear() is implemented, and it returns a 4-tuple:
        (kappa, gamma1, gamma2, gamma). If gamma parameter is False, gamma1 and gamma2 are stored,
        otherwise gamma is stored. All returned values from get_convergence_shear() are assumed to be floats.
        """

        kappa_gamma_distribution = np.empty(
            (self.samples_number, 2 if gamma_tot else 3)
        )

        loop = range(self.samples_number)
        if self.samples_number > 999:
            loop = tqdm(loop)

        start_time = time.time()

        if gamma_tot:
            for i in loop:
                self.enhance_halos_table_random_pos()
                kappa, gamma = self.get_convergence_shear(
                    gamma12=False, diff=diff, diff_method=diff_method
                )
                kappa_gamma_distribution[i] = [kappa, gamma]
        else:
            for i in loop:
                self.enhance_halos_table_random_pos()
                kappa, gamma1, gamma2 = self.get_convergence_shear(
                    gamma12=True, diff=diff, diff_method=diff_method
                )
                kappa_gamma_distribution[i] = [kappa, gamma1, gamma2]

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution

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
        (
            mass_correction_od,
            mass_correction_ds,
            mass_correction_os,
        ) = self._filter_mass_correction_by_condition(zd, zs)
        return (
            self._build_lens_data(halos_ds, mass_correction_ds, zd=zd, zs=zs),
            self._build_lens_data(halos_od, mass_correction_od, zd=0, zs=zd),
            self._build_lens_data(halos_os, mass_correction_os, zd=0, zs=zs),
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

    def _filter_mass_correction_by_condition(self, zd, zs):
        """Filters the mass corrections based on redshift conditions relative
        to deflector and source redshifts.

        This internal method segregates mass corrections into three categories:
        1. Between the deflector and source redshifts (ds).
        2. From zero redshift up to the deflector redshift (od).
        3. From zero redshift up to the source redshift (os).

        If `self.mass_correction_list` is None, all returned values will be None.

        Parameters
        ----------
        - zd (float): Deflector redshift.
        - zs (float): Source redshift.

        Returns
        ----------
        - tuple:
            * mass_correction_od (DataFrame or None): Mass corrections with redshift less than the deflector redshift.
            * mass_correction_ds (DataFrame or None): Mass corrections with redshift greater than or equal to the deflector redshift and less than the source redshift.
            * mass_correction_os (DataFrame or None): Mass corrections with redshift less than the source redshift.

        Note
        ----------
        This method assumes `self.mass_correction_list` is a DataFrame containing a 'z' column that represents the redshift of each mass correction entry.
        """
        if self.mass_correction_list is None:
            return None, None
        mass_correction_ds = self.mass_correction_list[
            (self.mass_correction_list["z"] >= zd)
            & (self.mass_correction_list["z"] < zs)
            ]
        mass_correction_od = self.mass_correction_list[
            self.mass_correction_list["z"] < zd
            ]
        mass_correction_os = self.mass_correction_list[
            self.mass_correction_list["z"] < zs
            ]
        return mass_correction_od, mass_correction_ds, mass_correction_os

    def _build_lens_data(self, halos, mass_correction, zd, zs):
        """Constructs lens data based on the provided halos, mass corrections,
        and redshifts.

        Parameters
        ----------
        halos : DataFrame
            Contains information about the halos, including their redshift ('z') and mass ('mass').

        mass_correction : DataFrame or None
            Contains information about the mass correction, including redshift ('z') and kappa_ext ('kappa_ext').
            If there's no mass correction, this can be None.

        zd : float
            Begin redshift.

        zs : float
            End redshift.

        Returns
        -------
        lens_model : object
            The constructed lens model based on the provided data.

        lens_cosmo_list : list
            Thr list of lens cosmologies constructed from the combined redshift list.

        kwargs_lens : list
            The list of keyword arguments to define the lens model.

        Raises
        ------
        ValueError:
            - If source redshift (zs) is less than deflector redshift (zd).
            - If any halo's redshift is smaller than the deflector redshift.
            - If any halo's redshift is larger than the source redshift.

        Notes
        -----
        The method consolidates halos and mass corrections to determine the redshift distribution of the lens model.
        It also takes into account certain conditions and constraints related to the redshifts of halos and the source.
        """
        n_halos = len(halos)
        n_mass_correction = len(mass_correction) if mass_correction is not None else 0
        z_halo = halos["z"]
        mass_halo = halos["mass"]
        px_halo = halos["px"]
        py_halo = halos["py"]
        c_200_halos = halos["c_200"]

        if mass_correction is not None and self.mass_sheet:
            z_mass_correction = mass_correction["z"]
            kappa_ext_list = mass_correction["kappa_ext"]
        else:
            z_mass_correction = []
            kappa_ext_list = []

        combined_redshift_list = np.concatenate((z_halo, z_mass_correction))

        if not combined_redshift_list.size:
            warnings.warn(
                f"No halos OR mass correction in the given redshift range from zd={zd} to zs={zs}."
            )
            return None, None, None
        if zs < zd:
            raise ValueError(
                f"Source redshift {zs} cannot be less than deflector redshift {zd}."
            )
        if min(combined_redshift_list) < zd:
            raise ValueError(
                f"Redshift of the farthest {min(combined_redshift_list)}"
                f" halo cannot be smaller than deflector redshift{zd}."
            )
        if max(combined_redshift_list) > zs:
            raise ValueError(
                f"Redshift of the closet halo {max(combined_redshift_list)} "
                f"cannot be larger than source redshift {zs}."
            )

        lens_cosmo_list = self._build_lens_cosmo_list(combined_redshift_list, zs)
        lens_model, lens_model_list = self._build_lens_model(
            combined_redshift_list, zs, n_halos
        )
        kwargs_lens = self._build_kwargs_lens(
            n_halos,
            n_mass_correction,
            z_halo,
            mass_halo,
            px_halo,
            py_halo,
            c_200_halos,
            lens_model_list,
            kappa_ext_list,
            lens_cosmo_list,
        )

        return lens_model, lens_cosmo_list, kwargs_lens

    def _build_lens_cosmo_list(self, combined_redshift_list, z_source):
        """Constructs a list of LensCosmo instances based on the provided
        combined redshift list and source redshift.

        The method creates a LensCosmo instance for each lens redshift in the combined redshift list with the specified source redshift.

        Parameters
        ----------
        combined_redshift_list : list or array-like
            List of redshifts representing the halos and mass corrections combined.

        z_source : float
            Source redshift.

        Returns
        -------
        lens_cosmo_list : list of LensCosmo objects
            List containing LensCosmo instances initialized with the individual redshift values from the combined redshift list and the specified source redshift.

        Notes
        -----
        This method assumes the availability of a `LensCosmo` class and a `cosmo` attribute in the current instance.
        """
        return [
            LensCosmo(z_lens=z, z_source=z_source, cosmo=self.cosmo)
            for z in combined_redshift_list
        ]

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
        to any additional redshifts present in `combined_redshift_list`.
        """
        n_halos = n_halos

        if len(combined_redshift_list) - n_halos > 0:
            lens_model_list = ["NFW"] * n_halos + ["CONVERGENCE"] * (
                    len(combined_redshift_list) - n_halos
            )
        elif len(combined_redshift_list) - n_halos < 0:
            raise ValueError(
                f"Combined redshift list shorter than number of halos."
                f"{len(combined_redshift_list)} < {n_halos}"
            )
        else:
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
            n_mass_correction,
            z_halo,
            mass_halo,
            px_halo,
            py_halo,
            c_200_halos,
            lens_model_list,
            kappa_ext_list,
            lens_cosmo_list=None,
    ):
        """Constructs the lens keyword arguments based on provided input
        parameters.

        Based on the provided numbers of halos and mass corrections, redshifts, masses, and lens models, this method
        assembles the lensing keyword arguments needed for the lens model. It caters for cases with and without
        'CONVERGENCE' in the lens model list.

        Parameters
        ----------
        n_halos : int
            Number of halos.

        n_mass_correction : int
            Number of mass corrections.

        z_halo : list or array-like
            List of redshifts of halos.

        mass_halo : list or array-like
            List of halo masses.

        lens_model_list : list of str
            List of lens models ('NFW', 'CONVERGENCE', etc.).

        kappa_ext_list : list or array-like
            List of external convergence values.

        lens_cosmo_list : list of LensCosmo objects, optional
            List containing LensCosmo instances for each redshift value. Defaults to None.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing the keyword arguments for each lens model.

        Notes
        -----
        This method assumes the presence of a method `get_nfw_kwargs` in the current class that provides NFW parameters
        based on given redshifts and masses.
        """
        if n_halos == 0 and "CONVERGENCE" not in lens_model_list:
            return None
        if n_halos == 0 and "CONVERGENCE" in lens_model_list:
            return [
                {"kappa": kappa_ext_list[h], "ra_0": 0, "dec_0": 0}
                for h in range(n_mass_correction)
            ]

        Rs_angle, alpha_Rs = self.get_nfw_kwargs(
            z=z_halo, mass=mass_halo, n_halos=n_halos, lens_cosmo=lens_cosmo_list, c=c_200_halos
        )
        # TODO: different lens_cosmo

        if "CONVERGENCE" in lens_model_list:
            return [
                {
                    "Rs": Rs_angle[i],
                    "alpha_Rs": alpha_Rs[i],
                    "center_x": px_halo[i],
                    "center_y": py_halo[i],
                }
                for i in range(n_halos)
            ] + [
                {"kappa": kappa_ext_list[h], "ra_0": 0, "dec_0": 0}
                for h in range(n_mass_correction)
            ]
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
        The function implements the following formulae:

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

    def get_kappaext_gammaext_distib_zdzs(self, zd, zs):
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
            self.enhance_halos_table_random_pos()
            kappa, gamma = self.get_kext_gext_values(zd=zd, zs=zs)
            kappa_gamma_distribution[i] = [kappa, gamma]

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(
                f"For this halos lists, zd,zs, elapsed time for computing weak-lensing maps: {elapsed_time} seconds"
            )

        return kappa_gamma_distribution

    def generate_distributions_0to5(self, output_format="dict"):
        """Generates distributions of external convergence (kappa_ext) and
        external shear (gamma_ext) for a range of deflector and source
        redshifts from 0 to 5.

        Parameters
        ----------
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
                kappa_gamma_dist = self.get_kappaext_gammaext_distib_zdzs(zd=zd, zs=zs)
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
