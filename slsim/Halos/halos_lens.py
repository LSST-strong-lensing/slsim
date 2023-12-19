from lenstronomy.Util.util import make_grid
import numpy as np
import astropy.units as u
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import warnings
from tqdm.notebook import tqdm
import math
import time
import multiprocessing
from collections.abc import Iterable
import matplotlib.pyplot as plt


def deg2_to_cone_angle(solid_angle_deg2):
    """Convert solid angle in square degrees to half cone angle in radians.

    Parameters
    ----------
    solid_angle_deg2 : float
        The solid angle in square degrees to be converted.

    Returns
    -------
    float
        The cone angle in radians corresponding to the provided solid angle.

    Notes
    -----
    This function calculates the cone angle using the relationship between
    the solid angle in steradians and the cone's apex angle.
    """
    solid_angle_sr = solid_angle_deg2 * (np.pi / 180) ** 2
    theta = np.arccos(1 - solid_angle_sr / (2 * np.pi))  # rad
    return theta


def cone_radius_angle_to_physical_area(radius_rad, z, cosmo):
    """Convert cone radius angle to physical area at a given redshift.

    Parameters
    ----------
    radius_rad : float
        The half cone's angle in radians.
    z : float
        The redshift at which the physical area is to be calculated.
    cosmo : astropy.Cosmology instance
        The cosmology used for the conversion.

    Returns
    -------
    float
        The physical area in Mpc^2 corresponding to the given cone radius and redshift.

    Notes
    -----
    The function calculates the physical area of a patch of the sky with
    a specified cone angle and redshift using the angular diameter distance.
    """
    physical_radius = cosmo.angular_diameter_distance(z) * radius_rad  # Mpc
    area_physical = np.pi * physical_radius ** 2
    return area_physical  # in Mpc2


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
    if isinstance(mass, (list, np.ndarray)) and len(mass) == 1:
        mass = mass[0]
    c_200 = A * ((1 + z) ** d) * (mass ** m)
    c_200 = np.maximum(c_200, 1)
    return c_200
    # TODO: Make this able for list


def compute_kappa(args):
    """
    Helper function to compute convergence for a given set of arguments.

    Parameters
    ----------
    args : tuple
        A tuple containing parameters needed to compute the convergence for a specific (i, j) point.

    Returns
    -------
    i, j, kappa : tuple
        Returns the indices (i, j) and the computed convergence value or None if the point lies outside the defined sky area.
    """
    i, j, X, Y, mask, instance, kwargs, diff, diff_method, lens_model, zdzs = args
    if mask[i, j]:
        return i, j, instance.xy_convergence(
            x=X[i, j], y=Y[i, j], kwargs=kwargs, diff=diff, diff_method=diff_method,
            lens_model=lens_model, zdzs=zdzs
        )
    else:
        return i, j, None


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
    halos_list : astropy.Table
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
            z_source=5,
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
        if mass_sheet and mass_correction_list is None:
            warnings.warn("Mass sheet correction is not applied")
        if not mass_sheet:
            mass_correction_list = {}
            self.n_correction = 0
            self.mass_sheet_correction_redshift = mass_correction_list.get("z", [])
            #            self.mass_first_moment = mass_correction_list.get("first_moment", [])
            self.mass_sheet_kappa = mass_correction_list.get("kappa", [])
        else:
            self.n_correction = len(mass_correction_list)
            self.mass_sheet_correction_redshift = mass_correction_list["z"]
            #            D
            self.mass_sheet_kappa = mass_correction_list["kappa"]
        self.z_source = z_source
        self.halos_list = halos_list
        self.mass_correction_list = mass_correction_list
        self.mass_sheet = mass_sheet
        self.n_halos = len(self.halos_list)
        self.sky_area = sky_area
        self.halos_redshift_list = halos_list["z"]
        self.mass_list = halos_list["mass"]
        self.samples_number = samples_number
        self._z_source_convention = (
            5  # if this need to be changed, change it in the halos.py too
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
        c_200 = [concentration_from_mass(z=zi, mass=mi)
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
                    z_source=self.z_source,
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
                    z_source=self.z_source,
                    z_source_convention=self._z_source_convention,
                )
        else:
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
        if n_halos == 0:
            return [], []
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
        """Constructs and returns the list of keyword arguments for each halo
        to be used in the lens model for lenstronomy.

        Returns
        -------
        kwargs_halos : list of dicts
            The list of dictionaries containing the keyword arguments for each halo.!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        if self.mass_sheet and self.n_correction > 0:
            Rs_angle, alpha_Rs = self.get_nfw_kwargs()
            #    first_moment = self.mass_first_moment
            #    kappa = self.kappa_ext_for_mass_sheet(self.mass_sheet_correction_redshift,
            #                                          self.lens_cosmo[-self.n_correction:], first_moment)
            kappa = self.mass_sheet_kappa
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
        if self.n_halos == 1:
            is_nan = np.isnan(self.halos_list['z'])
            if is_nan:
                if gamma12:
                    return 0, 0, 0
                else:
                    return 0, 0
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

    def get_kappa_gamma_distib(self, gamma_tot=False, diff=1.0, diff_method="square", listmean=False):
        """Computes and returns the distribution of convergence and shear
        values.

        This method uses multiprocessing to compute the convergence and shear values for multiple samples in parallel.

        Parameters
        ----------
        listmean
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
        if listmean:
            if gamma_tot:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0])
                kappa_gamma_distribution[:, 0] -= kappa_mean
            else:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0, 0])
                kappa_gamma_distribution[:, 0, 0] -= kappa_mean
        return kappa_gamma_distribution
        # TODO: Maybe considering a choice between multiprocessing and not multiprocessing.

    def get_kappa_gamma_distib_without_multiprocessing(
            self, gamma_tot=False, diff=1.0, diff_method="square", listmean=False
    ):
        """Runs the method get_convergence_shear() a specific number of times
        and stores the results for kappa, gamma1, and gamma2 in separate lists.

        Parameters
        ----------
        listmean
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
            if listmean:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0])
                kappa_gamma_distribution[:, 0] -= kappa_mean
        else:
            for i in loop:
                self.enhance_halos_table_random_pos()
                kappa, gamma1, gamma2 = self.get_convergence_shear(
                    gamma12=True, diff=diff, diff_method=diff_method
                )
                kappa_gamma_distribution[i] = [kappa, gamma1, gamma2]
            if listmean:
                kappa_mean = np.mean(kappa_gamma_distribution[:, 0, 0])
                kappa_gamma_distribution[:, 0, 0] -= kappa_mean

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

        If `self.mass_correction_list` is {}, all returned values will be None.

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
        if not self.mass_correction_list:
            return None, None, None
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
        n_mass_correction = len(mass_correction) if mass_correction is not None and self.mass_sheet else 0
        z_halo = halos["z"]
        mass_halo = halos["mass"]
        px_halo = halos["px"]
        py_halo = halos["py"]
        c_200_halos = halos["c_200"]

        if (mass_correction is not None) and (len(mass_correction) > 0) and self.mass_sheet:  # check
            z_mass_correction = mass_correction["z"]
            #    mass_first_moment = mass_correction["first_moment"]
            mass_correction_kappa = mass_correction["kappa"]
        else:
            z_mass_correction = []
            #    mass_first_moment = []
            mass_correction_kappa = []
        combined_redshift_list = np.concatenate((z_halo, z_mass_correction))
        # If this above code need to be changed, notice the change in the following code
        # including the lens_cosmo_dict one since it assume halos is in front of mass sheet
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

        lens_cosmo_dict = self._build_lens_cosmo_dict(combined_redshift_list, zs)
        lens_model, lens_model_list = self._build_lens_model(
            combined_redshift_list, zs, n_halos
        )

        if mass_correction is not None and len(mass_correction) > 0 and self.mass_sheet:  # check
            #    kappa_ext_list = self.kappa_ext_for_mass_sheet(
            #        z_mass_correction, relevant_lens_cosmo_list, mass_first_moment
            #    )
            kappa_ext_list = mass_correction_kappa
        else:
            kappa_ext_list = []

        lens_cosmo_list = list(lens_cosmo_dict.values())
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
        # Note: If MASS_MOMENT (moment),this need to be change
        return lens_model, lens_cosmo_list, kwargs_lens

    def kappa_ext_for_mass_sheet(self, z, lens_cosmo, first_moment):
        """
        Deprecated
        """
        cone_opening_angle = deg2_to_cone_angle(self.sky_area)
        # TODO: make it possible for other geometry model
        area = []
        sigma_crit = []
        for i in range(len(lens_cosmo)):
            sigma_crit.append(lens_cosmo[i].sigma_crit)

        for z_val in z:
            area_val = cone_radius_angle_to_physical_area(cone_opening_angle, z_val, self.cosmo)
            area.append(area_val)
        area_values = [a.value for a in area]
        if isinstance(first_moment[0], np.void):
            first_moment_values = [entry['first_moment'] for entry in first_moment]
        else:
            first_moment_values = first_moment
        first_moment_d_area = np.divide(np.array(first_moment_values), np.array(area_values))
        kappa_ext = np.divide(first_moment_d_area, sigma_crit)
        assert kappa_ext.ndim == 1
        return -kappa_ext

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
        if n_halos == 0 and ("CONVERGENCE" not in lens_model_list):
            return None
        elif n_halos == 0 and ("CONVERGENCE" in lens_model_list):
            return [
                {"kappa": kappa_ext_list[h], "ra_0": 0, "dec_0": 0}
                for h in range(n_mass_correction)
            ]
        if n_halos != 0:
            assert len(z_halo) == len(lens_cosmo_list[:n_halos])
        Rs_angle, alpha_Rs = self.get_nfw_kwargs(
            z=z_halo,
            mass=mass_halo,
            n_halos=n_halos,
            lens_cosmo=lens_cosmo_list[:n_halos],
            c=c_200_halos
        )
        # TODO: different lens_cosmo ( for halos and sheet )

        if ("CONVERGENCE" in lens_model_list):
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
            self.enhance_halos_table_random_pos()
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
            self.enhance_halos_table_random_pos()
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
                      enhance_pos=False,
                      ):
        if mass_sheet is not None:
            self.mass_sheet = mass_sheet

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
        Temporary changes made to the instance (like `mass_sheet`) are reverted at the end of the function.
        """
        import matplotlib.pyplot as plt

        original_mass_sheet = self.mass_sheet
        radial = False
        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806

        if kwargs is None:
            kwargs = self.get_halos_lens_kwargs()

        try:
            kappa_image, _ = self.compute_kappa(diff=diff,
                                                num_points=num_points,
                                                diff_method=diff_method,
                                                kwargs=kwargs,
                                                lens_model=lens_model,
                                                mass_sheet=mass_sheet,
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
                                 zdzs=None):
        r"""
        Compares and plots the convergence for different configurations of the mass sheet.

        This function invokes the `plot_convergence` method three times, with different
        configurations for the `mass_sheet` arguments,
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
        2. `mass_sheet` set to `True` set to `True`.
        3. `mass_sheet` set to `True` set to `False`.

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
                              zdzs=zdzs,
                              mass_sheet=False,
                              enhance_pos=False
                              )
        print('mass_sheet=True')
        self.plot_convergence(diff=diff,
                              diff_method=diff_method,
                              kwargs=kwargs,
                              lens_model=lens_model,
                              zdzs=zdzs,
                              mass_sheet=True,
                              enhance_pos=False)


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

    def get_kappa_mass_relation(
            self, diff=1.0, diff_method="square"
    ):
        kappa_mean, kappa_2sigma = self.get_kappa_mean_range(diff=diff, diff_method=diff_method)
        mass = self.total_halo_mass()
        mass_divide_kcrit = self.mass_divide_kcrit()
        mass_divide_kcrit_tot = np.sum(mass_divide_kcrit)
        if mass_divide_kcrit_tot>5.0:
            mass_divide_kcrit_tot=5.1
        return kappa_mean, kappa_2sigma, mass, mass_divide_kcrit_tot

    def plot_convergence_test(self,
                              diff=0.0000001,
                              num_points=500,
                              diff_method="square",
                              kwargs=None,
                              lens_model=None,
                              mass_sheet=None,
                              enhance_pos=True,
                              ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        self.enhance_halos_pos_to0()
        original_mass_sheet = self.mass_sheet
        radial = False
        radius_arcsec = deg2_to_cone_angle(self.sky_area) * 206264.806
        kwargs = self.get_halos_lens_kwargs()
        try:
            kappa_image, _ = self.compute_kappa(diff=diff,
                                                num_points=num_points,
                                                diff_method=diff_method,
                                                kwargs=kwargs,
                                                lens_model=lens_model,
                                                mass_sheet=mass_sheet,
                                                enhance_pos=False, )

            colors = [(1, 0, 0, 1)] + [(plt.cm.viridis(i)) for i in range(1, 256)]
            new_cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

            plt.imshow(kappa_image, cmap=new_cmap,
                       extent=[-radius_arcsec, radius_arcsec, -radius_arcsec, radius_arcsec])
            plt.colorbar(label=r'$\kappa$')
            plt.title(f'Convergence Plot, radius is {radius_arcsec} arcsec')
            plt.xlabel('x-coordinate (arcsec)')
            plt.ylabel('y-coordinate (arcsec)')
            # plt.legend() # Only include if you have legend elements
            plt.show()

        finally:
            self.mass_sheet = original_mass_sheet
            if enhance_pos:
                self.enhance_halos_table_random_pos()

    def enhance_halos_pos_to0(self):
        n_halos = self.n_halos
        px = np.array([0 for _ in range(n_halos)]).T
        py = np.array([0 for _ in range(n_halos)]).T
        # Adding the computed attributes to the halos table
        self.halos_list['px'] = px
        self.halos_list['py'] = py

    def mass_divide_kcrit(self):
        mass_list = self.mass_list
        z = self.halos_redshift_list
        cone_opening_angle = deg2_to_cone_angle(self.sky_area)
        # TODO: make it possible for other geometry model
        area = []
        sigma_crit = []
        lens_cosmo=self.lens_cosmo[:self.n_halos]
        for i in range(len(lens_cosmo)):
            sigma_crit.append(lens_cosmo[i].sigma_crit)
        for z_val in z:
            area_val = cone_radius_angle_to_physical_area(cone_opening_angle, z_val, self.cosmo)
            area.append(area_val)
        area_values = [a.value for a in area]
        mass_list_values = np.array(mass_list).flatten()
        mass_d_area = np.divide(np.array(mass_list_values), np.array(area_values))
        kappa_ext = np.divide(mass_d_area, sigma_crit)
        assert kappa_ext.ndim == 1
        return kappa_ext
