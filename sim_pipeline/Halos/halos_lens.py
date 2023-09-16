import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import warnings
from tqdm.notebook import tqdm
import time
import multiprocessing


def concentration_from_mass(z, mass, A=75.4, d=-0.422, m=-0.089):
    """
    Get the halo concentration from halo masses using the fit in Childs et al. 2018 Eq(19),
    Table 2 for all individual Halos, both relaxed and unrelaxed.

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
    # TODO: Add test function


class HalosLens(object):
    """
    Manage lensing properties of Halos.

    Provides methods to compute lensing properties of Halos, such as their convergence and shear.

    Parameters
    ----------
    halos_list : table
    cosmo : astropy.cosmology instance, optional
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

    Notes
    -----
    This class need external libraries such as lenstronomy for its computations.
    """

    # TODO: ADD test functions
    # TODO: Add documentation for all methods, CHANGE the documentation for all methods
    def __init__(self, halos_list, mass_correction_list=None, cosmo=None, sky_area=4 * np.pi, samples_number=1000,
                 mass_sheet=True,
                 ):

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
        self.halos_redshift_list = halos_list['z']
        self.mass_list = halos_list['mass']
        self.mass_sheet_correction_redshift = mass_correction_list['z']
        self.kappa_ext_list = mass_correction_list['kappa_ext']
        # self.first_moment = mass_correction_list['first_moment']
        self.samples_number = samples_number
        self._z_source_convention = 10  # if this need to be changed, change it in the halos.py too
        if cosmo is None:
            warnings.warn("No cosmology provided, instead uses astropy.cosmology import default_cosmology")
            import astropy.cosmology as default_cosmology
            self.cosmo = default_cosmology
        else:
            self.cosmo = cosmo

        self.combined_redshift_list = np.concatenate((self.halos_redshift_list, self.mass_sheet_correction_redshift))

        self.lens_cosmo = [LensCosmo(z_lens=self.combined_redshift_list[h],
                                     z_source=self._z_source_convention,
                                     cosmo=self.cosmo)
                           for h in range(self.n_halos + self.n_correction)]

        self.lens_model = self.get_lens_model()
        # TODO: Set z_source as an input parameter or other way

    def get_lens_model(self):
        if self.mass_sheet:
            lens_model = LensModel(lens_model_list=['NFW'] * self.n_halos + ['CONVERGENCE'] * self.n_correction,
                                   lens_redshift_list=self.combined_redshift_list,
                                   cosmo=self.cosmo,
                                   observed_convention_index=[],
                                   multi_plane=True,
                                   z_source=5, z_source_convention=self._z_source_convention)
        else:
            lens_model = LensModel(lens_model_list=['NFW'] * self.n_halos,
                                   lens_redshift_list=self.halos_redshift_list,
                                   cosmo=self.cosmo,
                                   observed_convention_index=[],
                                   multi_plane=True,
                                   z_source=5, z_source_convention=self._z_source_convention)
        return lens_model

    def random_position(self):
        """
        Generates and returns random positions in the sky using a uniform distribution.

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

<<<<<<< Updated upstream
    def get_nfw_kwargs(self, z=None, mass=None, n_halos=None):
=======
    def get_nfw_kwargs(self):
>>>>>>> Stashed changes
        """
        Returns the angle at scale radius, observed bending angle at the scale radius, and positions of the Halos in
        the lens plane from physical mass and concentration parameter of an NFW profile.

        Returns
        -------
        Rs_angle, alpha_Rs, px, py : np.array
            Rs_angle (angle at scale radius) (in units of arcsec)
            alpha_Rs (observed bending angle at the scale radius) (in units of arcsec)
            Arrays containing Rs_angle, alpha_Rs, and x and y positions of all the Halos.
        """
<<<<<<< Updated upstream
        if n_halos is None:
            n_halos = self.n_halos
        Rs_angle = []
        alpha_Rs = []
        if z is None:
            z = self.halos_redshift_list
        if mass is None:
            mass = self.mass_list
        c_200 = concentration_from_mass(z=z, mass=mass)
        for h in range(n_halos):
            Rs_angle_h, alpha_Rs_h = self.lens_cosmo[h].nfw_physical2angle(M=mass[h],
=======
        n_halos = self.n_halos
        Rs_angle = []
        alpha_Rs = []
        c_200 = concentration_from_mass(z=self.halos_redshift_list, mass=self.mass_list)
        for h in range(n_halos):
            Rs_angle_h, alpha_Rs_h = self.lens_cosmo[h].nfw_physical2angle(M=self.mass_list[h],
>>>>>>> Stashed changes
                                                                           c=c_200[h])
            Rs_angle.extend(Rs_angle_h)
            alpha_Rs.extend(alpha_Rs_h)
        # TODO: Check if I need to add mulit-processing when n_halos is large (Probably not)
        px, py = np.array([self.random_position() for _ in range(n_halos)]).T
        Rs_angle = np.array(Rs_angle)
        Rs_angle = np.array(Rs_angle)
        return Rs_angle, alpha_Rs, px, py

    def get_halos_lens_kwargs(self):
        """
        Constructs and returns the list of keyword arguments for each halo to be used in the lens model for lenstronomy.

        Returns
        -------
        kwargs_halos : list of dicts
            The list of dictionaries containing the keyword arguments for each halo.!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        if self.mass_sheet:
            Rs_angle, alpha_Rs, px, py = self.get_nfw_kwargs()
            kappa = self.kappa_ext_list
            ra_0 = [0] * self.n_correction
            dec_0 = [0] * self.n_correction

            kwargs_lens = [{'Rs': Rs_angle[h], 'alpha_Rs': alpha_Rs[h], 'center_x': px[h], 'center_y': py[h]}
                           for h in range(self.n_halos)] + \
                          [{'kappa': kappa[h], 'ra_0': ra_0[h], 'dec_0': dec_0[h]} for h in range(self.n_correction)]

        else:
            Rs_angle, alpha_Rs, px, py = self.get_nfw_kwargs()

            kwargs_lens = [{'Rs': Rs_angle[h], 'alpha_Rs': alpha_Rs[h], 'center_x': px[h], 'center_y': py[h]}
                           for h in range(self.n_halos)]

        return kwargs_lens

<<<<<<< Updated upstream
    def get_convergence_shear(self, gamma12=False, diff=0.0001, diff_method='square'):
=======
    def get_convergence_shear(self, gamma12=False, diff=1.0, diff_method='square'):
>>>>>>> Stashed changes
        """
        Calculates and returns the convergence and shear at the origin due to all the Halos.

        Parameters
        ----------
        gamma12 : bool, optional
            If True, the function will return gamma1 and gamma2 instead of gamma. Default is False.
        diff : float, optional
<<<<<<< Updated upstream
            Differential used in the computation of the Hessian matrix. Default is 0.0001.
=======
            Differential used in the computation of the Hessian matrix. Default is 1.0.
>>>>>>> Stashed changes
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
        f_xx, f_xy, f_yx, f_yy = self.lens_model.hessian(0.0, 0.0, self.get_halos_lens_kwargs(),
                                                         diff=diff,
                                                         diff_method=diff_method)
        kappa = 1 / 2. * (f_xx + f_yy)
        if gamma12:
            gamma1 = 1. / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            return kappa, gamma1, gamma2
        else:
            gamma = np.sqrt(f_xy ** 2 + 0.25 * (f_xx - f_yy) ** 2)
            return kappa, gamma

    @staticmethod
    def compute_kappa_gamma(i, obj, gamma_tot, diff, diff_method):
        """
        Compute the convergence and shear values for a given index.

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
        if gamma_tot:
            kappa, gamma = obj.get_convergence_shear(gamma12=False, diff=diff, diff_method=diff_method)
            return [kappa, gamma]
        else:
            kappa, gamma1, gamma2 = obj.get_convergence_shear(gamma12=True, diff=diff, diff_method=diff_method)
            return [kappa, gamma1, gamma2]

<<<<<<< Updated upstream
    def get_kappa_gamma_distib(self, gamma_tot=False, diff=0.0001, diff_method='square'):
=======
    def get_kappa_gamma_distib(self, gamma_tot=False, diff=1.0, diff_method='square'):
>>>>>>> Stashed changes
        """
        Computes and returns the distribution of convergence and shear values.

        This method uses multiprocessing to compute the convergence and shear values for multiple samples in parallel.

        Parameters
        ----------
        gamma_tot : bool, optional
            If True, the function will return total shear gamma values. If False, it will return gamma1 and gamma2 values.
            Default is False.
        diff : float, optional
<<<<<<< Updated upstream
            Differential used in the computation of the Hessian matrix. Default is 0.0001.
=======
            Differential used in the computation of the Hessian matrix. Default is 1.0.
>>>>>>> Stashed changes
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
        kappa_gamma_distribution = np.empty((self.samples_number, 2 if gamma_tot else 3))
        start_time = time.time()

        with multiprocessing.Pool() as pool:
            results = pool.starmap(self.compute_kappa_gamma,
                                   [(i, self, gamma_tot, diff, diff_method) for i in range(self.samples_number)])

        for i, result in enumerate(results):
            kappa_gamma_distribution[i] = result

        if self.samples_number > 2000:
            elapsed_time = time.time() - start_time
            print(f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds")

        return kappa_gamma_distribution
        # TODO: Maybe considering a choice between multiprocessing and not multiprocessing.

    def get_kappa_gamma_distib_without_multiprocessing(self, gamma_tot=False, diff=1.0, diff_method='square'):
        """
        Runs the method get_convergence_shear() a specific number of times and stores the results
        for kappa, gamma1, and gamma2 in separate lists.

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

        kappa_gamma_distribution = np.empty((self.samples_number, 2 if gamma_tot else 3))

        loop = range(self.samples_number)
        if self.samples_number > 999:
            loop = tqdm(loop)

        start_time = time.time()

        if gamma_tot:
            for i in loop:
                kappa, gamma = self.get_convergence_shear(gamma12=False, diff=diff, diff_method=diff_method)
                kappa_gamma_distribution[i] = [kappa, gamma]
        else:
            for i in loop:
                kappa, gamma1, gamma2 = self.get_convergence_shear(gamma12=True, diff=diff, diff_method=diff_method)
                kappa_gamma_distribution[i] = [kappa, gamma1, gamma2]

        if self.samples_number > 999:
            elapsed_time = time.time() - start_time
            print(f"For this Halos list, elapsed time for computing weak-lensing maps: {elapsed_time} seconds")

        return kappa_gamma_distribution
<<<<<<< Updated upstream

    def filter_halos_by_redshift(self, zd):
        """
        Filters halos and mass corrections based on redshift and returns separate LensModel and LensCosmo instances
        for the regions between the observer and the deflector (od) and the deflector to the source (ds).

        Parameters
        ----------
        zd : float
            The redshift threshold to differentiate between observer-deflector (od) and deflector-source (ds) halos.

        Returns
        -------
        lens_model_ds : lenstronomy.LensModel instance
            The LensModel instance for halos in the deflector-source (ds) region.
        lens_cosmo_ds : list of lenstronomy.Cosmo.lens_cosmo.LensCosmo instances
            List of LensCosmo instances for halos in the deflector-source (ds) region.
        kwargs_lens_ds : list of dicts
            List of keyword arguments for each halo in the deflector-source (ds) region.
        lens_model_od : lenstronomy.LensModel instance
            The LensModel instance for halos in the observer-deflector (od) region.
        lens_cosmo_od : list of lenstronomy.Cosmo.lens_cosmo.LensCosmo instances
            List of LensCosmo instances for halos in the observer-deflector (od) region.
        kwargs_lens_od : list of dicts
            List of keyword arguments for each halo in the observer-deflector (od) region.

        Notes
        -----
        This function filters halos based on their redshift and splits them into two categories: those between
        the observer and the deflector (od) and those between the deflector and the source (ds).
        It then creates LensModel and LensCosmo instances for each category.
        """

        def _filter_and_build(halos_z_condition, mass_z_condition):
            """
            Filters halos and mass corrections using the provided conditions and constructs associated lenstronomy instances.

            Parameters
            ----------
            halos_z_condition : np.ndarray of bool
                Condition to filter halos based on their redshift.
            mass_z_condition : np.ndarray of bool
                Condition to filter mass corrections based on their redshift.

            Returns
            -------
            lens_model_filtered : lenstronomy.LensModel instance
                The LensModel instance based on filtered halos.
            lens_cosmo_filtered : list of lenstronomy.Cosmo.lens_cosmo.LensCosmo instances
                List of LensCosmo instances for filtered halos.
            kwargs_lens_filtered : list of dicts
                List of keyword arguments for each filtered halo.
            """
            # Filter halos by redshift
            filtered_halos = self.halos_list[halos_z_condition]
            n_halos_filtered = len(filtered_halos)
            halos_redshift_list_filtered = filtered_halos['z']
            mass_list_filtered = filtered_halos['mass']

            # Filter mass corrections by redshift (if available)
            if self.mass_correction_list is not None and self.mass_sheet:
                filtered_mass_corrections = self.mass_correction_list[mass_z_condition]
                n_correction_filtered = len(filtered_mass_corrections)
                mass_sheet_correction_redshift_filtered = filtered_mass_corrections['z']
                kappa_ext_list_filtered = filtered_mass_corrections['kappa_ext']
            else:
                n_correction_filtered = 0
                kappa_ext_list_filtered = [0]
                mass_sheet_correction_redshift_filtered = []

            # Compute combined redshift list
            combined_redshift_list_filtered = np.concatenate(
                (halos_redshift_list_filtered, mass_sheet_correction_redshift_filtered))

            if len(combined_redshift_list_filtered) == 0:
                return None

            if combined_redshift_list_filtered[0] > zd:
                z_source = 5
                z_source_convention = self._z_source_convention
            else:
                z_source = zd
                z_source_convention = None

            # Build lens_cosmo instances
            lens_cosmo_filtered = [LensCosmo(z_lens=combined_redshift_list_filtered[h],
                                             z_source=z_source,
                                             cosmo=self.cosmo)
                                   for h in range(n_halos_filtered + n_correction_filtered)]

            # Build lens_model instance
            if self.mass_sheet:
                lens_model_list_filtered = ['NFW'] * n_halos_filtered + ['CONVERGENCE'] * n_correction_filtered
            else:
                lens_model_list_filtered = ['NFW'] * n_halos_filtered

            lens_model_filtered = LensModel(lens_model_list=lens_model_list_filtered,
                                            lens_redshift_list=combined_redshift_list_filtered,
                                            cosmo=self.cosmo,
                                            observed_convention_index=[],
                                            multi_plane=True,
                                            z_source=z_source,
                                            z_source_convention=z_source_convention)

            # Construct the kwargs_lens based on filtered data
            Rs_angle, alpha_Rs, px, py = self.get_nfw_kwargs(z=halos_redshift_list_filtered,
                                                             mass=mass_list_filtered,
                                                             n_halos=n_halos_filtered)

            if self.mass_sheet:
                kwargs_lens_filtered = [{'Rs': Rs_angle[h], 'alpha_Rs': alpha_Rs[h], 'center_x': px[h],
                                         'center_y': py[h]}
                                        for h in range(n_halos_filtered)] + \
                                       [{'kappa': kappa_ext_list_filtered[h], 'ra_0': 0, 'dec_0': 0}
                                        for h in range(n_correction_filtered)]
            else:
                kwargs_lens_filtered = [
                    {'Rs': Rs_angle[h], 'alpha_Rs': alpha_Rs[h], 'center_x': px[h], 'center_y': py[h]}
                    for h in range(n_halos_filtered)]

            return lens_model_filtered, lens_cosmo_filtered, kwargs_lens_filtered

        print(len(self.halos_list))
        print(len(self.halos_list[self.halos_list['z'] > zd]))
        print(len(self.halos_list[self.halos_list['z'] < zd]))
        # Separate halos into those from deflector to source and observer to deflector
        lens_model_ds, lens_cosmo_ds, kwargs_lens_ds = _filter_and_build(halos_z_condition=self.halos_list['z'] >= zd
                                                                         , mass_z_condition=self.mass_correction_list[
                                                                                                'z'] >= zd
                                                                         )
        lens_model_od, lens_cosmo_od, kwargs_lens_od = _filter_and_build(halos_z_condition=self.halos_list['z'] < zd
                                                                         , mass_z_condition=self.mass_correction_list[
                                                                                                'z'] < zd)

        return lens_model_ds, lens_cosmo_ds, kwargs_lens_ds, lens_model_od, lens_cosmo_od, kwargs_lens_od
=======
>>>>>>> Stashed changes
