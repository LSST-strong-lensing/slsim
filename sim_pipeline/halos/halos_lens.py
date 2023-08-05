import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import warnings
from tqdm import tqdm
import time


def concentration_from_mass(z, mass, A=75.4, d=-0.422, m=-0.089):
    """
    Get the halo concentration from halo masses using the fit in Childs et al. 2018 Eq(19),
    Table 2 for all individual halos, both relaxed and unrelaxed.

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

    .. math::
        C_{200c} = A(1+z)^d M^m;

    Here, A=75.4, d=-0.422, and m=-0.089 by default. The concentration parameter cannot be less than 1,
    hence the function returns the maximum of the calculated concentration and 1.

    References
    ----------
    .. [1] Childs et al. 2018, arXiv:1804.10199, doi:10.3847/1538-4357/aabf95
    """
    c_200 = A * ((1 + z) ** d) * (mass ** m)
    c_200 = np.maximum(c_200, 1)
    return c_200
    # TODO: Add test function


class HalosLens(object):
    """
    A class to manage the lensing properties of the halos.

    Parameters
    ----------
    halos_list : table
        List of halos with ['z']&['mass'] for which the lensing properties are computed.
    cosmo : astropy.cosmology instance, optional
        The cosmology assumed in the lensing computations. If none is provided, the default astropy cosmology is used.
    sky_area : float, optional
        The total sky area over which the halos are distributed. Default is the full sky (4pi steradians).

    Attributes
    ----------
    halos_list : table
        List of halos for which the lensing properties are computed.
    n_halos : int
        The number of halos in `halos_list`.
    sky_area : float
        The total sky area in deg^2 over which the halos are distributed.
    redshift_list : array_like
        The redshifts of the halos.
    mass_list : array_like
        The masses of the halos.
    cosmo : astropy.Cosmology instance
        The cosmology assumed in the lensing computations.
    lens_model : lenstronomy.LensModel instance
                The LensModel object with a NFW profile for every halo.
    Methods
    -------
    random_position() :
        Returns random x and y coordinates in the sky, uniformly distributed within a circular sky area.
    get_nfw_kwargs() :
        Returns the scale radius, observed bending angle at the scale radius, and positions of the halos
        in the lens plane.
    get_halos_lens_kwargs() :
        Returns the list of keyword arguments for each halo to be used in the lens model.
    get_convergence_shear() :
        Returns the convergence and shear at the origin due to all the halos.
    """

    # TODO: ADD test functions
    def __init__(self, halos_list, cosmo=None, sky_area=4 * np.pi, samples_number=1000):
        self.halos_list = halos_list
        self.n_halos = len(self.halos_list)
        self.sky_area = sky_area
        self.redshift_list = halos_list['z']
        self.mass_list = halos_list['mass']
        self.samples_number = samples_number
        if cosmo is None:
            warnings.warn("No cosmology provided, instead uses astropy.cosmology import default_cosmology")
            from astropy.cosmology import default_cosmology
            self.cosmo = default_cosmology.get()
        else:
            self.cosmo = cosmo

        self.lens_model = LensModel(lens_model_list=['NFW'] * self.n_halos,
                                    lens_redshift_list=self.redshift_list,
                                    cosmo=self.cosmo,
                                    observed_convention_index=[],
                                    multi_plane=True,
                                    z_source=5
                                    )
        # TODO: Set z_source as an input parameter or other way

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
        random_radius = 3600 * np.random.uniform(0, upper_bound)
        px = random_radius * np.cos(2 * phi)
        py = random_radius * np.sin(2 * phi)
        return px, py

    def get_nfw_kwargs(self):
        """
        Returns the angle at scale radius, observed bending angle at the scale radius, and positions of the halos in
        the lens plane from physical mass and concentration parameter of an NFW profile.

        Returns
        -------
        Rs_angle, alpha_Rs, px, py : np.array
            Rs_angle (angle at scale radius) (in units of arcsec)
            alpha_Rs (observed bending angle at the scale radius) (in units of arcsec)
            Arrays containing Rs_angle, alpha_Rs, and x and y positions of all the halos.
        """
        n_halos = self.n_halos
        Rs_angle, alpha_Rs = np.empty(n_halos), np.empty(n_halos)
        px, py = np.empty(n_halos), np.empty(n_halos)
        c_200 = np.empty(n_halos)
        for h in range(n_halos):
            lens_cosmo = LensCosmo(z_lens=self.redshift_list[h], z_source=9999, cosmo=self.cosmo)
            c_200[h] = concentration_from_mass(z=self.redshift_list[h], mass=self.mass_list[h])
            Rs_angle_h, alpha_Rs_h = lens_cosmo.nfw_physical2angle(M=self.mass_list[h],
                                                                   c=c_200[h])
            px[h], py[h] = self.random_position()
            Rs_angle[h] = Rs_angle_h
            alpha_Rs[h] = alpha_Rs_h
        # TODO: Also z_source
        return Rs_angle, alpha_Rs, px, py

    def get_halos_lens_kwargs(self):
        """
        Constructs and returns the list of keyword arguments for each halo to be used in the lens model for lenstronomy.

        Returns
        -------
        kwargs_halos : list of dicts
            The list of dictionaries containing the keyword arguments for each halo.
        """
        Rs_angle, alpha_Rs, px, py = self.get_nfw_kwargs()

        kwargs_halos = [{'Rs': Rs_angle[h], 'alpha_Rs': alpha_Rs[h], 'center_x': px[h], 'center_y': py[h]}
                        for h in range(self.n_halos)]
        return kwargs_halos

    def get_convergence_shear(self):
        """
        Calculates and returns the convergence and shear at the origin due to all the halos.

        Returns
        -------
        kappa, gamma1, gamma2 : float
            The computed convergence and two components of the shear at the origin.
        """
        kappa = self.lens_model.kappa(0.0, 0.0, self.get_halos_lens_kwargs(),
                                      diff=1.0,
                                      diff_method='square')
        gamma1, gamma2 = self.lens_model.gamma(0.0, 0.0, self.get_halos_lens_kwargs(),
                                               diff=1.0,
                                               diff_method='square')
        gamma = np.sqrt(gamma1 ** 2 + gamma2 ** 2)

        return kappa, gamma, gamma1, gamma2

    # TODO: Add option to return gamma1, gamma2 OR gamma (also computer gamma only)

    def get_kappa_gamma_distib(self, gamma_tot=False):
        """
        Runs the method get_convergence_shear() a specific number of times and stores the results
        for kappa, gamma1, and gamma2 in separate lists.

        Parameters
        ----------
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
        kappa_gamma_distribution = []
        start_time = time.time()
        if gamma_tot:
            for _ in tqdm(range(self.samples_number)):
                kappa, gamma, _, _ = self.get_convergence_shear()
                kappa_gamma_distribution.append([kappa, gamma])
        else:
            for _ in tqdm(range(self.samples_number)):
                kappa, _, gamma1, gamma2 = self.get_convergence_shear()
                kappa_gamma_distribution.append([kappa, gamma1, gamma2])

        print(f"For this halos list, elapsed time for computer weak-lensing maps: {time.time() - start_time} seconds")

        return kappa_gamma_distribution
