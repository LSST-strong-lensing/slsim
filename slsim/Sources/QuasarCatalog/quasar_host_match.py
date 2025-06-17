import numpy as np
from astropy.table import Table, hstack
from scipy.interpolate import interp1d
from tqdm import tqdm

def sample_eddington_rate(z, z0=0.6, gamma_e=-0.65, gamma_z=3.47, A=0.00071,
                              lambda_min=0.1, lambda_max=1.0, size=1, n_grid=1000):
    """
    Sample Eddington ratios from a redshift-dependent power-law distribution.
    This function generates random samples of Eddington ratios (lambda) following
    a power-law distribution with redshift evolution. The probability density 
    function is given by:
    P(lambda|z) = A * (1+z)/(1+z0)^gamma_z * lambda^gamma_e
    
    :param z: Redshift at which to sample Eddington ratios
    :type z: float
    :param z0: Reference redshift for redshift evolution (default: 0.6)
    :type z0: float, optional
    :param gamma_e: Power-law index for Eddington ratio dependence (default: -0.65)
    :type gamma_e: float, optional
    :param gamma_z: Power-law index for redshift evolution (default: 3.47)
    :type gamma_z: float, optional
    :param A: Normalization constant (default: 0.00071)
    :type A: float, optional
    :param lambda_min: Minimum Eddington ratio to sample (default: 0.1)
    :type lambda_min: float, optional
    :param lambda_max: Maximum Eddington ratio to sample (default: 1.0)
    :type lambda_max: float, optional
    :param size: Number of samples to generate (default: 1)
    :type size: int, optional
    :param n_grid: Number of grid points for numerical CDF calculation (default: 1000)
    :type n_grid: int, optional

    :return: Sampled Eddington ratio(s). Returns float if size=1, otherwise numpy array
    :rtype: numpy.ndarray or float
    
    Notes
    -----
    The default parameters are based on observational constraints from
    quasar luminosity function studies. (See Eq. 16 in Korytov et al. 2019, https://arxiv.org/abs/1907.06530)
    """
    # Can grid in log-space to resolve low-end accurately
    lambda_grid = np.linspace(lambda_min, lambda_max, n_grid)
    # Redshift-dependent prefactor
    prefactor = A*(1 + z) / (1 + z0)**gamma_z
    pdf = prefactor * lambda_grid**gamma_e
    # Cumulative distribution function (numerical)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]  # Normalize to [0, 1]
    # Inverse CDF interpolation
    inv_cdf = interp1d(cdf, lambda_grid, bounds_error=False)
    # Sample uniformly in [0,1]
    u = np.random.uniform(0, 1, size)
    return inv_cdf(u)

def black_hole_mass_from_vel_disp(sigma_e, 
                                  alpha=4.38, beta=0.310):
    """
    Calculate black hole mass from bulge velocity dispersion using the
    relationship derived from the M-sigma relation.

    :param sigma_e: Bulge Velocity dispersion in km/s
    :type sigma_e: float
    :param alpha: Power-law index (default: 4.38)
    :type alpha: float, optional
    :param beta: Normalization constant (default: 0.310)
    :type beta: float, optional
    :return: Black hole mass in solar masses
    :rtype: float

    Notes
    -----
    See Kormendy, J. and Ho, L. C. (2013) "The Coevolution of Supermassive Black Holes and Host Galaxies"
    """
    rslt = 10**9 * beta * (sigma_e / 200)**(alpha)
    return rslt

def calculate_lsst_magnitude(lsst_band, black_hole_mass_msun, eddington_ratio):
    """
    Calculates the absolute magnitude of a quasar in a given LSST band.

    The calculation proceeds in three main steps:
    1.  Calculate the Eddington luminosity based on the black hole mass.
    2.  Calculate the bolometric luminosity from the Eddington ratio.
    3.  Convert the bolometric luminosity to an absolute magnitude in the
        specified LSST band using a bolometric correction. (Ref. Runnoe+ 2012 https://ui.adsabs.harvard.edu/abs/2012MNRAS.422..478R/abstract)

    Args:
        lsst_band (str): The desired LSST band. Must be one of
                         ['u', 'g', 'r', 'i', 'z', 'y'].
        black_hole_mass_msun (float): The mass of the black hole in solar masses (M_sun).
        eddington_ratio (float): The Eddington ratio (L_bol / L_edd).

    Returns:
        float: The absolute magnitude of the quasar in the specified LSST band.
               Returns None if an invalid band is provided.
               
    Raises:
        ValueError: If the lsst_band is not a valid LSST band.
    """

    # Eddington Luminosity from the black hole mass.
    L_Edd = 3.2e4 * black_hole_mass_msun  # L_sun

    # Bolometric Luminosity from the Eddington ratio.
    L_bol = L_Edd * eddington_ratio # L_sun

    # Absolute Bolometric Magnitude (M_bol)
    M_bol_sun = 4.74
    M_bol = M_bol_sun - 2.5 * np.log10(L_bol)  # L_sun
    
    # Bolometric corrections for LSST bands.
    # Effective wavelengths for LSST bands (approximate):
    # u: 365 nm, g: 480 nm, r: 622 nm, i: 754 nm, z: 869 nm, y: 971 nm
    bolometric_corrections = {
        # Corresponds to Runnoe+ (2012) BC for 3000 Å (300 nm) as a UV proxy
        'u': 5.2,
        # Corresponds to Runnoe+ (2012) BC for 5100 Å (510 nm) as an optical proxy
        'g': 8.1,
        'r': 8.1,
        'i': 8.1,
        'z': 8.1,
        'y': 8.1,
    } #TODO: use AMOEBA to get the bolometric corrections for each band

    if lsst_band not in bolometric_corrections:
        raise ValueError(f"Invalid LSST band '{lsst_band}'. Must be one of {list(bolometric_corrections.keys())}")

    bc_band = bolometric_corrections[lsst_band]
    
    # The relationship is typically defined as L_bol = BC * L_band, where BC is a factor.
    # In magnitudes, this becomes M_band = M_bol + 2.5 * log10(BC_factor)
    # The Runnoe+ (2012) values are factors (zeta), not magnitude differences.
    # So we calculate M_band = M_bol + 2.5*log10(zeta).
    # This is equivalent to M_band = M_bol - BC_mag, where BC_mag = -2.5*log10(zeta)
    M_band = M_bol + 2.5 * np.log10(bc_band)

    return M_band


class QuasarHostMatch:
    """Class to generate a host galaxy catalog for a given quasar
    catalog."""

    def __init__(
        self,
        quasar_catalog,
        galaxy_catalog,
    ):
        """

        :param quasar_catalog: quasar catalog with redshifts and absolute magnitude in "i" band
        :type quasar_catalog: astropy Table
        :param galaxy_catalog: quasar host galaxy candidate catalog
        :type galaxy_catalog: astropy Table
        """
        self.quasar_catalog = quasar_catalog
        self.galaxy_catalog = galaxy_catalog

    def match(self):
        """Generates catalog in which quasars are matched with host galaxies.

        :return: catalog with quasar redshifts and their corresponding host galaxies
        :return type: astropy Table
        """
        # Create a tuple of data types (dtype_tuple).
        dtype_tuple = tuple(["float64"] * len(self.galaxy_catalog.colnames))
        # Convert 3rd element to object from float64.
        dtype_tuple = dtype_tuple[:2] + ("object",) + dtype_tuple[3:]
        # Create a new Table object where quasars and their host galaxy will be stored.
        # This table will be created dynamically with the format of the given galaxy catalog.
        matched_catalog = Table(
            names=(tuple(self.galaxy_catalog.colnames)),
            dtype=dtype_tuple,
        )
        # Specify appropriate redshift range based on galaxy catalog sky area (1 deg^2 ~ 1e6
        # galaxies).
        if len(self.galaxy_catalog) > 1e6:
            z_range = 0.05 / 2
        else:
            z_range = 0.05

        # check if galaxy catalog has vel_disp column.
        if "vel_disp" not in self.galaxy_catalog.colnames:
            raise ValueError(
                "Galaxy catalog must have 'vel_disp' column to calculate host galaxies."
            )

        # prepare a blank catalog with 0 rows to store matched galaxies.
        matched_galaxies = self.galaxy_catalog.copy()[:0]

        # Iterate through the quasar catalog.
        for redshift, M_i in tqdm(zip(
            self.quasar_catalog["z"],
            self.quasar_catalog["M_i"]
        ), total=len(self.quasar_catalog), desc="Matching quasars with host galaxies"):

            # Select host galaxy candidates in the specified redshift range.
            host_galaxy_candidates = self.galaxy_catalog[
                (self.galaxy_catalog["z"] >= (redshift - z_range))
                & (self.galaxy_catalog["z"] <= (redshift + z_range))
            ]

            # compute the BH mass from the velocity dispersion, append it to the host galaxy candidates.
            bh_masses = black_hole_mass_from_vel_disp(
                host_galaxy_candidates["vel_disp"]
            )

            # Calculate the Eddington ratio for each host galaxy candidate.
            eddington_ratios = sample_eddington_rate(
                redshift,
                size=len(host_galaxy_candidates)
            )

            # use both to calculate the absolute magnitude in the "i" band.
            quasar_abs_magnitudes_i_band = [
                calculate_lsst_magnitude(
                    'i',
                    bh_mass,
                    eddington_ratio
                )
                for bh_mass, eddington_ratio in zip(bh_masses, eddington_ratios)
            ]

            # select host galaxy with closest quasar absolute magnitude in "i" band.
            closest_index = np.argmin(
                np.abs(M_i - quasar_abs_magnitudes_i_band)
            )
            host_galaxy = host_galaxy_candidates[closest_index]
            matched_galaxies.add_row(host_galaxy)
        
        # remove 'z' column from the matched galaxies.
        matched_galaxies.remove_column("z")

        # concatenate the matched galaxies with the quasar catalog.
        matched_catalog = hstack(
            [self.quasar_catalog, matched_galaxies],
            table_names=["quasar", "host_galaxy"],
            join_type="exact",
        )

        return matched_catalog