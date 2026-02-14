from skypy.galaxies.redshift import redshifts_from_comoving_density
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy.table import Table, vstack
import os
from pathlib import Path
import speclite.filters
from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen import Quasar_sed, params_agile
from slsim.Sources.SourceCatalogues.QuasarCatalog.quasar_host_match import (
    QuasarHostMatch,
)
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline

from scipy.interpolate import RegularGridInterpolator

from slsim.ImageSimulation.image_quality_lenstronomy import get_speclite_filternames

"""
References:
Richards et al. 2005
Richards et al. 2006
Oguri & Marshall (2010)
"""


class QuasarRate(object):
    """Class to calculate quasar luminosity functions and generate quasar
    samples."""

    def __init__(
        self,
        zeta: float = 2.98,
        xi: float = 4.05,
        z_star: float = 1.60,
        alpha: float = -3.31,
        beta: float = -1.45,
        phi_star: float = 5.34e-6 * (0.70**3),
        cosmo: FlatLambdaCDM = None,
        skypy_config: str = None,
        sky_area: Quantity = None,
        noise: bool = True,
        redshifts: np.ndarray = None,
        host_galaxy_candidate: Table = None,
        use_qsogen_sed: bool = False,
        qsogen_bands: list = None,
        use_sed_interpolator: bool = True,
    ):
        """Initializes the QuasarRate class with given parameters.

        :param h: Hubble constant parameter H0/100, where H0 = 70 km s^-1 Mpc^-1.
        :type h: float
        :param zeta: (1) Best fit value of the observed evolution of the quasar luminosity function
        from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type zeta: float
        :param xi: (2) Best fit value of the observed evolution of the quasar luminosity function
        from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type xi: float
        :param z_star: (3) Best fit value of the observed evolution of the quasar luminosity
        function from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type z_star: float
        :param alpha: Bright end slope of quasar luminosity density profile.
        :type alpha: float
        :param beta: Faint end slope of quasar luminosity density profile.
        :type beta: float
        :param phi_star: Renormalization of the quasar luminosity function for a given h.
        :type phi_star: float
        :param cosmo: Cosmology object.
        :type cosmo: ~astropy.cosmology object
        :param skypy_config: path to SkyPy configuration yaml file. Default is None.
        If None, the default configuration will be used.
        :type skypy_config: string
        :param sky_area: Sky area for sampled quasars in [solid angle].
        :type sky_area: `~Astropy.units.Quantity`
        :param noise: Poisson-sample the number of galaxies in quasar density lightcone.
        :type noise: bool
        :param redshifts: Redshifts for quasar density lightcone to be evaluated at.
        :type redshifts: np.ndarray
        :param host_galaxy_candidate: Galaxy catalog in an Astropy table. This catalog
         is used to match with the supernova population. If None, the galaxy catalog is
         generated within this class.
        :type host_galaxy_candidate: `~astropy.table.Table`
        :param use_qsogen_sed: If True, uses qsogen to generate realistic SEDs and compute magnitudes.
        :param qsogen_bands: List of strings for filters (e.g., ['u', 'g', 'r', 'i', 'z', 'y', 'F062', ...]).
                             Defaults to LSST bands if None.
        :param use_sed_interpolator: If True, uses a pre-computed SED magnitude interpolator on a z, M_i grid for speed. This is only relevant if `use_qsogen_sed` is True.
        """
        self.zeta = zeta
        self.xi = xi
        self.z_star = z_star
        self.alpha = alpha
        self.beta = beta
        self.phi_star = phi_star
        self.cosmo = cosmo if cosmo is not None else FlatLambdaCDM(H0=70, Om0=0.3)
        self.skypy_config = skypy_config
        self.sky_area = (
            sky_area if sky_area is not None else Quantity(0.05, unit="deg2")
        )
        self.noise = noise
        self.redshifts = (
            np.array(redshifts) if redshifts is not None else np.linspace(0.1, 5.0, 100)
        )
        self.host_galaxy_candidate = host_galaxy_candidate

        # SED Generation Configuration
        self.use_qsogen_sed = use_qsogen_sed
        self.use_sed_interpolator = use_sed_interpolator
        if qsogen_bands is None:
            self.qsogen_bands = [
                "lsst2023-u",
                "lsst2023-g",
                "lsst2023-r",
                "lsst2023-i",
                "lsst2023-z",
                "lsst2023-y",
            ]
        else:
            # convert to speclite filter names
            qsogen_bands = get_speclite_filternames(qsogen_bands)
            # make sure it has 'lsst2023-i' for anchoring
            if "lsst2023-i" not in qsogen_bands:
                qsogen_bands.append("lsst2023-i")
            self.qsogen_bands = qsogen_bands

        # Construct the dynamic path to the data file
        base_path = Path(os.path.dirname(__file__))
        file_path = base_path / "i_band_Richards_et_al_2006.txt"

        data = np.loadtxt(file_path)

        # The data is assumed to be in two columns: redshift and K-correction
        self.redshifts_kcorr = data[:, 0]
        self.K_corrections = data[:, 1]

        # Precompute the interpolation function
        self.k_corr = interp1d(
            self.redshifts_kcorr,
            self.K_corrections,
            kind="linear",
            fill_value="extrapolate",
        )

    def k_corr_interp(self, z):
        """This function computes the k-correction for a quasar at a given
        redshift.

        :param z: Redshift value at which k correction need to be
            computed.
        :type z: float or np.array
        :return: k-correction value for given redshifts.
        """

        return self.k_corr(z) - self.k_corr(0)

    def M_star(self, z_value):
        """Calculates the break absolute magnitude of quasars for a given
        redshift according to Eq. (11) in Oguri & Marshall (2010): DOI:
        10.1111/j.1365-2966.2010.16639.x.

        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: M_star value.
        :rtype: float or np.ndarray :unit: mag
        """
        z_value = np.atleast_1d(z_value)
        denominator = (
            np.sqrt(np.exp(self.xi * z_value)) + np.sqrt(np.exp(self.xi * self.z_star))
        ) ** 2
        result = (
            -20.90
            + (5 * np.log10(self.cosmo.h))
            - (
                2.5
                * np.log10(
                    np.exp(self.zeta * z_value)
                    * (1 + np.exp(self.xi * self.z_star))
                    / denominator
                )
            )
        )

        if np.any(denominator == 0):
            raise ValueError(
                "Encountered zero denominator in M_star calculation. Check input values."
            )

        return result

    def dPhi_dM(self, M, z_value):
        """Calculates dPhi_dM for a given M and redshift according to Eq (10)
        in Oguri & Marshall (2010): DOI: 10.1111/j.1365-2966.2010.16639.x.

        :param M: Absolute i-band magnitude.
        :type M: float or numpy.ndarray
        :param z_value: Redshift value.
        :type z_value: float or numpy.ndarray
        :return: dPhi_dM value in the unit of comoving volume.
        :rtype: float or np.ndarray :unit: mag^-1 Mpc^-3
        """
        M = np.atleast_1d(M)
        z_value = np.atleast_1d(z_value)

        if z_value.shape == ():
            z_value = np.full_like(M, z_value)
        if M.shape == ():
            M = np.full_like(z_value, M)

        alpha_val = np.where(z_value > 3, -2.58, self.alpha)
        M_star_value = self.M_star(z_value)

        denominator_dphi_dm = (10 ** (0.4 * (alpha_val + 1) * (M - M_star_value))) + (
            10 ** (0.4 * (self.beta + 1) * (M - M_star_value))
        )

        # Handle division by zero
        term1 = np.divide(
            self.phi_star,
            denominator_dphi_dm,
            out=np.full_like(denominator_dphi_dm, np.nan),
            where=denominator_dphi_dm != 0,
        )

        return term1

    def convert_magnitude(self, magnitude, z, conversion="apparent_to_absolute"):
        """Converts between apparent and absolute magnitudes using
        K-corrections determined in Table 4 of Richards et al. 2006: DOI:
        10.1086/503559.

        :param magnitude: Apparent or absolute i-band magnitude.
        :type magnitude: float or np.ndarray
        :param z: Redshift.
        :type z: float or np.ndarray
        :param conversion: Conversion direction, either
            'apparent_to_absolute' or 'absolute_to_apparent'.
        :type conversion: str
        :return: Converted magnitude.
        :rtype: float or np.ndarray :unit: mag
        """

        DM = self.cosmo.distmod(z).value
        K_corr = self.k_corr_interp(z)

        if conversion == "apparent_to_absolute":
            converted_magnitude = magnitude - DM - K_corr
        elif conversion == "absolute_to_apparent":
            converted_magnitude = magnitude + DM + K_corr
        else:
            raise ValueError(
                "Conversion must be either 'apparent_to_absolute' or 'absolute_to_apparent'"
            )

        return converted_magnitude

    def n_comoving(self, m_min, m_max, z_value):
        """Calculates the comoving number density of quasars by integrating
        dPhi/dM over the range of absolute magnitudes.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float or np.ndarray
        :param m_max: Maximum apparent magnitude.
        :type m_max: float or np.ndarray
        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: Comoving number density of quasars.
        :rtype: float or np.ndarray :unit: Mpc^-3
        """
        M_min = self.convert_magnitude(
            m_min, z_value, conversion="apparent_to_absolute"
        )
        M_max = self.convert_magnitude(
            m_max, z_value, conversion="apparent_to_absolute"
        )

        if isinstance(z_value, np.ndarray):
            integrals = np.zeros_like(z_value)
            for i, z in enumerate(z_value):
                integral, _ = quad(self.dPhi_dM, M_min[i], M_max[i], args=(z,))
                integrals[i] = integral
            return integrals
        else:
            integral, _ = quad(self.dPhi_dM, M_min, M_max, args=(z_value,))
            return integral

    def generate_quasar_redshifts(self, m_min, m_max):
        """Generates redshift locations of quasars using a light cone
        formulation.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :return: Redshift locations of quasars.
        :rtype: np.ndarray
        """
        n_comoving_values = np.array(
            [self.n_comoving(m_min, m_max, z) for z in self.redshifts]
        )

        sampled_redshifts = redshifts_from_comoving_density(
            redshift=self.redshifts,
            density=n_comoving_values,
            sky_area=self.sky_area,
            cosmology=self.cosmo,
            noise=self.noise,
        )

        # Ensure redshifts are stored as numpy array
        sampled_redshifts = np.array(sampled_redshifts, dtype=float)

        return sampled_redshifts

    def compute_cdf_data(self, m_min, m_max, quasar_redshifts):
        """Computes cumulative distribution function (CDF) data for given
        redshift values.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param quasar_redshifts: Redshift values generated from `generate_quasar_redshifts`.
        :type quasar_redshifts: array-like
        :return: Dictionary containing CDF data for each redshift.
        :rtype: dict
        """
        cdf_data_dict = {}

        for z in np.unique(quasar_redshifts):
            M_min = self.convert_magnitude(m_min, z, conversion="apparent_to_absolute")
            M_max = self.convert_magnitude(m_max, z, conversion="apparent_to_absolute")

            M_values = np.linspace(M_min, M_max, 100)

            dPhi_dM_values = np.array([self.dPhi_dM(M, z) for M in M_values])

            sorted_M_values = np.sort(M_values)
            cumulative_probabilities = np.cumsum(dPhi_dM_values)
            max_cumulative_probabilities = np.max(cumulative_probabilities)
            cumulative_prob_norm = (
                cumulative_probabilities / max_cumulative_probabilities
            )

            cdf_data_dict[z] = (sorted_M_values, cumulative_prob_norm)

        return cdf_data_dict

    def inverse_cdf_fits_for_redshifts(self, m_min, m_max, quasar_redshifts):
        """Creates inverse Cumulative Distribution Function (CDF) fits for each
        redshift.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param quasar_redshifts: Redshift values generated from `generate_quasar_redshifts`.
        :type quasar_redshifts: array-like
        :return: Dictionary containing inverse CDF functions for each redshift.
        :rtype: dict
        """
        cdf_data = self.compute_cdf_data(m_min, m_max, quasar_redshifts)
        inverse_cdf_dict = {}

        for redshift, (sorted_M_values, cumulative_prob_norm) in cdf_data.items():
            inverse_cdf = interp1d(
                cumulative_prob_norm,
                sorted_M_values,
                kind="linear",
                fill_value="extrapolate",
            )
            inverse_cdf_dict[redshift] = inverse_cdf

        return inverse_cdf_dict

    def _calculate_sed_magnitudes(self, z, M_i, target_apparent_i_mag, filters):
        """Helper function to compute magnitudes from SED using speclite.

        We generate the SED using qsogen (with AGILE params). To ensure
        consistency with the sampled apparent magnitude (ps_mag_i), we
        'anchor' the SED calculation to the i-band.

        :param z: Redshift of the quasar
        :param M_i: Absolute magnitude (used in qsosed to set Baldwin
            effect slopes etc)
        :param target_apparent_i_mag: The apparent i-band magnitude
            sampled by this class
        :param filters: loaded speclite filters object
        :return: dictionary of {filter_name: magnitude}
        """
        if self.use_sed_interpolator:
            # Interpolate to get raw magnitudes
            raw_mags_array = self._sed_interpolator((z, M_i))
            raw_mags = {
                band: raw_mags_array[i] for i, band in enumerate(self.qsogen_bands)
            }

            # Determine the anchor band (i-band) to normalize the flux
            raw_anchor_mag = raw_mags["lsst2023-i"]

            # Calculate the normalization offset
            mag_offset = target_apparent_i_mag - raw_anchor_mag

            # Apply offset to all bands
            final_mags = {band: mag + mag_offset for band, mag in raw_mags.items()}

            return final_mags

        # Define a broader wavelength coverage to handle high-z shifting.
        # Default qsosed is logspace(2.95, ...).
        # start at 2.35 (~223 Angstroms) to ensure the blue end of the
        # u-band (approx 3000A) is covered even at z ~ 12.
        # 3000 / (1 + 12) ~ 230 A.
        wavlen = np.logspace(2.35, 4.48, num=25000, endpoint=True)

        # 1. Generate the SED using AGILE parameters
        # params_agile controls the physics (slopes, breaks, etc.)
        # z and M_i control the specific instance (redshifting, emission line strengths)
        quasar = Quasar_sed(
            z=z, M_i=M_i, params=params_agile, wavlen=wavlen, cosmo=self.cosmo
        )

        # 2. Extract flux and wavelength in the observed frame
        # qsosed.py calculates wavred = (1+z)*wavlen
        flux_sed = quasar.flux  # f_lambda in arbitrary units
        wave_sed = quasar.wavred  # Observed wavelength in Angstroms

        # 3. Calculate raw magnitudes from the unscaled SED
        # This returns an astropy table of magnitudes corresponding to self.qsogen_bands
        raw_mags = filters.get_ab_magnitudes(flux_sed, wave_sed)
        # convert to dict
        raw_mags = dict(zip(raw_mags.colnames, raw_mags[0]))

        # 4. Determine the anchor band (i-band) to normalize the flux
        raw_anchor_mag = raw_mags["lsst2023-i"]

        # 5. Calculate the normalization offset
        # true_mag = raw_mag + offset
        # offset = true_mag - raw_mag
        mag_offset = target_apparent_i_mag - raw_anchor_mag

        # 6. Apply offset to all bands
        final_mags = {band: mag + mag_offset for band, mag in raw_mags.items()}

        # Return as a dictionary mapping band name to mag
        return final_mags

    def _build_sed_interpolator(self, z_min=0.05, z_max=6.5, Mi_min=-30, Mi_max=-18):
        """Pre-computes a grid of SED magnitudes to create a fast lookup table.

        :param z_min: Minimum redshift for the grid.
        :type z_min: float
        :param z_max: Maximum redshift for the grid.
        :type z_max: float
        :param Mi_min: Minimum absolute magnitude for the grid.
        :type Mi_min: float
        :param Mi_max: Maximum absolute magnitude for the grid.
        :type Mi_max: float Instead of generating a spectrum for every
            source (O(N)), we generate spectra for a grid of (z, M_i)
            and interpolate.
        """
        # Load filters
        filters = speclite.filters.load_filters(*self.qsogen_bands)
        self._sed_band_names = [b.split("-")[1] for b in self.qsogen_bands]

        # Define grid
        # Z grid
        z_grid = np.linspace(z_min, z_max, 1000)
        # M_i grid
        mi_grid = np.linspace(Mi_min, Mi_max, 20)

        grid_mags = np.zeros((len(z_grid), len(mi_grid), len(self.qsogen_bands)))

        # Pre-compute shared wavelength array
        # logspace(2.35, ...) covers ~223 Angstroms to IR
        wavlen = np.logspace(2.35, 4.48, num=25000, endpoint=True)

        # Loop over grid and compute magnitudes
        for i, z in enumerate(z_grid):
            for j, m_i in enumerate(mi_grid):
                # Generate SED
                quasar = Quasar_sed(
                    z=z, M_i=m_i, params=params_agile, wavlen=wavlen, cosmo=self.cosmo
                )
                flux_sed = quasar.flux
                wave_sed = quasar.wavred

                # Get magnitudes
                mags = filters.get_ab_magnitudes(flux_sed, wave_sed)
                # Extract values in order of qsogen_bands
                for k, band in enumerate(self.qsogen_bands):
                    grid_mags[i, j, k] = mags[band][0]

        # 2D Interpolator (z, M_i) -> (mag_band1, mag_band2, ...)
        self._sed_interpolator = RegularGridInterpolator(
            (z_grid, mi_grid), grid_mags, bounds_error=False, fill_value=None
        )

    def quasar_sample(self, m_min, m_max, seed=42, host_galaxy=False):
        """Generates random redshift values and associated apparent i-band
        magnitude values for quasar samples.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param host_galaxy: Host galaxy catalog generation flag. If True, the
            host galaxy catalog will be generated and matched with the quasar
            catalog. If False, no host galaxy catalog will be generated.
        :return: astropy Table with redshift and associated apparent i-band magnitude values.
        :rtype: `~astropy.table.Table`
        """
        np.random.seed(seed)
        quasar_redshifts = self.generate_quasar_redshifts(m_min=m_min, m_max=m_max)
        inverse_cdf_dict = self.inverse_cdf_fits_for_redshifts(
            m_min, m_max, quasar_redshifts
        )
        table_data = {"z": [], "M_i": [], "ps_mag_i": []}

        for redshift in quasar_redshifts:
            inverse_cdf = inverse_cdf_dict[redshift]
            random_inverse_cdf_value = np.random.rand()
            random_abs_M_i_value = inverse_cdf(random_inverse_cdf_value)

            # Convert the absolute magnitude back to apparent magnitude
            apparent_i_mag = self.convert_magnitude(
                random_abs_M_i_value, redshift, conversion="absolute_to_apparent"
            )
            table_data["M_i"].append(random_abs_M_i_value)
            table_data["z"].append(redshift)
            table_data["ps_mag_i"].append(apparent_i_mag)

        # --- QSOGEN SED FOR MULTIBAND PHOTOMETRY ---
        if self.use_qsogen_sed:
            # Load filters once
            filters = speclite.filters.load_filters(*self.qsogen_bands)

            # Build interpolator if not already done
            if self.use_sed_interpolator and not hasattr(self, "_sed_interpolator"):
                self._build_sed_interpolator(
                    z_min=np.min(table_data["z"]) - 0.1,
                    z_max=np.max(table_data["z"]) + 0.1,
                    Mi_min=np.min(table_data["M_i"]) - 1.0,
                    Mi_max=np.max(table_data["M_i"]) + 1.0,
                )

            # Prepare storage for magnitude columns
            split_bands = [
                ("ps_mag_" + band.split("-")[1]) for band in self.qsogen_bands
            ]
            qsogen_results = {band: [] for band in split_bands}

            # Iterate through the generated quasars to compute multi-band photometry
            for z, M_i, ps_mag_i in zip(
                table_data["z"], table_data["M_i"], table_data["ps_mag_i"]
            ):
                mags = self._calculate_sed_magnitudes(
                    z=z,
                    M_i=M_i,
                    target_apparent_i_mag=ps_mag_i,
                    filters=filters,
                )

                for band, mag in mags.items():
                    # split the band name and remove the observatory name
                    band_name = band.split("-")[1]
                    qsogen_results["ps_mag_" + band_name].append(mag)

            # Add new columns to the table
            new_colnames = qsogen_results.keys()
            old_colnames = table_data.keys()
            extra_cols = set(new_colnames) - set(old_colnames)
            for col in extra_cols:
                table_data[col] = qsogen_results[col]

        # -----------------------------------------------------

        # Create an Astropy Table from the collected data
        table = Table(table_data)

        # Generate and match the host galaxy catalog if requested
        if host_galaxy is True:
            if self.host_galaxy_candidate is None:
                pipeline = SkyPyPipeline(
                    skypy_config=self.skypy_config,
                    sky_area=self.sky_area,
                    filters=None,
                    cosmo=self.cosmo,
                )
                host_galaxy_catalog = vstack(
                    [pipeline.red_galaxies, pipeline.blue_galaxies],
                    join_type="exact",
                )
            else:
                host_galaxy_catalog = self.host_galaxy_candidate

            # compute "vel_disp" if not present
            if "vel_disp" not in host_galaxy_catalog.colnames:
                self._f_vel_disp = vel_disp_abundance_matching(
                    host_galaxy_catalog,
                    z_max=0.5,
                    sky_area=self.sky_area,
                    cosmo=self.cosmo,
                )
                host_galaxy_catalog["vel_disp"] = self._f_vel_disp(
                    np.log10(host_galaxy_catalog["stellar_mass"])
                )

            matching_catalogs = QuasarHostMatch(
                quasar_catalog=table,
                galaxy_catalog=host_galaxy_catalog,
            )
            matched_table = matching_catalogs.match()

            return matched_table

        return table
