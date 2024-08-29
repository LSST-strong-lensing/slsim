import sncosmo  # Importing sncosmo for supernova cosmology-related functionality
import numpy as np  # Importing numpy for numerical operations

from slsim.lens import Lens  # Importing Lens class from slsim.lens module
from typing import Optional, Union  # Importing Optional and Union for type hinting
from astropy.cosmology import Cosmology  # Importing Cosmology class for cosmological calculations
from slsim.lens import theta_e_when_source_infinity  # Importing a function to calculate Einstein radius
from slsim.Sources.source_pop_base import SourcePopBase  # Importing base class for source population
from slsim.ParamDistributions.los_config import LOSConfig  # Importing line-of-sight configuration class
from slsim.Deflectors.deflectors_base import DeflectorsBase  # Importing base class for deflector population
from slsim.lensed_population_base import LensedPopulationBase  # Importing base class for lensed populations

# Class definition for LensPop, which inherits from LensedPopulationBase
class LensPop(LensedPopulationBase):
    """Class to perform samples of lens population."""

    # Constructor method to initialize LensPop instances
    def __init__(
        self,
        deflector_population: DeflectorsBase,  # Deflector population, instance of DeflectorsBase or its subclass
        source_population: SourcePopBase,  # Source population, instance of SourcePopBase or its subclass
        cosmo: Optional[Cosmology] = None,  # Optional Cosmology object, defaults to None if not provided
        sky_area: Optional[float] = None,  # Optional sky area for simulation, defaults to None if not provided
        lightcurve_time: Optional[np.ndarray] = None,  # Optional array for lightcurve observation times, defaults to None
        sn_type: Optional[str] = None,  # Optional supernova type, defaults to None
        sn_absolute_mag_band: Optional[Union[str, sncosmo.Bandpass]] = None,#Optional band for magnitude normalization, defaults to None. Shorthand for Union[str, sncosmo.Bandpass, None]
        sn_absolute_zpsys: Optional[str] = None,  # Optional zero-point system, either 'AB' or 'Vega', defaults to None
        los_config: Optional[LOSConfig] = None,  # Optional line-of-sight configuration, defaults to None
        sn_modeldir: Optional[str] = None,  # Optional directory path for sncosmo model files, defaults to None
    ):
        """
        Initializes a LensPop instance.

        Args:
            deflector_population (DeflectorsBase): Deflector population as an instance of a DeflectorsBase subclass.
            source_population (SourcePopBase): Source population as an instance of a SourcePopBase subclass.
            cosmo (Optional[Cosmology], optional): AstroPy Cosmology instance. Defaults to None.
            sky_area (Optional[float], optional): Sky area in which to simulate the lens population. Defaults to None.
            lightcurve_time (Optional[np.ndarray], optional): Observation time array for lightcurve in days. Defaults to None.
            sn_type (Optional[str], optional): Type of supernova. Defaults to None.
            sn_absolute_mag_band (Optional[Union[str, sncosmo.Bandpass]], optional): Band used for absolute magnitude normalization. Defaults to None.
            sn_absolute_zpsys (Optional[str], optional): Zero point system (AB or Vega). Defaults to None.
            los_config (Optional[LOSConfig], optional): Line of sight configuration. Defaults to None.
            sn_modeldir (Optional[str], optional): Directory path for sncosmo model files. Defaults to None.
        """

        # Call the constructor of the parent class (LensedPopulationBase) to initialize inherited attributes
        super().__init__(
            sky_area=sky_area,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            sn_type=sn_type,
            sn_absolute_mag_band=sn_absolute_mag_band,
            sn_absolute_zpsys=sn_absolute_zpsys,
            sn_modeldir=sn_modeldir,
        )
        self.cosmo = cosmo  # Assign the cosmology object to an instance attribute
        self._lens_galaxies = deflector_population  # Assign the deflector population to an instance attribute
        self._sources = source_population  # Assign the source population to an instance attribute

        # Calculate and store scaling factors for the source and deflector populations based on the sky area
        self._factor_source = self.sky_area.to_value("deg2") / self._sources.sky_area.to_value("deg2")
        self._factor_deflector = self.sky_area.to_value("deg2") / self._lens_galaxies.sky_area.to_value("deg2")

        # If line of sight configuration is not provided, initialize it with a default LOSConfig instance
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()

    # Method to randomly select a lens based on the deflector and source populations
    def select_lens_at_random(self, **kwargs_lens_cut):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        :return: Lens() instance with parameters of the deflector and lens and source light
        """
        while True:  # Infinite loop to keep trying until a valid lens is found
            source = self._sources.draw_source()  # Draw a random source from the source population
            lens = self._lens_galaxies.draw_deflector()  # Draw a random deflector from the deflector population
            
            # Create a Lens instance with the drawn source and deflector
            gg_lens = Lens(
                deflector_dict=lens,  # Dictionary containing deflector parameters
                source_dict=source,  # Dictionary containing source parameters
                variability_model=self._sources.variability_model,  # Variability model for the source
                kwargs_variability=self._sources.kwargs_variability,  # Additional variability parameters for the source
                sn_type=self.sn_type,  # Supernova type
                sn_absolute_mag_band=self.sn_absolute_mag_band,  # Band used for magnitude normalization
                sn_absolute_zpsys=self.sn_absolute_zpsys,  # Zero-point system
                cosmo=self.cosmo,  # Cosmology object
                source_type=self._sources.source_type,  # Type of the source (e.g., point source, extended source)
                light_profile=self._sources.light_profile,  # Light profile model for the source
                lightcurve_time=self.lightcurve_time,  # Array of lightcurve observation times
                los_config=self.los_config,  # Line-of-sight configuration
                sn_modeldir=self.sn_modeldir,  # Directory path for sncosmo model files
            )
            # Check if the lens configuration is valid according to the provided criteria
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens  # If valid, return the lens instance

    @property
    def deflector_number(self):
        """Calculate the number of potential deflectors scaled by the sky area.

        :return: Number of potential deflectors
        """
        # Calculate the number of deflectors by scaling the base number of deflectors by the area ratio
        return round(self._factor_deflector * self._lens_galaxies.deflector_number())

    @property
    def source_number(self):
        """Calculate the number of potential sources scaled by the sky area.

        :return: Number of potential sources
        """
        # Calculate the number of sources by scaling the base number of sources by the area ratio
        return round(self._factor_source * self._sources.source_number_selected)

    # Method to compute the mean number of sources to be tested within a specified test area
    def get_num_sources_tested_mean(self, testarea):
        """Compute the mean of source galaxies needed to be tested within the test area.

        num_sources_tested_mean / testarea = num_sources / sky_area; 
        testarea is in arcsec^2, sky_area is in deg^2. 1 deg^2 = 12960000 arcsec^2

        :param testarea: Area of the test region in arcsec^2
        :return: Mean number of sources that need to be tested within the test area
        """
        num_sources = self.source_number  # Number of potential sources
        # Calculate mean number of sources tested using a ratio based on the sky area and test area
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self._factor_source * self._sources.sky_area.to_value("deg2")
        )
        return num_sources_tested_mean

    # Method to draw a realization of the distribution for the number of sources tested
    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Draw a realization of the expected distribution (Poisson) around the mean for
        the number of source galaxies tested.

        :param testarea: Area of the test region in arcsec^2
        :param num_sources_tested_mean: Mean number of sources tested
        :return: Randomly sampled number of sources tested
        """
        if num_sources_tested_mean is None:  # If mean number of sources tested is not provided
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)  # Compute it using the test area
        # Generate a random number of sources tested based on a Poisson distribution
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    # Method to draw a full population of lenses based on the deflector and source populations
    def draw_population(self, kwargs_lens_cuts, speed_factor=1):
        """Return full population list of all lenses within the area.

        :param kwargs_lens_cuts: Validity test keywords
        :param speed_factor: Factor by which the number of deflectors is decreased to speed up calculations
        :type kwargs_lens_cuts: dict
        :return: List of Lens instances with parameters of the deflectors and lens and source light
        :rtype: list
        """

        # Initialize an empty list to store the Lens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self.deflector_number

        # Loop to draw a population of galaxy-galaxy lenses within the area
        for _ in range(int(num_lenses / speed_factor)):
            lens = self._lens_galaxies.draw_deflector()  # Draw a random deflector
            test_area = draw_test_area(deflector=lens)  # Calculate the test area around the deflector
            # Calculate the number of sources to be tested within the test area
            num_sources_tested = self.get_num_sources_tested(testarea=test_area * speed_factor)
            
            if num_sources_tested > 0:  # If there are sources to be tested
                n = 0  # Initialize counter for sources tested
                while n < num_sources_tested:  # Loop until the required number of sources are tested
                    source = self._sources.draw_source()  # Draw a random source
                    # Create a Lens instance with the drawn source and deflector
                    gg_lens = Lens(
                        deflector_dict=lens,  # Dictionary containing deflector parameters
                        source_dict=source,  # Dictionary containing source parameters
                        variability_model=self._sources.variability_model,  # Variability model for the source
                        kwargs_variability=self._sources.kwargs_variability,  # Additional variability parameters for the source
                        sn_type=self.sn_type,  # Supernova type
                        sn_absolute_mag_band=self.sn_absolute_mag_band,  # Band used for magnitude normalization
                        sn_absolute_zpsys=self.sn_absolute_zpsys,  # Zero-point system
                        cosmo=self.cosmo,  # Cosmology object
                        test_area=test_area,  # Test area for the lens
                        source_type=self._sources.source_type,  # Type of the source (e.g., point source, extended source)
                        los_config=self.los_config,  # Line-of-sight configuration
                        light_profile=self._sources.light_profile,  # Light profile model for the source
                        lightcurve_time=self.lightcurve_time,  # Array of lightcurve observation times
                        sn_modeldir=self.sn_modeldir,  # Directory path for sncosmo model files
                    )
                    # Check if the lens configuration is valid
                    if gg_lens.validity_test(**kwargs_lens_cuts):
                        gg_lens_population.append(gg_lens)  # If valid, add to the population list
                        n = num_sources_tested  # Exit the loop by setting n to the total number of sources tested
                    else:
                        n += 1  # Increment the counter if the lens is not valid
        return gg_lens_population  # Return the list of valid Lens instances

# Function to calculate the test area around a deflector
def draw_test_area(deflector):
    """Draw a test area around the deflector.

    :param deflector: Deflector dictionary
    :return: Test area in arcsec^2
    """
    theta_e_infinity = theta_e_when_source_infinity(deflector)  # Calculate Einstein radius assuming source at infinity
    test_area = np.pi * (theta_e_infinity * 2.5) ** 2  # Calculate the area around the deflector
    return test_area
