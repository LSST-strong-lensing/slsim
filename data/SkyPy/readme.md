# Galaxy Simulation Configuration File
This YAML configuration file is used to generate simulated galaxy populations, including their redshift, luminosity, morphology, and other properties, using the SkyPy package. The file defines two populations of galaxies: blue galaxies (sources) and red galaxies (lenses), each with their unique parameters.

## Parameters
*  mag_lim: Magnitude limit of the galaxy sample (default: 35)
*  fsky: Sky area in square degrees (default: 0.1 deg^2) 
*  z_range: Redshift range for the galaxies
*  M_star_blue: Linear1D model for the blue galaxy population's characteristic absolute magnitude [Astropy-Linear1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Linear1D.html#)
*  phi_star_blue: Exponential1D model for the blue galaxy population's normalization [Astropy-Exponential1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Exponential1D.html)
*  alpha_blue: Faint-end slope of the blue galaxy population's Schechter luminosity function
*  M_star_red: Linear1D model for the red galaxy population's characteristic absolute magnitude [Astropy-Linear1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Linear1D.html#)
*  phi_star_red: Exponential1D model for the red galaxy population's normalization [Astropy-Exponential1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Exponential1D.html)
*  alpha_red: Faint-end slope of the red galaxy population's Schechter luminosity function [Astropy-default_cosmology-github](https://docs.astropy.org/en/stable/api/astropy.cosmology.default_cosmology.html)
*  cosmology: Cosmological model used in the simulation

## Tables
The file contains two tables, one for each of the blue and red galaxies. Each table computes several properties for the galaxies, such as redshift, absolute magnitude, stellar mass, apparent magnitudes, physical size, angular size, and ellipticity. These properties are computed using various models and functions from the SkyPy package.

### 1. Blue Galaxy (Source Galaxies)
* z: Redshifts computed using the Schechter luminosity function [Skypy_schechter_lf_redshift](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.redshift.schechter_lf_redshift.html?highlight=skypy.galaxies.redshift.schechter_lf_redshift)
* M: Absolute magnitudes computed using the Schechter luminosity function [Skypy_schechter_lf_magnitude](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.luminosity.schechter_lf_magnitude.html?highlight=schechter_lf_magnitude)
* coeff: Dirichlet coefficients for the blue galaxy's spectrum (Spectral coefficients to calculate the rest-frame spectral energy) [Skypy_galaxies_spectrum-dirichlet_coefficients](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.spectrum.dirichlet_coefficients.html?highlight=galaxies.spectrum.dirichlet_coefficients)
* stellar_mass: Stellar masses computed using the kcorrect method
* mag_g, mag_r, mag_i, mag_z, mag_Y: Apparent magnitudes in LSST filters [Speclite_filters information](https://speclite.readthedocs.io/en/latest/filters.html?highlight=lsst%20filters#lsst-filters)
* physical_size: Physical sizes computed using the **late-type** lognormal size model [Skypy_late_type_lognormal_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.late_type_lognormal_size.html)
* angular_size: Angular sizes computed based on physical sizes and redshifts [Skypy_angular_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.angular_size.html)
* ellipticity: Ellipticities computed using the beta_ellipticity in skypy [Skypy_beta_ellipticity](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.beta_ellipticity.html)

### 2. Red Galaxy (Lensing Galaxies)
* z: Redshifts computed using the Schechter luminosity function
* M: Absolute magnitudes computed using the Schechter luminosity function
* coeff: Dirichlet coefficients for the red galaxy population's spectrum
* stellar_mass: Stellar masses computed using the kcorrect method
* mag_g, mag_r, mag_i, mag_z, mag_Y: Apparent magnitudes in LSST filters
* physical_size: Physical sizes computed using the **early-type** lognormal size model [Skypy_early_type_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.early_type_lognormal_size.html)
* angular_size: Angular sizes computed based on physical sizes and redshifts 
* ellipticity: Ellipticities computed using the beta_ellipticity model


# Dark Matter Halo Simulation Configuration File
This YAML configuration file is used to generate a simulated population of dark matter halos, including their redshifts and masses, using the SkyPy package and a custom pipeline.

## Warning: 
### The halos pipeline require specific versions of the following packages:
* skypy halos branch [Skypy-halos-github](https://github.com/skypyproject/skypy/tree/module/halos)
* hmf

## Parameters
*  fsky: Sky area in square degrees (default: 0.0001 deg^2) 
*  z_range: Redshift range for the halos !numpy.linspace 
*  cosmology: Cosmological model used in the simulation
*  wavenumber: Wavenumber range for the power spectrum
*  power_spectrum: Power spectrum model used in the simulation, specifically the Eisenstein & Hu model [SkyPy_power_spectrum_eisenstein_hu](https://skypy.readthedocs.io/en/v0.3/api/skypy.power_spectrum.eisenstein_hu.html#skypy.power_spectrum.eisenstein_hu)
    * A_s: Amplitude of the primordial power spectrum (2.1982e-09)
    * n_s: Scalar spectral index (0.969453)
    * cosmology: !astropy.cosmology.default_cosmology.get []
*  collapse_function: The function that describes the collapse of dark matter halos [SkyPy_halo_mass_ellipsoidal_collapse_function](https://skypy.readthedocs.io/en/module-halos/api/skypy.halos.mass.ellipsoidal_collapse_function.html#skypy.halos.mass.ellipsoidal_collapse_function)

## Tables
The file contains a table for dark matter halos. The table computes several properties for the halos, such as redshift and mass, using various functions from the SkyPy package and the custom simulation pipeline.

### Dark Matter Halos
* z: Redshifts (a array like list) computed using a function from the custom simulation pipeline that generates a redshift distribution for halos from a given comoving density. The function uses the following parameters:
  - `redshift_list`: The range of redshifts for the halos
  - `sky_area` : The area of the sky being simulated
  - `cosmology` : The cosmological model being used
  - `m_min`: The minimum halo mass (in solar masses, M☉) considered in the simulation.
  - `m_max`: The maximum halo mass (in solar masses, M☉) considered in the simulation.
  - `resolution`: 1000
  - `wavenumber`: The wavenumber for the power spectrum
  - `power_spectrum`: The power spectrum model being used
  - `collapse_function`: The function that describes the collapse of dark matter halos
  - `params`: The parameters for the collapse function [0.3, 0.7, 0.3, 1.686]
* mass: Halo masses computed using a function from the custom simulation pipeline that generates a mass function for halos at a given redshift. The function uses the same parameters as the redshift function.
  - `z` : Redshifts where the halo mass is to be calculated. It is derived from the redshifts of the halos calculated in the previous step.
  - `cosmology` : The cosmological model used for the simulation. It defaults to the current default cosmology in Astropy if not provided.
  - `m_min` : The minimum halo mass (in solar masses, M☉) considered in the simulation.
  - `m_max` : The maximum halo mass (in solar masses, M☉) considered in the simulation.
  - `resolution` : The number of pieces used for trapezoidal integration from log(min mass) to log(max mass) when calculating the halo mass.
  - `wavenumber` : The array of wavenumbers (in 1/Mpc) at which the power spectrum is calculated.
  - `power_spectrum` : The power spectrum used in the simulation. Here, it is calculated using the Eisenstein & Hu fitting function.
  - `collapse_function` : The collapse function used to calculate the halo mass function. The ellipsoidal collapse function is used in this case.
  - `params` : The parameters for the collapse function [See details in halos-mass](https://github.com/skypyproject/skypy/blob/module/halos/skypy/halos/mass.py)
