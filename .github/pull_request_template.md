# Pull Request - Compound lens model 

## Overview
We have added codes for the deflector population from the compound lens model in slsim. We have added the funtions for both reading files generated from our code and for generating in slsim.

## Changes Made
Detail the major changes in this pull request.
- Added `compound_lens_halos_galaxies.py` in `slsim/slsim/Deflectors`
  - **Purpose**: Implements a class named `CompoundLensHalosGalaxies`.
- Added `sl_hammocks_pipeline.py` in `slsim/slsim/Pipelines`
  - **Purpose**: Implements a class named `SLHammocksPipeline` and imports `colossus.cosmology` package.
- Added `halo_population.py` in `slsim/slsim/Pipelines`
  - **Purpose**: Function to generate the deflector population for dark matter halos.
- Added `galaxy_population.py` in `slsim/slsim/Pipelines`
  - **Purpose**: Function to generate the deflector population for galaxies; imports `colossus.halo`.
- Added `pop_salp_a1_zl001_to_5_wo_sub.csv` in `slsim/data/SL-Hammocks`
  - **Purpose**: CSV file containing deflector population data generated from the SL-Hammocks code.
- Modified `lens_pop.py`
  - **Changes**: Added an input parameter named `slhammocks_config`. Added a new deflector type, "halo-models".


## Testing Done

Describe how the new and modified code has been tested.
- Tests for `CompoundLensHalosGalaxies` class
- Tests for `SLHammocksPipeline` functionality.
- Verification of the deflector population generation for both halos and galaxies.
By executing the following code:

```Python
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.lens_pop import LensPop

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
sky_area = Quantity(value=0.001, unit="deg2")
kwargs_deflector_cut ={"z_min": 0.01, "z_max": 2.0}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

skypy_config='/your-path-to-slsim/data/SkyPy/lsst-like.yml'
slhammocks_config='/your-path-to-slsim/data/SL-Hammocks/pop_salp_a0001_zl001_to_5_wo_sub.csv'
gg_lens_pop = LensPop(
    deflector_type="halo-models",
    source_type="galaxies",
    kwargs_deflector_cut=kwargs_deflector_cut,
    kwargs_source_cut=kwargs_source_cut,
    kwargs_mass2light=None,
    skypy_config=skypy_config,
    slhammocks_config=slhammocks_config,
    sky_area=sky_area,
    cosmo=cosmo,
)
```
and
```Python
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.lens_pop import LensPop

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
sky_area = Quantity(value=0.001, unit="deg2")
kwargs_deflector_cut ={"z_min": 0.01, "z_max": 2.0}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

skypy_config='/your-path-to-slsim/data/SkyPy/lsst-like.yml'
slhammocks_config=None
gg_lens_pop = LensPop(
    deflector_type="halo-models",
    source_type="galaxies",
    kwargs_deflector_cut=kwargs_deflector_cut,
    kwargs_source_cut=kwargs_source_cut,
    kwargs_mass2light=None,
    skypy_config=skypy_config,
    slhammocks_config=slhammocks_config,
    sky_area=sky_area,
    cosmo=cosmo,
)
```

## Additional Notes
This codes require to install [colossus]("https://bdiemer.bitbucket.io/colossus/") packages.
