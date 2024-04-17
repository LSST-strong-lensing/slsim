# README.md

## Overview

This README file provides an overview of the CSV file located in the `data/SL-Hammocks` folder. This document aims to help users and contributors understand the structure and the specifics of the data contained within.

## File Description

### `pop_salp_a0001_zl001_to_5_wo_sub.csv`

- **Purpose**: This file provides the deflector population of compound lens model. We set "sky_area= 0.001 [deg^2]", "z_min=0.01", and "z_max=0.05" to generate this population.
  
- **Source**: [SL-Hammocks](https://github.com/kta-cosmo/SL-Hammocks) Currently, this is private project


## Column Descriptions

This section details what each column in the CSV file represents. Below is a list of the columns along with a brief description for each.

1. **id**: Original deflector ID before cutting out to create this file (Basically ignore it.)

2. **zl**: redshift of each deflector

3. **m_h**: mass of dark matter halo component in units of Msun/h

4. **m_acc**: mass of subhalo component at the accretion time in units of Msun/h. For host halos, this value becomes 0. Currently, this file does not include subhalos.

5. **e_h**: ellipticily of dark matter halo

6. **p_h**: position angle of dark matter halo

7. **con**: concentration parameter of dark matter halo

8. **m_g**: mass of galaxy(stellar) component in units of Msun/h

9. **e_g**: ellipticily of galaxy

10. **p_g**: position angle of galaxy

11. **tb**: the scale radius appreared in Hernquist profile in units of arcsec. This parameter relates to the commonly used galaxy effective (half-mass) radius by t_b = 0.551*theta_eff.

12. **vel_disp**: the luminosity-weighted averaged value of the line-of-sight
velocity dispersion within the galaxy effective radius

## Usage

- **Loading Data**: An example of how to load this CSV file is slsim:

  ```python
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