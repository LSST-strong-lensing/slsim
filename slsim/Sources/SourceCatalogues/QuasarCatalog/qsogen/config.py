#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration file with parameters for qsosed.py Parameters derived from
SDSS DR16Q x UKIDSS LAS x unWISE as described in Temple, Hewett & Banerji
(2021).

Idea is to load emission line templates, host galaxy template, and extinction
curve once to minimise i/o.

@author: Matthew Temple
Edit 2021 July: v20210625 emline_templates and associated model params
Edit 2022 May: update reference to published paper 2021MNRAS.508..737T
"""

import os
import numpy as np
from pathlib import Path

base_path = Path(os.path.dirname(__file__))

f1 = base_path / "qsosed_emlines_20210625.dat"
emline_template = np.genfromtxt(f1, unpack=True)
# wav, median_emlines, continuum, peaky_line, windy_lines, narrow_lines

f2 = base_path / "S0_template_norm.sed"
galaxy_template = np.genfromtxt(f2, unpack=True)
# S0 galaxy template from SWIRE
# https://ui.adsabs.harvard.edu/abs/2008MNRAS.386..697R/

f3 = base_path / "pl_ext_comp_03.sph"
reddening_curve = np.genfromtxt(f3, unpack=True)
# Extinction curve, format: [lambda, E(lambda-V)/E(B-V)]
# Recall flux_reddened(lambda) = flux(lambda)*10^(-A(lambda)/2.5)
# where A(lambda) = E(B-V)*[E(lambda-V)/E(B-V) + R]
# so taking R=3.1, A(lambda) = E(B-V)*[Col#2 + 3.1]

# fit to DR16Q median 2sigma-clipped colours in multi-imag bins
params_temple = {
    # -- Continuum Slopes & Breaks --
    "plslp1": -0.349,  # UV slope
    "plslp2": 0.593,  # Optical slope
    "plstep": -1.0,  # Step for sub-LyAlpha
    "plbrk1": 3880.0,
    "plbrk3": 500.0,
    # -- Blackbody (Hot Dust) --
    "tbb": 1243.0,
    "bbnorm": 3.961,  # Set >0 to add hot dust bump
    # -- Emission Lines --
    "scal_emline": -0.9936,
    "emline_type": None,  # 0=Average, >0=High EW, <0=High Blueshift
    "scal_halpha": 1.0,
    "scal_lya": 1.0,
    "scal_nlr": 1.0,
    "beslope": 0.183,  # Baldwin effect slope (0=off)
    "benorm": -27.0,
    "bcnorm": False,  # Balmer continuum
    # -- Galaxy & Host --
    "gflag": True,  # Include host galaxy
    "fragal": 0.244,  # Fraction of galaxy light at 4000-5000A
    "gplind": 0.684,  # Power law index for galaxy scaling
    # -- Absorption --
    "lyForest": True,
    "lylim": 912.0,
    # -- External Data --
    "emline_template": emline_template,
    "galaxy_template": galaxy_template,
    "reddening_curve": reddening_curve,
    "zlum_lumval": np.array(
        [
            [0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
            [
                -21.76,
                -22.9,
                -24.1,
                -25.4,
                -26.0,
                -26.6,
                -27.1,
                -27.6,
                -27.9,
                -28.1,
                -28.4,
                -28.6,
                -28.9,
            ],
        ]
    ),
    "M_i": None,  # Absolute Magnitude for scaling
}

# These are sourced from Viitanen++ in prep. for the AGILE project.
# This is more accurate for simulating quasar-only SEDs without host galaxy contamination.
params_agile = {
    # -- Continuum Slopes & Breaks (AGILE Fig 5) --
    "plslp1": -0.43,  # Steeper UV slope (pure AGN)
    "plslp2": -0.05,  # Flattened Optical slope (removed red host galaxy)
    "plstep": -1.0,  # Default (Not fit for in AGILE)
    "plbrk1": 3900.97,  # Break wavelength
    "plbrk3": 500.0,  # Default (Not modified by AGILE)
    # -- Blackbody (Hot Dust) (AGILE Fig 5) --
    "tbb": 1391.09,  # Hotter dust temperature
    "bbnorm": 2.94,  # Dust normalization
    # -- Emission Lines (Fixed in AGILE to Temple+21 defaults) --
    "scal_emline": -0.9936,
    "emline_type": None,
    "scal_halpha": 1.0,
    "scal_lya": 1.0,
    "scal_nlr": 1.0,
    "beslope": 0.183,
    "benorm": -27.0,
    "bcnorm": False,
    # -- Galaxy & Host (Disabled in AGILE) --
    # Host galaxy is removed to allow separate EGG simulation
    "gflag": False,
    "fragal": 0.0,
    "gplind": 0.0,
    # -- Absorption --
    "lyForest": True,
    "lylim": 912.0,
    # -- External Data --
    "emline_template": emline_template,
    "galaxy_template": galaxy_template,
    "reddening_curve": reddening_curve,
    "zlum_lumval": np.array(
        [
            [0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
            [
                -21.76,
                -22.9,
                -24.1,
                -25.4,
                -26.0,
                -26.6,
                -27.1,
                -27.6,
                -27.9,
                -28.1,
                -28.4,
                -28.6,
                -28.9,
            ],
        ]
    ),
    "M_i": None,
}
