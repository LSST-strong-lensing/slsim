# Overview

SLSim allows for generating populations of strong lenses using realistic source galaxies instead of simple Sersics. This is done by starting with a set of Sersic parameters and matching them to a source galaxy within a catalog. This section makes use of a downselected version of the COSMOS Web dataset. The downselected version `COSMOSWeb_galaxy_catalog.fits` along with the corresponding galaxy images can be downloaded [here](https://zenodo.org/records/19188494).

# The Matching Algorithm

We start with the following set of desired parameters:

- axis ratio
- Sersic angle
- Sersic index (optional)
- angular size (half light radius of the Sersic that is the geometric mean of the major and minor axes, in arcseconds)
- redshift
- physical size (computed from the angular size and redshift using a user-supplied instance of astropy cosmology, kpc)

The parameters being matched are the axis ratio and physical size, with the option of also including Sersic index. By default, Sersic index matching is False because SLSim simply sets all Sersic indices to one when generating populations of sources.

The matching is done by first normalizing each parameter's set of values in the catalog (with the desired value also included) so that the minimum value is zero and the maximum value is one. This normalization is done by a simple shifting and scaling of the data. The matched source is then selected as the nearest point to the desired parameters in 2D space (3D if Sersic index is also being matched).

The final output is the cutout of the source, the scaling factor required to match the desired angular size, the angle of rotation required to match the desired angle, and the ID of the source in the original catalog. These outputs are later used to initialize lenstronomy's INTERPOL class.

# Tutorial

A notebook demonstrating how to match Sersics to real sources is given [here](https://github.com/LSST-strong-lensing/slsim/blob/main/notebooks/sersic_to_real_galaxy_source_matching.ipynb). In order to use the catalog, a path to the directory containing both the catalog and the corresponding images must be provided.