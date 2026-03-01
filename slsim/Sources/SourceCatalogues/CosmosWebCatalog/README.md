# Overview

SLSim allows for generating populations of strong lenses using realistic source galaxies instead of simple Sersics. This is done by starting with a set of Sersic parameters and matching them to a source galaxy within a catalog. Here, we focus on the COSMOS Web dataset, which can be downloaded [here](https://cosmos2025.iap.fr/catalog.html). Specifically, the master catalog `COSMOSWeb_mastercatalog_v1.fits` and the `detection_images` should be downloaded.

# Processing the Catalog

Rather than working with the full catalog of sources, we first perform several cuts on the sources to ensure that we only select those galaxies with the highest SNR. These cuts are:

- redshift < 1
- magnitude < 20 in at least one band
- warn_flag = 0 (to ensure high quality sources; see documentation about quality flags [here](https://cosmos2025.iap.fr/catalog.html#quality-flags))
- redshift != -99 (some galaxies have their redshifts in the catalog set to -99; the documentation does not explain why)
- type = 0 (to select source galaxies only; type = 1 is starts and type = 2 is QSOs)

Additionally, a source exclusion list is applied to further filter out sources with contaminants within 75 pixels as well as sources containing nans within a 100x100 pixel cutout. Currently, the five cuts outlined above are hardcoded and not flexible for change, since the source exclusion list would need to be modified to include additional sources that would not have been filtered out by the initial cuts. Such an exclusion list can become several GB in size. Alternatively, the filtering of nans and contaminants would have to be done on-the-fly, which can take several minutes.

The final output is the same `astropy` table that comes with the catalog, but with columns renamed for clarity and extraneous columns removed.

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

A notebook demonstrating how to match Sersics to real sources is given [here](https://github.com/LSST-strong-lensing/slsim/blob/main/notebooks/sersic_to_real_galaxy_source_matching.ipynb). In order to use the catalog, a path to the directory containing both the master catalog and the detection images must be given.