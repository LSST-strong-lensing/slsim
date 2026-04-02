# Overview

SLSim allows for generating populations of strong lenses using realistic source galaxies instead of simple Sersics. This is done by starting with a set of Sersic parameters and matching them to a source galaxy within a catalog. Here, we focus on the HST COSMOS_23.5_training_sample dataset, which can be downloaded [here](https://zenodo.org/records/3242143).

# Processing the Catalog

Rather than working with the full catalog of sources, we first perform several cuts on the sources to ensure that we only select those galaxies with the highest SNR. These cuts are:

- redshift < 1
- magnitude < 20
- half light radius > 10 pixels

Additionally, a source exclusion list is applied to further filter out sources with contaminants in them. This source exclusion list was created by Sebastian Wagner-Carena, and the details of its creation are unknown. Currently, the three cuts outlined above are hardcoded and not flexible for change, since the source exclusion list would need to be modified to include additional sources that would not have been filtered out by the initial cuts.

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

A notebook demonstrating how to match Sersics to real sources is given [here](https://github.com/LSST-strong-lensing/slsim/blob/main/notebooks/sersic_to_real_galaxy_source_matching.ipynb). In order to use the catalog, a path to the `COSMOS_23.5_training_sample` directory must be given.

# Acknowledgments

This catalog is described in [Mandelbaum et al. 2018](https://doi.org/10.1093/mnras/sty2420).

```
@article{Mandelbaum2018,
    author = {Mandelbaum, Rachel and Lanusse, François and Leauthaud, Alexie and Armstrong, Robert and Simet, Melanie and Miyatake, Hironao and Meyers, Joshua E and Bosch, James and Murata, Ryoma and Miyazaki, Satoshi and Tanaka, Masayuki},
    title = {Weak lensing shear calibration with simulations of the HSC survey},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {481},
    number = {3},
    pages = {3170-3195},
    year = {2018},
    month = {09},
    abstract = {We present results from a set of simulations designed to constrain the weak lensing shear calibration for the Hyper Suprime-Cam (HSC) survey. These simulations include HSC observing conditions and galaxy images from the Hubble Space Telescope (HST), with fully realistic galaxy morphologies and the impact of nearby galaxies included. We find that the inclusion of nearby galaxies in the images is critical to reproducing the observed distributions of galaxy sizes and magnitudes, due to the non-negligible fraction of unrecognized blends in ground-based data, even with the excellent typical seeing of the HSC survey (0.58 arcsec in the i band). Using these simulations, we detect and remove the impact of selection biases due to the correlation of weights and the quantities used to define the sample (S/N and apparent size) with the lensing shear. We quantify and remove galaxy property-dependent multiplicative and additive shear biases that are intrinsic to our shear estimation method, including an ∼10 per cent-level multiplicative bias due to the impact of nearby galaxies and unrecognized blends. Finally, we check the sensitivity of our shear calibration estimates to other cuts made on the simulated samples, and find that the changes in shear calibration are well within the requirements for HSC weak lensing analysis. Overall, the simulations suggest that the weak lensing multiplicative biases in the first-year HSC shear catalogue are controlled at the 1 per cent level.},
    issn = {0035-8711},
    doi = {10.1093/mnras/sty2420},
    url = {https://doi.org/10.1093/mnras/sty2420},
    eprint = {https://academic.oup.com/mnras/article-pdf/481/3/3170/25823242/sty2420.pdf},
}
```

```
@dataset{mandelbaum_2012_3242143,
  author       = {Mandelbaum, Rachel and
                  Lackner, Claire and
                  Leauthaud, Alexie and
                  Rowe, Barnaby},
  title        = {COSMOS real galaxy dataset},
  month        = jan,
  year         = 2012,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3242143},
  url          = {https://doi.org/10.5281/zenodo.3242143},
}
```

This dataset is based on observations made with the NASA/ESA Hubble Space Telescope obtained from the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5–26555. These observations are associated with program(s) 9822 and 10092.