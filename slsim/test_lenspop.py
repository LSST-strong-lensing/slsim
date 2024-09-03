import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy.table import Table

import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors
from slsim.lens_pop import LensPop
from slsim.Plots.lens_plots import LensingPlots

def test_lenspop():
    # Seed for reproducibility
    np.random.seed(1)

    # Cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Sky area
    sky_area = Quantity(value=0.05, unit="deg2")

    # Deflector and source cuts
    kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.2, "z_max": 1.0}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.21, "z_max": 5.0}

    # Pipeline and source population
    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=None,
        sky_area=sky_area,
        filters=None,
    )
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )

    # Load cluster and member catalogs
    #cluster_catalog = Table.read("../data/redMaPPer/clusters_example.fits")
    #members_catalog = Table.read("../data/redMaPPer/members_example.fits")
    cluster_catalog = Table.read("C:/Users/pinov/slsim_pruebas/data/redMaPPer/clusters_example.fits")
    members_catalog = Table.read("C:/Users/pinov/slsim_pruebas/data/redMaPPer/members_example.fits")

    # Cluster deflectors
    lens_clusters = deflectors.ClusterDeflectors(
        cluster_list=cluster_catalog,
        members_list=members_catalog,
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    # Initialize LensPop
    lenspop = LensPop(
        deflector_population=lens_clusters,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    # Plotting setup
    kwargs_lens_cut_plot = {
        "min_image_separation": 2.0,
        "max_image_separation": 30.0,
        "mag_arc_limit": {"g": 22, "r": 22, "i": 22},
    }

    cluster_lens_plot = LensingPlots(lenspop, num_pix=200, coadd_years=10)

    # Generate montage plot
    fig, axes = cluster_lens_plot.plot_montage(
        rgb_band_list=["i", "r", "g"],
        add_noise=True,
        n_horizont=5,
        n_vertical=3,
        kwargs_lens_cut=kwargs_lens_cut_plot,
    )
    plt.show()

if __name__ == "__main__":
    test_lenspop()
