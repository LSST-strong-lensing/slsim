import numpy as np
import os
import pandas as pd
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline


def load_in_rubin_sim(
    ra,
    dec,
    columns=["filter", "observationStartMJD", "fiveSigmaDepth", "visitExposureTime"],
):
    baseline_file = get_baseline()
    name = os.path.basename(baseline_file).replace(".db", "")
    out_dir = "temp"
    results_db = maf.db.ResultsDb(out_dir=out_dir)

    bundle_list = []
    # The point on the sky we would like to get visits for
    # columns at:  https://rubin-sim.lsst.io/rs_scheduler/output_schema.html
    metric = maf.metrics.PassMetric(cols=columns)  # mag_zero_point
    # Select all the visits. Could be something like "filter='r'", "night < 365", etc
    sql = ""
    slicer = maf.slicers.UserPointsSlicer(ra=ra, dec=dec)
    bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name=name))
    bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
    bg = maf.metricBundles.MetricBundleGroup(
        bd, baseline_file, out_dir=out_dir, results_db=results_db
    )
    bg.run_all()
    return bundle_list


def get_rubin_cadence(
    ra,
    dec,
    columns=[
        "filter",
        "observationStartMJD",
        "fiveSigmaDepth",
        "skyBrightness",
        "visitExposureTime",
    ],
):
    bundle_list = load_in_rubin_sim(ra, dec, columns)
    lsst_cadence = pd.DataFrame(bundle_list[0].metric_values[0])
    lsst_cadence["observationStartMJD"] = lsst_cadence["observationStartMJD"] - np.min(
        lsst_cadence["observationStartMJD"]
    )
    lsst_cadence = lsst_cadence.sort_values("observationStartMJD")
    each_band_obs = lsst_cadence.groupby("filter").agg(list)
    return each_band_obs
