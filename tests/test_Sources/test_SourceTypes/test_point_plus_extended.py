import copy

import numpy as np
from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.Sources.SourceTypes.point_plus_extended_source import PointPlusExtendedSource
import os
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import pytest


@pytest.fixture
def supernovae_lens_instance():

    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    source_dict = {
        "z": 1.5,
        "n_sersic_0": 1,
        "n_sersic_1": 4,
        "angular_size_0": 0.2,
        "angular_size_1": 0.15,
        "e1_0": 0.1,
        "e1_1": 0.002,
        "e2_0": 0.001,
        "e2_1": 0.003,
        "w0": 0.4,
        "w1": 0.6,
        "mag_i": 23,
    }
    deflector_dict = Table.read(
        os.path.join(path, "TestData/deflector_supernovae_new.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        kwargs_sn = {
            "variability_model": "light_curve",
            "kwargs_variability": {"supernovae_lightcurve", "i"},
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 1000),
            "sn_modeldir": None,
        }
        source = Source(
            cosmo=cosmo,
            point_source_type="supernova",
            extended_source_type="double_sersic",
            **kwargs_sn,
            **source_dict,
        )
        deflector_dict_new = copy.deepcopy(source_dict)
        deflector_dict_new.pop("z")
        deflector_dict_new["extended_source_type"] = "double_sersic"
        kwargs_mass = {
            "mass_type": "EPL",
            "theta_E": 1,
            "e1": 0.1,
            "e2": 0.05,
            "gamma_pl": 2.0,
        }
        deflector = Deflector(
            z=deflector_dict["z"],
            center_x=0.1,
            center_y=0,
            kwargs_mass=kwargs_mass,
            kwargs_light=deflector_dict_new,
        )

        supernovae_lens = Lens(
            deflector_class=deflector,
            source_class=source,
            cosmo=cosmo,
        )
        print(supernovae_lens.deflector.deflector_center, "deflector center")
        print(
            supernovae_lens.source(index=0).point_source_position, "supernovae position"
        )
        if supernovae_lens.validity_test():

            break
    return supernovae_lens


def test_kwargs_model(supernovae_lens_instance):
    assert isinstance(
        supernovae_lens_instance._source[0]._source, PointPlusExtendedSource
    )

    ps_type = supernovae_lens_instance._source[0].point_source_type(
        image_positions=False
    )
    assert ps_type == "SOURCE_POSITION"

    kwargs_model, kwargs_param = supernovae_lens_instance.lenstronomy_kwargs(band="i")
    assert kwargs_model["point_source_model_list"] == ["LENSED_POSITION"]


def test_update_microlensing_kwargs_source_morphology(supernovae_lens_instance):
    pp_source = supernovae_lens_instance._source[0]._source
    kwargs_morph = {}
    result = pp_source.update_microlensing_kwargs_source_morphology(kwargs_morph)
    assert isinstance(result, dict)
