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
    print(path, "test path")
    source_dict = Table.read(
        os.path.join(path, "TestData/source_supernovae_new.fits"), format="fits"
    )
    source_dict.rename_column("angular_size0", "angular_size_0")
    source_dict.rename_column("angular_size1", "angular_size_1")
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
        deflector = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
        )
        supernovae_lens = Lens(
            deflector_class=deflector,
            source_class=source,
            cosmo=cosmo,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
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
