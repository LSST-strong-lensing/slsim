import os

from slsim.Util.data_path import data_path


def test_data_path_resolves_packaged_data():
    assert os.path.isdir(data_path())
    for directory in (
        "DC2_data",
        "Filters",
        "SL-Hammocks",
        "SkyPy",
        "glass",
        "redMaPPer",
    ):
        assert os.path.isdir(data_path(directory))


def test_data_path_resolves_file():
    path = data_path("SkyPy", "lsst-like.yml")
    assert os.path.isfile(path)
    assert os.path.basename(path) == "lsst-like.yml"
