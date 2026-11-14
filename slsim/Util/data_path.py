from pathlib import Path


def data_path(*parts):
    """Return the path to a packaged SLSim data file.

    :param parts: path components relative to the ``slsim/data`` directory
    :return: absolute path to the requested data file or directory
    :rtype: str
    """
    return str(Path(__file__).resolve().parents[1] / "data" / Path(*parts))
