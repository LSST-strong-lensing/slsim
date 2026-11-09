import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from slsim.Deflectors import deflector_util
from slsim.Deflectors.deflector import Deflector
from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source
from slsim.Sources.SourcePopulation.galaxies import convert_catalog_to_source

from slsim.Lenses.arc_tracing import (
    _find_midline,
    compute_LW_phi,
    dist_hlr,
    sersic_params,
    solve_max_rad_re,
    trace_finite_source_images,
)

try:
    import jax  # noqa: F401

    use_jax = True
except ImportError:
    use_jax = False


def _is_float(x):
    return isinstance(x, (float, np.floating))


################################################
############## FIXTURES ########################
################################################


@pytest.fixture(scope="module")
def lens_instance():
    """A simple galaxy-galaxy lens built from the test data, same recipe as
    the Lens tests."""
    path = os.path.dirname(__file__)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    blue_one = Table.read(
        os.path.join(path, "../TestData/blue_one_modified.fits"), format="fits"
    )
    blue_one["angular_size"] = blue_one["angular_size"] / 4.84813681109536e-06
    red_one = Table.read(
        os.path.join(path, "../TestData/red_one_modified.fits"), format="fits"
    )
    red_one["angular_size"] = red_one["angular_size"] / 4.84813681109536e-06

    kwargs_source = convert_catalog_to_source(
        blue_one,
        extended_source_type="single_sersic",
        catalog_type=None,
        size_model=None,
        cosmo=cosmo,
        include_all_keywords=False,
    )
    kwargs_light = convert_catalog_to_source(
        red_one,
        extended_source_type="single_sersic",
        catalog_type=None,
        size_model=None,
        cosmo=cosmo,
        include_all_keywords=False,
    )
    z = kwargs_light.pop("z")
    kwargs_mass = deflector_util.light2mass(
        kwargs_light, halo_dict=red_one, mass_type="EPL"
    )

    source = Source(cosmo=cosmo, **kwargs_source)
    deflector = Deflector(z=z, kwargs_mass=kwargs_mass, kwargs_light=kwargs_light)
    return Lens(
        source_class=source,
        deflector_class=deflector,
        lens_equation_solver="lenstronomy_analytical",
        cosmo=cosmo,
        use_jax=use_jax,
    )


@pytest.fixture
def arc():
    """A synthetic quarter-annulus 'arc' made of interior pixel centers (not
    just the boundary), in arcsec, on a 0.1" pixel grid."""
    pix = 0.1
    idx = np.arange(20)
    gx, gy = np.meshgrid((idx + 0.5) * pix, (idx + 0.5) * pix, indexing="ij")
    r = np.hypot(gx, gy)
    keep = (r >= 1.0) & (r <= 1.6)
    return gx[keep], gy[keep]


################################################
############## sersic_params ###################
################################################


def test_sersic_params():
    kwargs_mag = [
        {
            "magnitude": 20.0,
            "R_sersic": 0.5,
            "n_sersic": 1.0,
            "e1": 0.0,
            "e2": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]
    sources = [(["SERSIC_ELLIPSE"], kwargs_mag), (["SERSIC_ELLIPSE"], kwargs_mag)]

    n_sersic, r_eff, amp = sersic_params(sources, mag_zero_point=25)

    for out in (n_sersic, r_eff, amp):
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
    assert np.allclose(n_sersic, 1.0)
    assert np.allclose(r_eff, 0.5)
    assert np.all(amp > 0)


################################################
############## solve_max_rad_re ################
################################################


def test_solve_max_rad_re():
    n_sersic = np.array([1.0, 2.0, 4.0])
    result = solve_max_rad_re(n_sersic, amp=np.array([10.0, 10.0, 10.0]), amp_min=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)

    # at the half light radius the amplitude is amp, so amp == amp_min -> radius of 1
    at_hlr = solve_max_rad_re(n_sersic, amp=np.ones(3), amp_min=1)
    assert np.allclose(at_hlr, 1.0)

    # a source fainter than the cutoff everywhere never gets clipped below zero
    faint = solve_max_rad_re(np.array([1.0]), amp=np.array([1e-10]), amp_min=1)
    assert np.all(faint >= 0)


################################################
############## dist_hlr ########################
################################################


def test_dist_hlr():
    source = SimpleNamespace(
        ellipticity=(0.0, 0.0),
        extended_source_position=(0.0, 0.0),
        angular_size=0.5,
    )

    # scalar-ish input
    d = dist_hlr(source, np.array([0.5]), np.array([0.0]))
    assert isinstance(d, np.ndarray)
    assert d.shape == (1,)
    assert np.allclose(d, 1.0)  # one half light radius away

    # array input keeps its shape
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    y = np.zeros((3, 4))
    d = dist_hlr(source, x, y)
    assert d.shape == (3, 4)
    assert np.all(d >= 0)


################################################
############## compute_LW_phi ##################
################################################


def test_find_midline(arc):
    x, y, dist_edge = _find_midline(arc, pix_width=0.1)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(dist_edge, np.ndarray)
    assert len(x) == len(y) == len(dist_edge)
    assert len(x) > 0
    assert np.all(dist_edge > 0)


def test_compute_LW_phi(arc):
    length, width, phi = compute_LW_phi(arc, pix_width=0.1)
    assert _is_float(length)
    assert _is_float(width)
    assert _is_float(phi)
    assert np.isfinite([length, width, phi]).all()
    assert length > 0
    assert width > 0
    assert 0 <= phi <= 1 + 1e-9


################################################
############## trace_finite_source_images ######
################################################


def _check_traced_structure(result, lens):
    """The output is a list (one entry per source) of lists of (X, Y)
    arrays."""
    assert isinstance(result, list)
    assert len(result) == len(lens._source)
    for source_images in result:
        assert isinstance(source_images, list)
        for image in source_images:
            assert len(image) == 2
            X, Y = image
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)
            assert X.shape == Y.shape
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(Y))


@pytest.mark.parametrize("return_interior", [False, True])
def test_trace_finite_source_images(lens_instance, return_interior):
    result = trace_finite_source_images(
        lens_instance,
        search_window=12,
        mag_band="i",  # the test data has no Roman F158 magnitudes
        return_interior=return_interior,
    )
    _check_traced_structure(result, lens_instance)


def test_trace_finite_source_images_cut(lens_instance):
    result = trace_finite_source_images(
        lens_instance,
        search_window=12,
        mag_band="i",
        cut=True,
    )
    _check_traced_structure(result, lens_instance)


def test_trace_finite_source_images_nothing_found(lens_instance):
    """A tiny search window far from the lens finds no images, but should
    still return one (empty) list per source rather than erroring."""
    result = trace_finite_source_images(
        lens_instance,
        search_window=2.4,
        x_center=50,
        y_center=50,
        mag_band="i",
    )
    _check_traced_structure(result, lens_instance)
    assert all(len(images) == 0 for images in result)


if __name__ == "__main__":
    pytest.main()