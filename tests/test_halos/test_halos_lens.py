from slsim.Halos.halos_lens import concentration_from_mass
import pytest
from astropy.cosmology import FlatLambdaCDM
from slsim.Halos.halos_lens import HalosLens
from astropy.table import Table
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


def test_single_halo_mass_and_redshift():
    z = 0.5
    mass = 1e12

    result = concentration_from_mass(z, mass)

    assert result == pytest.approx(5.433264, rel=1e-6)


@pytest.fixture
def setup_halos_lens():
    z = [0.5, 0.6, 0.7]

    mass = [2058751081954.2866, 1320146153348.6448, 850000000000.0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa_ext")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosLens(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        samples_number=1,
        mass_sheet=True,
    )


def test_init(setup_halos_lens):
    hl = setup_halos_lens
    assert hl.n_halos == 3
    assert hl.n_correction == 1


def test_random_position(setup_halos_lens):
    hl = setup_halos_lens
    px, py = hl.random_position()
    assert isinstance(px, float)
    assert isinstance(py, float)


def test_get_lens_model(setup_halos_lens):
    hl = setup_halos_lens
    lens_model = hl.get_lens_model()
    assert lens_model.lens_model_list == ["NFW", "NFW", "NFW", "CONVERGENCE"]


def test_get_halos_lens_kwargs(setup_halos_lens):
    hl = setup_halos_lens
    kwargs_lens = hl.get_halos_lens_kwargs()
    assert len(kwargs_lens) == 4


def test_get_convergence_shear(setup_halos_lens):
    hl = setup_halos_lens

    # Testing without gamma12
    kappa, gamma = hl.get_convergence_shear()
    assert isinstance(kappa, float)
    assert isinstance(gamma, float)

    # Testing with gamma12
    kappa, gamma1, gamma2 = hl.get_convergence_shear(gamma12=True)
    assert isinstance(kappa, float)
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)


def test_compute_kappa_gamma(setup_halos_lens):
    hl = setup_halos_lens
    i = 0
    gamma_tot = False
    diff = 0.0001
    diff_method = "square"
    results = hl.compute_kappa_gamma(i, hl, gamma_tot, diff, diff_method)
    assert len(results) == 3
    for res in results:
        assert isinstance(res, float)

    gamma_tot = True
    results = hl.compute_kappa_gamma(i, hl, gamma_tot, diff, diff_method)
    assert len(results) == 2
    for res in results:
        assert isinstance(res, float)


def test_get_kappa_gamma_distib(setup_halos_lens):
    hl = setup_halos_lens
    results = hl.get_kappa_gamma_distib()
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    results = hl.get_kappa_gamma_distib(gamma_tot=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma


def test_get_kappa_gamma_distib_without_multiprocessing(setup_halos_lens):
    hl = setup_halos_lens
    results = hl.get_kappa_gamma_distib_without_multiprocessing()
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    results = hl.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma


def test_filter_halos_by_redshift(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0

    lens_data_ds, lens_data_od, lens_data_os = hl.filter_halos_by_redshift(zd, zs)
    assert lens_data_od == (None, None, None)
    assert len(lens_data_ds) == 3
    assert len(lens_data_os) == 3
    assert isinstance(lens_data_ds[0], LensModel)
    assert isinstance(lens_data_ds[1], list)
    assert all(isinstance(item, LensCosmo) for item in lens_data_ds[1])
    assert isinstance(lens_data_ds[2], list)
    assert all(isinstance(item, dict) for item in lens_data_ds[2])


def test__filter_halos_by_condition(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    halos_od, halos_ds, halos_os = hl._filter_halos_by_condition(zd, zs)
    assert all(h["z"] < zd for h in halos_od)
    assert all(zd <= h["z"] < zs for h in halos_ds)
    assert all(h["z"] < zs for h in halos_os)


def test_filter_mass_correction_by_condition(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    (
        mass_correction_od,
        mass_correction_ds,
        mass_correction_os,
    ) = hl._filter_mass_correction_by_condition(zd, zs)
    if mass_correction_od is not None:
        assert all(h["z"] < zd for h in mass_correction_od)
    if mass_correction_ds is not None:
        assert all(zd <= h["z"] < zs for h in mass_correction_ds)
    if mass_correction_os is not None:
        assert all(h["z"] < zs for h in mass_correction_os)


def test_build_lens_data(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    halos = hl.halos_list
    mass_correction = hl.mass_correction_list
    lens_model, lens_cosmo_list, kwargs_lens = hl._build_lens_data(
        halos, mass_correction, zd, zs
    )

    assert isinstance(lens_model, LensModel)
    assert all(isinstance(item, LensCosmo) for item in lens_cosmo_list)
    assert isinstance(kwargs_lens, list)


def test_build_lens_cosmo_list(setup_halos_lens):
    hl = setup_halos_lens
    combined_redshift_list = [0.5, 0.6, 0.7]
    z_source = 1.0
    lens_cosmo_list = hl._build_lens_cosmo_list(combined_redshift_list, z_source)

    assert len(lens_cosmo_list) == len(combined_redshift_list)
    assert all(isinstance(item, LensCosmo) for item in lens_cosmo_list)


def test_build_lens_model(setup_halos_lens):
    hl = setup_halos_lens
    combined_redshift_list = [0.5, 0.6, 0.7]
    z_source = 1.0
    n_halos = len(combined_redshift_list)

    lens_model, lens_model_list = hl._build_lens_model(
        combined_redshift_list, z_source, n_halos
    )

    assert isinstance(lens_model, LensModel)
    assert len(lens_model_list) == len(combined_redshift_list)
    assert all(isinstance(model_type, str) for model_type in lens_model_list)


def test_build_kwargs_lens(setup_halos_lens):
    hl = setup_halos_lens
    n_halos = len(hl.halos_list)
    n_mass_correction = len(hl.mass_correction_list)
    z_halo = hl.halos_list["z"]
    mass_halo = hl.halos_list["mass"]
    lens_model_list = ["NFW"] * n_halos
    kappa_ext_list = hl.mass_correction_list["kappa_ext"] if hl.mass_sheet else []

    kwargs_lens = hl._build_kwargs_lens(
        n_halos, n_mass_correction, z_halo, mass_halo, lens_model_list, kappa_ext_list
    )

    assert len(kwargs_lens) == n_halos
    for kwargs in kwargs_lens:
        assert isinstance(kwargs, dict)


def test_get_lens_data_by_redshift(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    lens_data = hl.get_lens_data_by_redshift(zd, zs)
    assert "ds" in lens_data
    assert isinstance(lens_data["ds"]["lens_model"], LensModel)
    assert isinstance(lens_data["ds"]["lens_cosmo"], list)
    assert all(isinstance(item, LensCosmo) for item in lens_data["ds"]["lens_cosmo"])
    assert isinstance(lens_data["ds"]["kwargs_lens"], list)
    assert all(isinstance(item, dict) for item in lens_data["ds"]["kwargs_lens"])

    assert "od" in lens_data
    assert lens_data["od"]["lens_model"] is None
    assert lens_data["od"]["lens_cosmo"] is None
    assert lens_data["od"]["kwargs_lens"] is None

    assert "os" in lens_data
    assert isinstance(lens_data["os"]["lens_model"], LensModel)
    assert isinstance(lens_data["os"]["lens_cosmo"], list)
    assert all(isinstance(item, LensCosmo) for item in lens_data["os"]["lens_cosmo"])
    assert isinstance(lens_data["os"]["kwargs_lens"], list)
    assert all(isinstance(item, dict) for item in lens_data["os"]["kwargs_lens"])


def test_get_kext_gext_values(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    kext, gext = hl.get_kext_gext_values(zd, zs)

    assert isinstance(kext, float)
    assert isinstance(gext, float)
    assert -1 <= kext <= 1
    assert 0 <= gext


def test_get_kappaext_gammaext_distib_zdzs(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    kappa_gamma_distribution = hl.get_kappaext_gammaext_distib_zdzs(zd, zs)

    assert kappa_gamma_distribution.shape == (hl.samples_number, 2)


def test_generate_distributions_0to5(setup_halos_lens):
    hl = setup_halos_lens
    distributions = hl.generate_distributions_0to5()

    assert isinstance(distributions, list)
    for entry in distributions:
        assert "zd" in entry and isinstance(entry["zd"], float)
        assert "zs" in entry and isinstance(entry["zs"], float)
        assert "kappa" in entry and isinstance(entry["kappa"], float)
        assert "gamma" in entry and isinstance(entry["gamma"], float)
