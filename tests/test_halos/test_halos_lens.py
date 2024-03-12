from slsim.Halos.halos_lens import (
    concentration_from_mass,
    deg2_to_cone_angle,
    cone_radius_angle_to_physical_area,
)
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from slsim.Halos.halos_lens import HalosLens
from astropy.table import Table
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import matplotlib


def test_deg2_to_cone_angle():
    solid_angle_deg2 = 20626.4806247  # 2pi steradians in deg^2
    result = deg2_to_cone_angle(solid_angle_deg2)
    #  half cone angle of 2pi sky is pi/2
    assert result == pytest.approx(np.pi / 2, rel=1e-6)


def test_cone_radius_angle_to_physical_area():
    radius_rad = 0.1
    z1 = 0.000001
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    result = cone_radius_angle_to_physical_area(radius_rad, z1, cosmo)
    result_val = result.to_value("Mpc2")
    assert result_val < 0.001

    radius_rad2 = 0.2
    result = cone_radius_angle_to_physical_area(radius_rad2, z1, cosmo)
    result_val2 = result.to_value("Mpc2")
    assert result_val2 / result_val == pytest.approx(4, rel=1e-6)


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
        [z_correction, kappa_ext_correction], names=("z", "kappa")
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


@pytest.fixture
def setup_no_halos():
    z = [np.nan]

    mass = [np.nan]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
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


@pytest.fixture
def setup_mass_sheet_false():
    z = [0.5, 0.6]

    mass = [2058751081954.2866, 1320146153348.6448]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosLens(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        samples_number=1,
        mass_sheet=False,
    )


@pytest.fixture(autouse=True)
def set_matplotlib_backend():
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    yield
    matplotlib.use(original_backend)
    # for test_plot_convergence not to show plot


@pytest.fixture
def setup_halos_lens_large_sample_number():
    z = [0.5, 0.7]

    mass = [2058751081954.2866, 850000000000.0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosLens(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        samples_number=2500,
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


def test_get_lens_model_mass_sheet_false(setup_mass_sheet_false):
    hl2 = setup_mass_sheet_false
    lens_model2 = hl2.get_lens_model()
    assert lens_model2.lens_model_list == ["NFW", "NFW"]


def test_get_halos_lens_kwargs(setup_halos_lens, setup_mass_sheet_false):
    hl = setup_halos_lens
    kwargs_lens = hl.get_halos_lens_kwargs()
    assert len(kwargs_lens) == 4

    hl2 = setup_mass_sheet_false
    kwargs_lens2 = hl2.get_halos_lens_kwargs()
    assert len(kwargs_lens2) == 2


def test_get_nfw_kwargs(setup_halos_lens):
    hl = setup_halos_lens
    Rs_angle, alpha_Rs = hl.get_nfw_kwargs()
    assert len(Rs_angle) == 3
    assert len(alpha_Rs) == 3


def test_get_convergence_shear(setup_halos_lens, setup_no_halos):
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

    kappa, gamma = setup_no_halos.get_convergence_shear()
    assert kappa == 0
    assert gamma == 0

    kappa, gamma1, gamma2 = setup_no_halos.get_convergence_shear(gamma12=True)
    assert kappa == 0
    assert gamma1 == 0
    assert gamma2 == 0


def test_compute_kappa_gamma(setup_halos_lens):
    hl = setup_halos_lens
    i = 1
    gamma_tot = False
    diff = 0.0001
    diff_method = "square"
    results = hl.compute_kappa_gamma(i, gamma_tot, diff, diff_method)
    assert len(results) == 3
    for res in results:
        assert isinstance(res, float)

    gamma_tot = True
    results = hl.compute_kappa_gamma(i, gamma_tot, diff, diff_method)
    assert len(results) == 2
    for res in results:
        assert isinstance(res, float)


def test_get_kappa_gamma_distib(setup_halos_lens, setup_halos_lens_large_sample_number):
    hl = setup_halos_lens
    results = hl.get_kappa_gamma_distib()
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    results = hl.get_kappa_gamma_distib(gamma_tot=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma

    results = hl.get_kappa_gamma_distib(gamma_tot=True, listmean=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma

    results = hl.get_kappa_gamma_distib(gamma_tot=False, listmean=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    hl2 = setup_halos_lens_large_sample_number
    results2 = hl2.get_kappa_gamma_distib()
    assert len(results2) == 2500


def test_get_kappa_gamma_distib_without_multiprocessing(
    setup_halos_lens, setup_halos_lens_large_sample_number
):
    hl = setup_halos_lens
    results = hl.get_kappa_gamma_distib_without_multiprocessing()
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    results = hl.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma

    hl2 = setup_halos_lens_large_sample_number
    results2 = hl2.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True)
    assert results2.shape[0] == hl2.samples_number
    assert results2.shape[1] == 2  # kappa, gamma


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


def test_filter_halos_by_condition(setup_halos_lens):
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
    kappa_ext_list = hl.mass_correction_list["kappa"] if hl.mass_sheet else []
    z_mass_correction = hl.mass_correction_list["z"]
    px_halo = hl.halos_list["px"]
    py_halo = hl.halos_list["py"]
    c_200_halos = hl.halos_list["c_200"]

    combined_redshift_list = np.concatenate((z_halo, z_mass_correction))
    lens_cosmo_dict = hl._build_lens_cosmo_dict(combined_redshift_list, 5.0)
    lens_cosmo_list = list(lens_cosmo_dict.values())

    kwargs_lens = hl._build_kwargs_lens(
        n_halos,
        n_mass_correction,
        z_halo,
        mass_halo,
        px_halo,
        py_halo,
        c_200_halos,
        lens_model_list,
        kappa_ext_list,
        lens_cosmo_list,
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


def test_get_kappaext_gammaext_distib_zdzs(
    setup_halos_lens, setup_halos_lens_large_sample_number
):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    kappa_gamma_distribution = hl.get_kappaext_gammaext_distib_zdzs(zd, zs)

    assert kappa_gamma_distribution.shape == (hl.samples_number, 2)

    hl2 = setup_halos_lens_large_sample_number
    kappa_gamma_distribution2 = hl2.get_kappaext_gammaext_distib_zdzs(zd, zs)
    assert kappa_gamma_distribution2.shape == (hl2.samples_number, 2)

    k_g_distribution = hl.get_kappaext_gammaext_distib_zdzs(zd, zs, listmean=True)
    assert k_g_distribution.shape == (hl.samples_number, 2)
    assert k_g_distribution[0][0] == 0


def test_generate_distributions_0to5(setup_halos_lens):
    hl = setup_halos_lens
    distributions = hl.generate_distributions_0to5()

    assert isinstance(distributions, list)
    for entry in distributions:
        assert "zd" in entry and isinstance(entry["zd"], float)
        assert "zs" in entry and isinstance(entry["zs"], float)
        assert "kappa" in entry and isinstance(entry["kappa"], float)
        assert "gamma" in entry and isinstance(entry["gamma"], float)

    distributions2 = hl.generate_distributions_0to5(output_format="vector")
    assert isinstance(distributions2, list)

    pytest.raises(
        ValueError, hl.generate_distributions_0to5, output_format="wrong_format"
    )


def test_xy_convergence(setup_halos_lens):
    hl = setup_halos_lens
    x, y = 0.1, 0.1
    kappa = hl.xy_convergence(x, y)

    zdzs = (0.5, 1.0)
    kappa2 = hl.xy_convergence(x, y, zdzs=zdzs)

    assert isinstance(kappa, float)
    assert isinstance(kappa2, float)


def test_plot_convergence(setup_halos_lens):
    hl = setup_halos_lens
    try:
        hl.plot_convergence()
        hl.compare_plot_convergence()
    except Exception as e:
        pytest.fail(f"plot_convergence failed with an exception: {e}")


def test_total_mass(setup_halos_lens):
    hl = setup_halos_lens
    mass = hl.total_halo_mass()
    assert isinstance(mass, float)
    assert mass == (2058751081954.2866 + 1320146153348.6448 + 850000000000.0)


def test_total_critical_mass(setup_halos_lens):
    hl = setup_halos_lens
    mass1 = hl.total_critical_mass(method="differential_comoving_volume")
    mass2 = hl.total_critical_mass(method="comoving_volume")
    assert isinstance(mass1, float)
    assert isinstance(mass2, float)
    ratio = mass1 / mass2
    assert ratio == pytest.approx(1, rel=0.01)


def test_compute_various_kappa_gamma_values(
    setup_halos_lens, setup_halos_lens_large_sample_number
):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    (
        kappa_od,
        kappa_os,
        gamma_od1,
        gamma_od2,
        gamma_os1,
        gamma_os2,
        kappa_ds,
        gamma_ds1,
        gamma_ds2,
    ) = hl.compute_various_kappa_gamma_values(zd, zs)
    assert isinstance(kappa_od, float)
    assert isinstance(kappa_os, float)
    assert isinstance(gamma_od1, float)
    assert isinstance(gamma_od2, float)
    assert isinstance(gamma_os1, float)
    assert isinstance(gamma_os2, float)
    assert isinstance(kappa_ds, float)
    assert isinstance(gamma_ds1, float)
    assert isinstance(gamma_ds2, float)

    kappa_gamma_distribution, lens_instance = hl.get_alot_distib_(zd, zs)
    assert isinstance(kappa_gamma_distribution, np.ndarray)
    assert isinstance(lens_instance, np.ndarray)

    hl2 = setup_halos_lens_large_sample_number
    kappa_gamma_distribution2, lens_instance2 = hl2.get_alot_distib_(zd, zs)
    assert isinstance(kappa_gamma_distribution2, np.ndarray)
    assert isinstance(lens_instance2, np.ndarray)

    all_kappa_dicts = hl.compute_kappa_in_bins()
    assert isinstance(all_kappa_dicts, list)


def test_compute_kappa(setup_halos_lens, setup_mass_sheet_false):
    hl1 = setup_halos_lens
    hl2 = setup_mass_sheet_false

    kappa_image, kappa_values = hl1.compute_kappa()
    kappa_image2, kappa_values2 = hl2.compute_kappa(enhance_pos=True)

    assert isinstance(kappa_image, np.ndarray)
    assert isinstance(kappa_values, np.ndarray)
    assert isinstance(kappa_image2, np.ndarray)
    assert isinstance(kappa_values2, np.ndarray)


def test_setting():
    z = [0.5, 0.6]

    mass = [2058751081954.2866, 1320146153348.6448]

    halos = Table([z, mass], names=("z", "mass"))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    with pytest.warns(Warning, match=r".*mass_sheet is set to True.*"):
        HalosLens(
            halos_list=halos,
            mass_correction_list=None,
            sky_area=0.0001,
            cosmo=cosmo,
            samples_number=1,
            mass_sheet=True,
        )
    with pytest.warns(Warning, match=r".*default_cosmology*"):
        HalosLens(
            halos_list=halos,
            mass_correction_list=None,
            sky_area=0.0001,
            samples_number=1,
            mass_sheet=False,
        )


def test_sets_halos_coordinates_to_zero():
    z = [0.5, 0.6, 0.7]

    mass = [2058751081954.2866, 1320146153348.6448, 850000000000.0]

    halos = Table([z, mass], names=("z", "mass"))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    halos_lens = HalosLens(halos_list=halos, cosmo=cosmo)
    halos_lens.enhance_halos_pos_to0()
    assert np.all(halos_lens.halos_list["px"] == 0)
    assert np.all(halos_lens.halos_list["py"] == 0)


def test_mass_divide_kcrit():
    z = [0.5, 0.6, 0.7]

    mass = [2058754.2866, 132013348.6448, 8000000000.0]

    halos = Table([z, mass], names=("z", "mass"))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    halos_lens = HalosLens(halos_list=halos, cosmo=cosmo)
    mass_divide_kcrit = halos_lens.mass_divide_kcrit()
    assert len(mass_divide_kcrit) == 3
