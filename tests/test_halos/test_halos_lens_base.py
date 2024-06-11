from slsim.Halos.halos_lens_base import (
    concentration_from_mass,
    HalosLensBase,
)
from slsim.Util.param_util import deg2_to_cone_angle
from slsim.Util.astro_util import cone_radius_angle_to_physical_area
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
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
    return HalosLensBase(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        mass_sheet=True,
    )


@pytest.fixture
def setup_no_halos():
    z = [np.nan]

    mass = [0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosLensBase(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        mass_sheet=True,
    )


@pytest.fixture
def setup_no_halos_mass_sheet_false():
    z = [np.nan]

    mass = [0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosLensBase(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        mass_sheet=False,
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
    return HalosLensBase(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
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
    return HalosLensBase(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
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


def test_get_lens_model(
    setup_halos_lens, setup_no_halos, setup_no_halos_mass_sheet_false
):
    hl = setup_halos_lens
    lens_model = hl.get_lens_model()
    assert lens_model.lens_model_list == ["NFW", "NFW", "NFW", "CONVERGENCE"]

    hl2 = setup_no_halos
    lens_model2 = hl2.get_lens_model()
    assert lens_model2.lens_model_list == ["CONVERGENCE"]

    hl3 = setup_no_halos_mass_sheet_false
    lens_model3 = hl3.get_lens_model()
    assert lens_model3.lens_model_list == []


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

    assert ([], []) == hl.get_nfw_kwargs(n_halos=0)


def test_get_convergence_shear(setup_halos_lens, setup_no_halos):
    hl = setup_halos_lens

    # Testing without gamma12
    kappa, gamma = hl.halos_get_convergence_shear()
    assert isinstance(kappa, float)
    assert isinstance(gamma, float)

    # Testing with gamma12
    kappa, gamma1, gamma2 = hl.halos_get_convergence_shear(gamma12=True)
    assert isinstance(kappa, float)
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)

    kappa, gamma = setup_no_halos.halos_get_convergence_shear()
    assert kappa == 0
    assert gamma == 0

    kappa, gamma1, gamma2 = setup_no_halos.halos_get_convergence_shear(gamma12=True)
    assert kappa == 0
    assert gamma1 == 0
    assert gamma2 == 0


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

    pytest.raises(ValueError, hl.filter_halos_by_redshift, 0.5, 0.4)


def test_filter_halos_by_condition(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    halos_od, halos_ds, halos_os = hl._filter_halos_by_condition(zd, zs)
    assert all(h["z"] < zd for h in halos_od)
    assert all(zd <= h["z"] < zs for h in halos_ds)
    assert all(h["z"] < zs for h in halos_os)


def test_filter_mass_correction_by_condition(setup_halos_lens, setup_mass_sheet_false):
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

    hl2 = setup_mass_sheet_false
    assert (None, None, None) == hl2._filter_mass_correction_by_condition(0.5, 1.0)


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

    pytest.raises(ValueError, hl._build_lens_data, halos, mass_correction, 0.5, 0.4)
    pytest.raises(ValueError, hl._build_lens_data, halos, mass_correction, 1.0, 1.1)
    pytest.raises(ValueError, hl._build_lens_data, halos, mass_correction, 0.5, 0.6)


def test_build_lens_model(setup_halos_lens, setup_mass_sheet_false, setup_no_halos):
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

    combined_redshift_list2 = [0.5, 0.6, 0.7, 0.8]
    z_source = 1.0
    n_halos_mistake = len(combined_redshift_list2) + 1

    with pytest.raises(ValueError):
        hl._build_lens_model(combined_redshift_list2, z_source, n_halos_mistake)

    combined_redshift_list3 = []
    z_source = 1.0
    n_halos_none = 0
    lens_model_none, lens_model_list_none = hl._build_lens_model(
        combined_redshift_list3, z_source, n_halos_none
    )
    assert len(lens_model_list_none) == 0

    hl2 = setup_mass_sheet_false
    hl2z = [0.5, 0.6]
    n_halos2 = len(hl2z)
    lens_model2, lens_model_list2 = hl2._build_lens_model(hl2z, z_source, n_halos2)
    assert isinstance(lens_model2, LensModel)
    assert len(lens_model_list2) == len(hl2z)
    assert all(isinstance(model_type, str) for model_type in lens_model_list2)

    hl3 = setup_no_halos
    hl3z = [0.0]
    n_halos3 = len(hl3z)
    lens_model3, lens_model_list3 = hl3._build_lens_model(hl3z, z_source, n_halos3)
    assert isinstance(lens_model3, LensModel)
    assert len(lens_model_list3) == len(hl3z)
    assert all(isinstance(model_type, str) for model_type in lens_model_list3)


def test_build_kwargs_lens(
    setup_halos_lens, setup_no_halos, setup_no_halos_mass_sheet_false
):
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

    hl2 = setup_no_halos
    n_halos2 = len(hl2.halos_list)
    n_mass_correction2 = len(hl2.mass_correction_list)
    z_halo2 = hl2.halos_list["z"]
    mass_halo2 = hl2.halos_list["mass"]
    lens_model_list2 = ["NFW"] * n_halos2
    kappa_ext_list2 = hl2.mass_correction_list["kappa"] if hl2.mass_sheet else []
    z_mass_correction2 = hl2.mass_correction_list["z"]
    px_halo2 = hl2.halos_list["px"]
    py_halo2 = hl2.halos_list["py"]
    c_200_halos2 = hl2.halos_list["c_200"]

    combined_redshift_list2 = np.concatenate((z_halo2, z_mass_correction2))
    lens_cosmo_dict2 = hl2._build_lens_cosmo_dict(combined_redshift_list2, 5.0)
    lens_cosmo_list2 = list(lens_cosmo_dict2.values())

    kwargs_lens2 = hl2._build_kwargs_lens(
        n_halos2,
        n_mass_correction2,
        z_halo2,
        mass_halo2,
        px_halo2,
        py_halo2,
        c_200_halos2,
        lens_model_list2,
        kappa_ext_list2,
        lens_cosmo_list2,
    )

    assert len(kwargs_lens2) == n_halos2
    for kwargs in kwargs_lens2:
        assert isinstance(kwargs, dict)

    hl3 = setup_no_halos_mass_sheet_false
    n_halos3 = 0
    n_mass_correction3 = len(hl3.mass_correction_list)

    empty_result = hl3._build_kwargs_lens(
        n_halos3,
        n_mass_correction3,
        z_halo2,
        mass_halo2,
        px_halo2,
        py_halo2,
        c_200_halos2,
        lens_model_list2,
        kappa_ext_list2,
        lens_cosmo_list2,
    )
    assert empty_result == []


def test_get_lens_data_by_redshift(setup_halos_lens):
    hl = setup_halos_lens
    zd = 0.5
    zs = 1.0
    lens_data = hl.get_lens_data_by_redshift(zd, zs)
    assert "ds" in lens_data
    assert isinstance(lens_data["ds"]["param_lens_model"], LensModel)
    assert isinstance(lens_data["ds"]["param_lens_cosmo"], list)
    assert all(
        isinstance(item, LensCosmo) for item in lens_data["ds"]["param_lens_cosmo"]
    )
    assert isinstance(lens_data["ds"]["kwargs_lens"], list)
    assert all(isinstance(item, dict) for item in lens_data["ds"]["kwargs_lens"])

    assert "od" in lens_data
    assert lens_data["od"]["param_lens_model"] is None
    assert lens_data["od"]["param_lens_cosmo"] is None
    assert lens_data["od"]["kwargs_lens"] is None

    assert "os" in lens_data
    assert isinstance(lens_data["os"]["param_lens_model"], LensModel)
    assert isinstance(lens_data["os"]["param_lens_cosmo"], list)
    assert all(
        isinstance(item, LensCosmo) for item in lens_data["os"]["param_lens_cosmo"]
    )
    assert isinstance(lens_data["os"]["kwargs_lens"], list)
    assert all(isinstance(item, dict) for item in lens_data["os"]["kwargs_lens"])


def test_plot_convergence(setup_halos_lens):
    hl = setup_halos_lens
    try:
        hl.plot_halos_convergence()
        hl.plot_halos_convergence(mass_sheet=True)
        hl.plot_halos_convergence(mass_sheet=False)
        hl.halos_compare_plot_convergence()
    except Exception as e:
        pytest.fail(f"plot_convergence failed with an exception: {e}")


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
    ) = hl.compute_halos_nonlinear_correction_kappa_gamma_values(zd, zs)
    assert isinstance(kappa_od, float)
    assert isinstance(kappa_os, float)
    assert isinstance(gamma_od1, float)
    assert isinstance(gamma_od2, float)
    assert isinstance(gamma_os1, float)
    assert isinstance(gamma_os2, float)
    assert isinstance(kappa_ds, float)
    assert isinstance(gamma_ds1, float)
    assert isinstance(gamma_ds2, float)


def test_compute_kappa(setup_halos_lens, setup_mass_sheet_false):
    hl1 = setup_halos_lens
    hl2 = setup_mass_sheet_false

    kappa_image, kappa_values = hl1.halos_compute_kappa()
    kappa_image2, kappa_values2 = hl2.halos_compute_kappa(enhance_pos=True)
    kappa_image3, kappa_values3 = hl2.halos_compute_kappa(
        enhance_pos=False, mass_sheet_bool=False
    )

    assert isinstance(kappa_image, np.ndarray)
    assert isinstance(kappa_values, np.ndarray)
    assert isinstance(kappa_image2, np.ndarray)
    assert isinstance(kappa_values2, np.ndarray)


def test_enhance_halos_pos_to0(setup_halos_lens):
    hl = setup_halos_lens
    hl.enhance_halos_pos_to0()
    px_halo0 = hl.halos_list["px"]
    py_halo0 = hl.halos_list["py"]
    assert px_halo0[0] == 0
    assert py_halo0[0] == 0


def test_setting():
    z = [0.5, 0.6]

    mass = [2058751081954.2866, 1320146153348.6448]

    halos = Table([z, mass], names=("z", "mass"))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    with pytest.warns(Warning, match=r".*mass_sheet is set to True.*"):
        HalosLensBase(
            halos_list=halos,
            mass_correction_list=None,
            sky_area=0.0001,
            cosmo=cosmo,
            mass_sheet=True,
        )
    with pytest.warns(Warning, match=r".*default_cosmology*"):
        HalosLensBase(
            halos_list=halos,
            mass_correction_list=None,
            sky_area=0.0001,
            mass_sheet=False,
        )
