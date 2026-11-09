import numpy as np
import numpy.testing as npt
import pytest
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import constants
from lenstronomy.Util.param_util import ellipticity2phi_q

from slsim.Deflectors.deflector import Deflector
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_power_law,
)
from slsim.Deflectors.deflector_kinematics import DeflectorKinematics


def test_sphericalize_kwargs():
    kwargs = [
        {"amp": 1, "e1": 0.1, "e2": -0.2, "center_x": 0.3, "center_y": 0.1},
        {"theta_E": 1, "center_x": 0.3, "center_y": 0.1},
    ]
    kwargs_spherical = DeflectorKinematics.sphericalize_kwargs(kwargs)
    assert kwargs_spherical[0]["e1"] == 0 and kwargs_spherical[0]["e2"] == 0
    assert kwargs_spherical[1]["center_x"] == 0 and kwargs_spherical[1]["center_y"] == 0
    assert "e1" not in kwargs_spherical[1]
    # input is not modified
    assert kwargs[0]["e1"] == 0.1


def test_fit_mge_1d():
    radius = np.geomspace(1e-3, 10, 500)
    light_model = LightModel(["SERSIC"])
    profile = light_model.surface_brightness(
        radius, 0, [{"amp": 1, "R_sersic": 1, "n_sersic": 4}]
    )
    amp, sigma = DeflectorKinematics.fit_mge_1d(radius, profile, ngauss=20)
    profile_mge = np.sum(
        amp[:, None] * np.exp(-(radius[None, :] ** 2) / (2 * sigma[:, None] ** 2)),
        axis=0,
    )
    mask = radius > 0.01
    npt.assert_allclose(profile_mge[mask], profile[mask], rtol=0.02)


def test_mass_mge():
    lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5)
    kwargs_lens = [{"theta_E": 1, "center_x": 0.2, "center_y": 0}]
    surf, sigma, qobs = DeflectorKinematics._mass_mge(
        ["SIS"], kwargs_lens, q_mass=0.5, lens_cosmo=lens_cosmo, r_eff=1, ngauss=15
    )
    assert len(surf) <= 15
    npt.assert_almost_equal(qobs, 0.5)
    # Sigma of an SIS at 1 arcsec is Sigma_crit / 2
    sigma_intermediate = sigma * np.sqrt(0.5)
    sigma_1 = np.sum(surf * np.exp(-1 / (2 * sigma_intermediate**2)))
    npt.assert_allclose(sigma_1, lens_cosmo.sigma_crit / 2 / 1e12, rtol=0.02)


def test_draw_inclination():
    inc = [DeflectorKinematics.draw_inclination(60, rng=seed) for seed in range(100)]
    assert np.min(inc) >= 60 and np.max(inc) <= 90
    npt.assert_almost_equal(
        DeflectorKinematics.draw_inclination(60, rng=1),
        DeflectorKinematics.draw_inclination(60, rng=np.random.default_rng(1)),
    )


def test_kinematic_grid():
    x, y, shape = DeflectorKinematics.kinematic_grid(map_half_size=1.0, grid_size=0.2)
    assert shape == (10, 10)
    assert len(x) == 100
    npt.assert_almost_equal(np.mean(x), 0)
    npt.assert_almost_equal(np.max(y), 0.9)


def test_vrms_aperture():
    x, y, _ = DeflectorKinematics.kinematic_grid(map_half_size=2.0, grid_size=0.1)
    r = np.sqrt(x**2 + y**2)
    flux = np.exp(-r)
    vrms = np.where(r < 1, 200.0, 100.0)
    npt.assert_almost_equal(
        DeflectorKinematics.flux_weighted_vrms(
            vrms, flux, x, y, "circular", {"radius": 0.9}
        ),
        200,
    )
    npt.assert_almost_equal(
        DeflectorKinematics.flux_weighted_vrms(
            vrms, flux, x, y, "annulus", {"r_in": 1.1, "r_out": 1.9}
        ),
        100,
    )
    with pytest.raises(ValueError):
        DeflectorKinematics.flux_weighted_vrms(
            vrms, flux, x, y, "square", {"radius": 1}
        )


def test_vrms_to_J():
    vrms, dds, ds = 250.0, 800.0, 1500.0
    J = DeflectorKinematics.vrms_to_J(vrms, dds, ds)
    # helica definition: sig_e**2 * dds / ds / c**2 * 1e6
    npt.assert_almost_equal(J, vrms**2 * dds / ds / constants.c**2 * 1e6)
    npt.assert_almost_equal(DeflectorKinematics.J_to_vrms(J, dds, ds), vrms)


class TestDeflectorKinematics(object):
    def setup_method(self):
        # deflector of TDLDMC26/sims/kin_parametric_model_sim.ipynb
        self.kwargs_light = {
            "extended_source_type": "single_sersic",
            "mag_i": 21.097706,
            "angular_size": 0.934183,
            "e1": -0.156653,
            "e2": -0.131999,
            "n_sersic": 1,
            "stellar_mass": 2.89782e10,
        }
        self.kwargs_mass = {
            "mass_type": "EPL",
            "e1": -0.371517,
            "e2": -0.134980,
            "gamma_pl": 2,
            "vel_disp": 316.898046,
            "sis_convention": False,
        }
        self.deflector = Deflector(
            z=0.481053,
            center_x=0,
            center_y=0,
            kwargs_light=self.kwargs_light,
            kwargs_mass=self.kwargs_mass,
        )
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_source = 0.788751
        self.lens_cosmo = LensCosmo(
            z_lens=self.deflector.redshift, z_source=self.z_source, cosmo=self.cosmo
        )
        with pytest.warns(UserWarning):
            # light and mass position angles differ by 10.1 deg
            self.kinematics = DeflectorKinematics(
                self.deflector, self.cosmo, self.z_source
            )

    def _deflector(self, kwargs_mass, kwargs_light=None):
        if kwargs_light is None:
            kwargs_light = self.kwargs_light
        return Deflector(
            z=0.481053,
            center_x=0,
            center_y=0,
            kwargs_light=kwargs_light,
            kwargs_mass=kwargs_mass,
        )

    def test_mge(self):
        _, q_light = ellipticity2phi_q(self.kwargs_light["e1"], self.kwargs_light["e2"])
        _, q_mass = ellipticity2phi_q(self.kwargs_mass["e1"], self.kwargs_mass["e2"])
        npt.assert_almost_equal(self.kinematics.q_light, q_light)
        npt.assert_almost_equal(self.kinematics.q_mass, q_mass)
        surf_lum, sigma_lum, qobs_lum = self.kinematics.light_mge()
        surf_pot, sigma_pot, qobs_pot = self.kinematics.mass_mge()
        assert len(surf_lum) <= 20 and len(surf_pot) <= 20
        assert len(self.kinematics.light_mge(ngauss=10)[0]) <= 10
        npt.assert_almost_equal(qobs_lum, q_light)
        npt.assert_almost_equal(qobs_pot, q_mass)
        # minimum inclination from the flattest component, q_mass
        cos_inc = np.sqrt((q_mass**2 - 0.05**2) / (1 - 0.05**2))
        npt.assert_almost_equal(
            self.kinematics.inclination_min(), np.degrees(np.arccos(cos_inc))
        )
        cos_inc = np.sqrt((q_mass**2 - 0.3**2) / (1 - 0.3**2))
        npt.assert_almost_equal(
            self.kinematics.inclination_min(q_intr_min=0.3),
            np.degrees(np.arccos(cos_inc)),
        )
        # q_intr_min below jampy's minimum, or not below q_obs
        with pytest.raises(ValueError):
            self.kinematics.inclination_min(q_intr_min=0.01)
        with pytest.raises(ValueError):
            self.kinematics.inclination_min(q_intr_min=0.5)

    def test_vrms_map(self, capsys):
        kwargs_bins = {
            "map_half_size": 0.5,
            "grid_size": 0.25,
            "inc": 80,
            "pix_size": 0.25,
        }
        sigma_psf = 0.5 / (2 * np.sqrt(2 * np.log(2)))
        vrms_map = self.kinematics.vrms_map(sigma_psf=sigma_psf, **kwargs_bins)
        xbin, ybin, _ = DeflectorKinematics.kinematic_grid(0.5, 0.25)
        assert len(vrms_map["model"]) == 16
        assert np.all(vrms_map["model"] > 0)
        npt.assert_allclose(vrms_map["xbin"], xbin)
        npt.assert_allclose(vrms_map["ybin"], ybin)
        # a single Gaussian PSF with norm_psf=1 is jam_axi_proj's default
        vrms_map_psf = self.kinematics.vrms_map(
            sigma_psf=[sigma_psf], norm_psf=[1.0], **kwargs_bins
        )
        npt.assert_allclose(vrms_map_psf["model"], vrms_map["model"])
        # other keyword arguments go to jam_axi_proj
        vrms_map_nang = self.kinematics.vrms_map(
            sigma_psf=sigma_psf, nang=10, **kwargs_bins
        )
        npt.assert_allclose(vrms_map_nang["model"], vrms_map["model"], rtol=1e-2)
        # MGE settings go to light_mge and mass_mge
        vrms_map_mge = self.kinematics.vrms_map(
            sigma_psf=sigma_psf,
            kwargs_mge_light={"ngauss": 15},
            kwargs_mge_mass={"ngauss": 15},
            **kwargs_bins,
        )
        npt.assert_allclose(vrms_map_mge["model"], vrms_map["model"], rtol=1e-2)
        # inc not given: drawn at random within the allowed range and printed
        vrms_map_random = self.kinematics.vrms_map(map_half_size=0.5, grid_size=0.25)
        assert np.all(vrms_map_random["model"] > 0)
        out = capsys.readouterr().out
        assert "randomly drawn" in out
        inc = float(out.split("inc = ")[1].split(" deg")[0])
        assert self.kinematics.inclination_min() < inc <= 90
        # arguments set by the class
        with pytest.raises(ValueError):
            self.kinematics.vrms_map(moment="xx", **kwargs_bins)

    def test_vrms_aperture(self):
        kwargs_map = {
            "map_half_size": 2.8,
            "grid_size": 0.2,
            "pix_size": 0.2,
            "inc": 80,
        }
        vrms_aperture = self.kinematics.vrms_aperture(sigma_psf=0.3, **kwargs_map)
        vrms_map = self.kinematics.vrms_map(sigma_psf=0.3, **kwargs_map)
        npt.assert_almost_equal(
            vrms_aperture,
            DeflectorKinematics.flux_weighted_vrms(
                vrms_map["model"],
                vrms_map["flux"],
                vrms_map["xbin"],
                vrms_map["ybin"],
                kwargs_aperture={"radius": self.kwargs_light["angular_size"]},
            ),
        )

    def test_cosmology_independence(self):
        # mass normalized by vel_disp: the physical V_rms does not depend on cosmology
        # or source redshift
        with pytest.warns(UserWarning):
            kinematics = DeflectorKinematics(
                self.deflector, FlatLambdaCDM(H0=60, Om0=0.4), 2.0
            )
        kwargs_map = {
            "map_half_size": 2.8,
            "grid_size": 0.4,
            "pix_size": 0.4,
            "inc": 80,
        }
        vrms = self.kinematics.vrms_map(**kwargs_map)["model"]
        vrms_other = kinematics.vrms_map(**kwargs_map)["model"]
        npt.assert_allclose(vrms_other, vrms, rtol=2e-3)

    def test_galkin(self):
        # spherical isotropic limit against slsim's GalKin velocity dispersion
        kwargs_light = dict(self.kwargs_light, e1=0, e2=0)
        kwargs_mass = dict(self.kwargs_mass, e1=0, e2=0)
        deflector = self._deflector(kwargs_mass, kwargs_light)
        kinematics = DeflectorKinematics(deflector, self.cosmo, self.z_source)
        r_eff = self.kwargs_light["angular_size"]
        vrms = kinematics.vrms_aperture(
            map_half_size=r_eff, grid_size=0.05, pix_size=0.05, inc=90
        )
        light_model_list, kwargs_light = deflector.light_model_lenstronomy()
        theta_E = self.lens_cosmo.sis_sigma_v2theta_E(self.kwargs_mass["vel_disp"])
        vel_disp_galkin = vel_disp_power_law(
            theta_E, 2, r_eff, kwargs_light, light_model_list, self.lens_cosmo
        )
        npt.assert_allclose(vrms, vel_disp_galkin, rtol=0.03)

    def test_inclination(self):
        kwargs_bins = {"map_half_size": 0.5, "grid_size": 0.25}
        inc_min = self.kinematics.inclination_min()
        # a given inclination is used, up to 90 deg
        vrms = [
            self.kinematics.vrms_map(inc=inc, **kwargs_bins)["model"]
            for inc in [inc_min + 1e-6, 75, 90]
        ]
        assert not np.allclose(vrms[1], vrms[2])
        # outside the allowed range, including the lower limit
        for inc in [inc_min - 1, inc_min, 91]:
            with pytest.raises(ValueError):
                self.kinematics.vrms_map(inc=inc, **kwargs_bins)
        # random inclinations come from draw_inclination
        for seed in range(5):
            inc = DeflectorKinematics.draw_inclination(
                self.kinematics.inclination_min(q_intr_min=0.3), rng=seed
            )
            assert self.kinematics.inclination_min(q_intr_min=0.3) < inc <= 90
            self.kinematics.vrms_map(inc=inc, **kwargs_bins)
        with pytest.raises(ValueError):
            self.kinematics.inclination_min(q_intr_min=0.01)

    def test_round(self):
        kwargs_light = dict(self.kwargs_light, e1=0, e2=0)
        kwargs_mass = dict(self.kwargs_mass, e1=0, e2=0)
        kinematics = DeflectorKinematics(
            self._deflector(kwargs_mass, kwargs_light), self.cosmo, self.z_source
        )
        # every inclination except face-on is allowed for a round deflector
        npt.assert_almost_equal(kinematics.inclination_min(), 0)
        vrms_map = kinematics.vrms_map(map_half_size=0.5, grid_size=0.25, inc=1)
        assert np.all(vrms_map["model"] > 0)
        with pytest.raises(ValueError):
            kinematics.vrms_map(map_half_size=0.5, grid_size=0.25, inc=0)

    def test_mass_types(self):
        kwargs_mass_list = [
            {"mass_type": "EPL", "vel_disp": 250, "gamma_pl": 2.1, "e1": 0.1},
            {"mass_type": "NFW", "halo_mass": 1e13, "concentration": 8, "e1": 0.1},
            {"mass_type": "HERNQUIST"},
            {
                "mass_type": "NFW_HERNQUIST",
                "halo_mass": 1e13,
                "concentration": 8,
                "e1": 0.1,
            },
            {
                "mass_type": "PJAFFE",
                "r_s": 2,
                "r_a": 0.1,
                "vel_disp": 200,
                "e1": 0.1,
                "e2": 0,
            },
        ]
        kwargs_light = dict(self.kwargs_light, e1=0.1, e2=0)
        kwargs_map = {
            "map_half_size": 2.8,
            "grid_size": 0.5,
            "pix_size": 0.5,
            "inc": 80,
        }
        for kwargs_mass in kwargs_mass_list:
            deflector = self._deflector(kwargs_mass, kwargs_light)
            kinematics = DeflectorKinematics(deflector, self.cosmo, self.z_source)
            _, q_mass = ellipticity2phi_q(
                *deflector.composite_mass_ellipticity(self.lens_cosmo)
            )
            mge_pot = kinematics.mass_mge()
            assert len(mge_pot[0]) <= 20
            npt.assert_almost_equal(mge_pot[2], q_mass)
            vrms_map = kinematics.vrms_map(**kwargs_map)
            assert np.all(np.isfinite(vrms_map["model"]))
            assert np.all(vrms_map["model"] > 0)

    def test_double_sersic(self):
        kwargs_light_double = {
            "extended_source_type": "double_sersic",
            "mag_i": 21.097706,
            "angular_size_0": 0.934183,
            "angular_size_1": 0.934183,
            "n_sersic_0": 1,
            "n_sersic_1": 1,
            # e2 only: DoubleSersic.ellipticity returns e1 in lenstronomy convention
            "e1_0": 0.0,
            "e2_0": 0.1,
            "e1_1": 0.0,
            "e2_1": 0.1,
            "w0": 0.5,
            "w1": 0.5,
            "stellar_mass": 2.89782e10,
        }
        kwargs_light = dict(self.kwargs_light, e1=0, e2=0.1)
        kwargs_mass = dict(self.kwargs_mass, e1=0, e2=0.1)
        deflector_double = self._deflector(kwargs_mass, kwargs_light_double)
        kinematics_double = DeflectorKinematics(
            deflector_double, self.cosmo, self.z_source
        )
        kinematics = DeflectorKinematics(
            self._deflector(kwargs_mass, kwargs_light), self.cosmo, self.z_source
        )
        # one MGE for the composite light, with the composite axis ratio
        mge_lum = kinematics_double.light_mge()
        assert len(mge_lum[0]) <= 20
        _, q_light = ellipticity2phi_q(*deflector_double.light_ellipticity)
        npt.assert_almost_equal(mge_lum[2], q_light)
        # two identical components give the single Sersic result; the numerical
        # half-light radius of DoubleSersic only changes the MGE in the outskirts
        kwargs_map = {
            "map_half_size": 2.8,
            "grid_size": 0.4,
            "pix_size": 0.4,
            "inc": 80,
        }
        vrms_double = kinematics_double.vrms_map(**kwargs_map)
        vrms = kinematics.vrms_map(**kwargs_map)
        inner = (
            np.hypot(vrms["xbin"], vrms["ybin"]) < 2 * self.kwargs_light["angular_size"]
        )
        npt.assert_allclose(
            vrms_double["model"][inner], vrms["model"][inner], rtol=5e-3
        )
        kwargs_aperture = {"radius": self.kwargs_light["angular_size"]}
        npt.assert_allclose(
            kinematics_double.vrms_aperture(
                kwargs_aperture=kwargs_aperture, **kwargs_map
            ),
            kinematics.vrms_aperture(kwargs_aperture=kwargs_aperture, **kwargs_map),
            rtol=1e-3,
        )

    def test_raise(self):
        with pytest.raises(ValueError):
            DeflectorKinematics(
                self._deflector(self.kwargs_mass, kwargs_light={}),
                self.cosmo,
                self.z_source,
            )
        deflector = self._deflector(self.kwargs_mass)
        deflector.light_model_lenstronomy = lambda band=None: (["INTERPOL"], [{}])
        with pytest.raises(NotImplementedError):
            DeflectorKinematics(deflector, self.cosmo, self.z_source)


if __name__ == "__main__":
    pytest.main()
