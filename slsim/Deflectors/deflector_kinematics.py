"""This module predicts the stellar kinematics of a deflector with the
axisymmetric Jeans Anisotropic Modelling (JAM) of JamPy, using the deflector's
total (lensing) mass as the potential and its light as the tracer.

This module is in active development. Currently, it only supports
constant stellar orbit anisotropy, and Gaussian PSF.
"""

import copy
import warnings

import numpy as np
import lenstronomy.Util.constants as constants
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import data_util, util
from lenstronomy.Util.param_util import ellipticity2phi_q

import mgefit
from jampy.axi.jam_axi_proj import jam_axi_proj

from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy

# position angle difference [deg] between light and mass above which JAM's assumption of
# a common symmetry axis is flagged
_MISALIGNMENT_WARNING_DEG = 10
# minimum intrinsic axis ratio accepted by jampy's deprojection
_Q_INTR_MIN_JAMPY = 0.05


class DeflectorKinematics:
    """Axisymmetric JAM prediction of the line-of-sight second moment V_rms =
    sqrt(<v_los^2>) of a deflector galaxy (single lens plane).

    The deflector's total (lensing) mass is the potential and its light
    is the tracer. Each is a multi-Gaussian expansion of its spherical
    profile, flattened with the axis ratio of the composite profile.
    V_rms is the observable prediction and is in general different from
    Deflector.velocity_dispersion(), which normalizes the mass model.
    """

    def __init__(self, deflector_class, astropy_cosmo, z_source):
        """

        :param deflector_class: deflector with extended light
        :type deflector_class: ~slsim.Deflectors.deflector.Deflector instance
        :param astropy_cosmo: cosmology
        :type astropy_cosmo: ~astropy.cosmology class
        :param z_source: source redshift, used to express the mass model in
            lenstronomy units
        :type z_source: float
        """

        self._deflector = deflector_class

        # save the effective radius to conveniently set the radial range of the MGE fits
        self._r_eff = deflector_class.angular_size_light

        if deflector_class.light.extended_source_type is None:
            raise ValueError("DeflectorKinematics requires a deflector with light.")

        # access light kwargs
        light_model_list, kwargs_light = deflector_class.light_model_lenstronomy()
        if "INTERPOL" in light_model_list:
            raise NotImplementedError(
                "DeflectorKinematics does not support interpolated light profiles."
            )
        self.light_model_list = light_model_list
        self.kwargs_light = kwargs_light

        e1_light, e2_light = deflector_class.light_ellipticity
        phi_light, self.q_light = ellipticity2phi_q(e1=e1_light, e2=e2_light)
        # light position angle in the lenstronomy (image) frame [rad], counterclockwise
        # from +x, to rotate V_rms maps into the sky frame
        self._phi_light_sky, _ = ellipticity2phi_q(
            *ellipticity_slsim_to_lenstronomy(e1_slsim=e1_light, e2_slsim=e2_light)
        )

        # initialize lenstronomy.Cosmo.LensCosmo
        lens_cosmo = LensCosmo(
            z_lens=deflector_class.redshift, z_source=z_source, cosmo=astropy_cosmo
        )
        self._lens_cosmo = lens_cosmo

        # calculate q_mass
        phi_mass, self.q_mass = ellipticity2phi_q(
            *deflector_class.composite_mass_ellipticity(
                lens_cosmo=lens_cosmo, r_window=3 * self._r_eff
            )
        )

        # check light & mass PA misalignment
        delta_phi = np.degrees((phi_light - phi_mass + np.pi / 2) % np.pi - np.pi / 2)
        if abs(delta_phi) > _MISALIGNMENT_WARNING_DEG:
            warnings.warn(
                f"Axis ratios: q_light={self.q_light:.3f}, q_mass={self.q_mass:.3f}. "
                f"Position angles of light and mass differ by {delta_phi:.1f} deg. "
                "JAM assumes a common symmetry axis; the difference is ignored.",
                category=UserWarning,
                stacklevel=2,
            )

        lens_model_list, kwargs_lens = deflector_class.mass.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo, spherical=True
        )
        self.lens_model_list = lens_model_list
        self.kwargs_lens = kwargs_lens

    def light_mge(self, ngauss=20, r_min=1e-3, r_max_reff=10, n_r=1000):
        """MGE of the deflector light, the tracer of JAM.

        :param ngauss: number of Gaussians
        :type ngauss: int
        :param r_min: minimum radius of the fit [arcsec]
        :param r_max_reff: maximum radius of the fit in units of the
            half-light radius
        :param n_r: number of radii of the fit
        :type n_r: int
        :return: surface brightness (arbitrary units), dispersion along
            the major axis [arcsec] and observed axis ratio of each
            Gaussian, as model_lum_proj of jam_axi_proj
        """
        light_model = LightModel(self.light_model_list)

        # amplitudes from magnitudes keep the relative weights of the components; the
        # overall normalization is irrelevant for JAM
        kwargs_light_amp = data_util.magnitude2amplitude(
            light_model, self.kwargs_light, magnitude_zero_point=0
        )

        return self._light_mge(
            light_model,
            kwargs_light_amp,
            q_light=self.q_light,
            r_eff=self._r_eff,
            ngauss=ngauss,
            r_min=r_min,
            r_max_factor=r_max_reff,
            n_r=n_r,
        )

    def mass_mge(self, ngauss=20, r_min=1e-3, r_max_reff=400, n_r=1000):
        """MGE of the deflector total (lensing) mass, the potential of JAM.

        :param ngauss: number of Gaussians
        :type ngauss: int
        :param r_min: minimum radius of the fit [arcsec]
        :param r_max_reff: maximum radius of the fit in units of the
            half-light radius
        :param n_r: number of radii of the fit
        :type n_r: int
        :return: surface mass density [M_sun/pc^2], dispersion along the
            major axis [arcsec] and observed axis ratio of each
            Gaussian, as model_pot_proj of jam_axi_proj
        """
        return self._mass_mge(
            self.lens_model_list,
            self.kwargs_lens,
            q_mass=self.q_mass,
            lens_cosmo=self._lens_cosmo,
            r_eff=self._r_eff,
            ngauss=ngauss,
            r_min=r_min,
            r_max_factor=r_max_reff,
            n_r=n_r,
        )

    def inclination_min(self, q_intr_min=0.05):
        """Minimum inclination at which both light and mass, as oblate
        components, have an intrinsic axis ratio of at least q_intr_min.

        :param q_intr_min: minimum intrinsic axis ratio, at least 0.05
            (the minimum of jampy)
        :return: minimum inclination [deg]
        """
        q_obs = min(self.q_light, self.q_mass)
        if q_intr_min < _Q_INTR_MIN_JAMPY:
            raise ValueError(
                f"q_intr_min={q_intr_min} is below jampy's minimum of {_Q_INTR_MIN_JAMPY}."
            )
        if q_obs <= q_intr_min:
            raise ValueError(
                f"q_obs={q_obs:.3f} is not deprojectable with q_intr >= {q_intr_min}."
            )
        cos_inc_max = np.sqrt((q_obs**2 - q_intr_min**2) / (1 - q_intr_min**2))
        return np.degrees(np.arccos(cos_inc_max))

    def vrms_map(
        self,
        map_half_size,
        grid_size,
        pix_size=0.0,
        inc=None,
        beta=0.0,
        sigma_psf=0.0,
        norm_psf=1.0,
        kwargs_mge_light=None,
        kwargs_mge_mass=None,
        dimensionless=False,
        frame="galaxy",
        **kwargs_jam_axi_proj,
    ):
        """V_rms map from jampy's jam_axi_proj (moment='zz'), or the map of the
        dimensionless kinematics J, centered on the deflector.

        The map is evaluated on a square grid of bins from
        DeflectorKinematics.kinematic_grid in the frame of the deflector's
        major axis, if frame="galaxy". If frame="sky", the bins are then
        rotated into the frame of the lenstronomy (image) coordinates.

        :param map_half_size: half size of the map [arcsec]
        :param grid_size: size of each bin of the map [arcsec]
        :param pix_size: pixel size of the kinematic data [arcsec], used in
            the PSF convolution (pixsize of jam_axi_proj)
        :param inc: inclination [deg], within (inclination_min(), 90]. If
            None, drawn at random with draw_inclination(inclination_min())
            and printed; for a reproducible draw or another q_intr_min, use
            draw_inclination(inclination_min(q_intr_min), rng).
        :type inc: float or None
        :param beta: velocity anisotropy, constant or one value per
            Gaussian of the light MGE
        :param sigma_psf: dispersions of the circular Gaussians describing
            the PSF of the kinematic observations [arcsec] (sigmapsf of
            jam_axi_proj)
        :param norm_psf: fraction of the total PSF flux contained in the
            circular Gaussians describing the PSF of the kinematic
            observations. The PSF will be used for seeing convolution of
            the model kinematics (normpsf of jam_axi_proj).
        :param kwargs_mge_light: keyword arguments of light_mge; if None,
            its defaults
        :type kwargs_mge_light: dict or None
        :param kwargs_mge_mass: keyword arguments of mass_mge; if None,
            its defaults
        :type kwargs_mge_mass: dict or None
        :param dimensionless: if True, "model" is the dimensionless
            kinematics J = V_rms^2 D_ds / (D_s c^2) (see vrms_to_J) instead
            of V_rms [km/s], with D_ds and D_s from z_source and the
            cosmology of the class
        :type dimensionless: bool
        :param frame: frame of the returned bins. "galaxy": x along the
            projected major axis of the light. "sky": axes of the
            lenstronomy (image) coordinates, i.e. the galaxy frame rotated
            counterclockwise by the light position angle.
        :type frame: str
        :param kwargs_jam_axi_proj: other keyword arguments of
            jam_axi_proj (e.g. nrad, nang, step); plot and quiet default
            to False and True. The MGEs, mbh, distance and moment are set
            by this class.
        :return: dictionary with "xbin", "ybin" [arcsec] (1d arrays of
            the bin centers in the given frame, relative to the deflector
            center), "model" (V_rms [km/s], or J if dimensionless), "flux"
            (PSF convolved model surface brightness, arbitrary units),
            "shape" (2d shape of the map, to reshape the 1d arrays),
            "dimensionless" (whether "model" is dimensionless J), "dds" and "ds" (angular
            diameter distances deflector-source and observer-source [Mpc],
            to convert "model" with vrms_to_J or J_to_vrms), "frame" and
            "pa_light_sky"
            (position angle of the light in the sky frame [rad],
            counterclockwise from +x, in the lenstronomy convention;
            util.rotate(xbin, ybin, pa_light_sky) converts sky-frame bins to
            the galaxy frame and util.rotate(xbin, ybin, -pa_light_sky)
            galaxy-frame bins to the sky frame, with util from
            lenstronomy.Util)
        """
        if frame not in ("galaxy", "sky"):
            raise ValueError(
                f"frame {frame} not supported. Choose among ['galaxy', 'sky']."
            )

        # make kinematic grid of bin centers
        xbin, ybin, shape = self.kinematic_grid(map_half_size, grid_size)

        inc_min = self.inclination_min()
        if inc is None:  # draw random inclination
            inc = self.draw_inclination(inc_min=inc_min)
            print(
                f"inc not given: randomly drawn inc = {inc:.2f} deg from isotropic "
                f"orientations within ({inc_min:.2f}, 90] deg."
            )
        elif not inc_min < inc <= 90:  # check user's input
            raise ValueError(
                f"inc={inc} deg is outside the allowed range ({inc_min:.2f}, 90] deg."
            )

        reserved = ["model_lum_proj", "model_pot_proj", "mbh", "distance", "moment"]
        reserved_given = [key for key in reserved if key in kwargs_jam_axi_proj]
        if len(reserved_given) > 0:
            raise ValueError(
                f"{reserved_given} are set by DeflectorKinematics and cannot be given."
            )
        kwargs_jam_axi_proj.setdefault("plot", False)
        kwargs_jam_axi_proj.setdefault("quiet", True)

        kwargs_mge_light = kwargs_mge_light if kwargs_mge_light is not None else {}
        kwargs_mge_mass = kwargs_mge_mass if kwargs_mge_mass is not None else {}

        jam = jam_axi_proj(
            model_lum_proj=self.light_mge(**kwargs_mge_light),
            model_pot_proj=self.mass_mge(**kwargs_mge_mass),
            inc=inc,
            mbh=0.0,
            distance=self._lens_cosmo.dd,
            xbin=xbin,
            ybin=ybin,
            beta=beta,
            moment="zz",
            sigmapsf=sigma_psf,
            normpsf=norm_psf,
            pixsize=pix_size,
            **kwargs_jam_axi_proj,
        )

        if dimensionless:
            # convert V_rms [km/s] to dimensionless kinematics J = V_rms^2 D_ds / (D_s c^2)
            jam.model = self.vrms_to_J(
                vrms=jam.model, dds=self._lens_cosmo.dds, ds=self._lens_cosmo.ds
            )

        xbin, ybin = jam.xbin, jam.ybin
        if frame == "sky":
            # util.rotate rotates the coordinate axes counterclockwise by the angle,
            # i.e. the points clockwise; -phi rotates the points counterclockwise
            # by phi, from the major axis frame into the sky frame
            xbin, ybin = util.rotate(xbin, ybin, -self._phi_light_sky)

        return {
            "xbin": xbin,
            "ybin": ybin,
            "model": jam.model,
            "flux": jam.flux,
            "shape": shape,
            "dimensionless": dimensionless,
            "dds": self._lens_cosmo.dds,
            "ds": self._lens_cosmo.ds,
            "frame": frame,
            "pa_light_sky": self._phi_light_sky,
        }

    def vrms_aperture(
        self,
        vrms_map_dict,
        aperture_type="circular",
        kwargs_aperture=None,
    ):
        """Flux-weighted V_rms (or J) of a map from vrms_map within an aperture
        centered on the deflector.

        The aperture is a circle or an annulus around (0, 0), so the
        result is the same for maps in the "galaxy" and "sky" frames.

        :param vrms_map_dict: output of vrms_map
        :type vrms_map_dict: dict
        :param aperture_type: "circular" with kwargs_aperture={"radius":
            r} or "annulus" with kwargs_aperture={"r_in": r_in, "r_out":
            r_out}
        :type aperture_type: str
        :param kwargs_aperture: aperture size [arcsec], in the format of
            aperture_type; a ValueError is raised otherwise
        :type kwargs_aperture: dict
        :return: V_rms within the aperture [km/s], or J within the
            aperture if vrms_map_dict["dimensionless"] is True; nan,
            with a warning, if no bin of the map is within the aperture
        """

        mask = self._make_mask(
            xbin=vrms_map_dict["xbin"],
            ybin=vrms_map_dict["ybin"],
            aperture_type=aperture_type,
            kwargs_aperture=kwargs_aperture,
        )

        return self.flux_weighted_model(
            model=vrms_map_dict["model"],
            dimensionless_model=vrms_map_dict["dimensionless"],
            flux=vrms_map_dict["flux"],
            mask=mask,
        )

    @staticmethod
    def sphericalize_kwargs(kwargs_list):
        """Copy of lenstronomy keyword arguments with the ellipticity removed
        and centered at (0, 0).

        :param kwargs_list: list of lenstronomy keyword arguments
        :type kwargs_list: list of dict
        :return: list of spherical keyword arguments
        """
        kwargs_spherical = copy.deepcopy(kwargs_list)
        for kwargs in kwargs_spherical:
            if "e1" in kwargs and "e2" in kwargs:
                kwargs["e1"] = 0.0
                kwargs["e2"] = 0.0
            if "center_x" in kwargs and "center_y" in kwargs:
                kwargs["center_x"] = 0.0
                kwargs["center_y"] = 0.0
        return kwargs_spherical

    @staticmethod
    def fit_mge_1d(radius, profile, ngauss=20, outer_slope=1):
        """Multi-Gaussian expansion of a radial profile with mgefit.fit_1d.

        :param radius: radii of the profile [arcsec]
        :param profile: profile values at radius
        :param ngauss: number of Gaussians
        :type ngauss: int
        :param outer_slope: maximum logarithmic slope of the profile
            beyond the last radius
        :return: peak surface density and dispersion [arcsec] of each
            Gaussian
        """

        mge = mgefit.fit_1d(
            radius, profile, ngauss=ngauss, outer_slope=outer_slope, quiet=True
        )
        total_counts, sigma = mge.sol
        amplitude = total_counts / (np.sqrt(2 * np.pi) * sigma)
        return amplitude, sigma

    @staticmethod
    def _light_mge(
        light_model_class,
        kwargs_light,
        q_light,
        r_eff,
        ngauss=20,
        r_min=1e-3,
        r_max_factor=10,
        n_r=1000,
    ):
        """MGE of the total light profile, flattened with a single axis ratio.

        All light components are made spherical and summed. The
        resulting profile along the intermediate axis is fitted once,
        and the Gaussian dispersions are converted to the major axis.

        :param light_model_class: lenstronomy light model class
        :param kwargs_light: lenstronomy light keyword arguments with
            amplitudes
        :param q_light: axis ratio of the composite light profile
        :param r_eff: half-light radius [arcsec]
        :param ngauss: number of Gaussians
        :param r_min: minimum radius of the fit [arcsec]
        :param r_max_factor: maximum radius of the fit in units of r_eff
        :param n_r: number of radii of the fit
        :return: surface brightness, dispersion along the major axis
            [arcsec] and observed axis ratio of each Gaussian
        """
        radius = np.geomspace(r_min, r_max_factor * r_eff, n_r)
        profile = light_model_class.surface_brightness(
            x=radius,
            y=0,
            kwargs_list=DeflectorKinematics.sphericalize_kwargs(kwargs_light),
        )
        surf, sigma_intermediate = DeflectorKinematics.fit_mge_1d(
            radius, profile, ngauss=ngauss
        )
        sigma = sigma_intermediate / np.sqrt(q_light)
        return surf, sigma, np.full_like(surf, q_light)

    @staticmethod
    def _mass_mge(
        lens_model_list,
        kwargs_lens,
        q_mass,
        lens_cosmo,
        r_eff,
        ngauss=20,
        r_min=1e-3,
        r_max_factor=400,
        n_r=1000,
    ):
        """MGE of the total projected mass, flattened with a single axis ratio.

        All mass components are made spherical and summed. The resulting
        surface mass density along the intermediate axis is fitted once,
        and the Gaussian dispersions are converted to the major axis.

        :param lens_model_list: lenstronomy lens model list
        :param kwargs_lens: lenstronomy lens keyword arguments
        :param q_mass: axis ratio of the composite mass profile
        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param r_eff: half-light radius [arcsec]
        :param ngauss: number of Gaussians
        :param r_min: minimum radius of the fit [arcsec]
        :param r_max_factor: maximum radius of the fit in units of r_eff
        :param n_r: number of radii of the fit
        :return: surface mass density [M_sun/pc^2], dispersion along the
            major axis [arcsec] and observed axis ratio of each Gaussian
        """
        radius = np.geomspace(r_min, r_max_factor * r_eff, n_r)
        kappa = LensModel(lens_model_list=lens_model_list).kappa(
            x=radius, y=0, kwargs=DeflectorKinematics.sphericalize_kwargs(kwargs_lens)
        )
        # M_sun/Mpc^2 to M_sun/pc^2
        profile = lens_cosmo.kappa2proj_mass(kappa) / 1e12

        surf, sigma_intermediate = DeflectorKinematics.fit_mge_1d(
            radius, profile, ngauss=ngauss
        )
        sigma = sigma_intermediate / np.sqrt(q_mass)
        return surf, sigma, np.full_like(surf, q_mass)

    @staticmethod
    def draw_inclination(inc_min, rng=None):
        """Draws an inclination from isotropic orientations (uniform in cos i)
        above inc_min.

        :param inc_min: minimum inclination [deg]
        :param rng: random number generator or seed
        :type rng: ~numpy.random.Generator, int or None
        :return: inclination [deg]
        """
        rng = np.random.default_rng(rng)
        cos_inc = rng.uniform(0.0, np.cos(np.radians(inc_min)))
        return np.degrees(np.arccos(cos_inc))

    @staticmethod
    def kinematic_grid(map_half_size, grid_size):
        """Square grid of bin centers.

        :param map_half_size: half size of the grid [arcsec]
        :param grid_size: size of each grid cell [arcsec]
        :return: x and y of the bin centers [arcsec] (1d arrays) and the
            2d shape
        """
        axis = np.arange(-map_half_size + grid_size / 2, map_half_size, grid_size)
        x_grid, y_grid = np.meshgrid(axis, axis)
        return x_grid.ravel(), y_grid.ravel(), x_grid.shape

    @staticmethod
    def _make_mask(xbin, ybin, aperture_type, kwargs_aperture):
        """Mask of bins within an aperture centered on (0, 0).

        :param xbin: x of each bin [arcsec]
        :param ybin: y of each bin [arcsec]
        :param aperture_type: "circular" with kwargs_aperture={"radius":
            r} or "annulus" with kwargs_aperture={"r_in": r_in, "r_out":
            r_out}
        :type aperture_type: str
        :param kwargs_aperture: aperture size [arcsec]; radius > 0, or 0
            <= r_in < r_out
        :type kwargs_aperture: dict
        :return: boolean mask of bins within the aperture
        """
        required_keys = {"circular": {"radius"}, "annulus": {"r_in", "r_out"}}
        if aperture_type not in required_keys:
            raise ValueError(
                f"aperture_type {aperture_type} not supported. Choose among "
                f"{list(required_keys)}."
            )
        keys = required_keys[aperture_type]
        if not isinstance(kwargs_aperture, dict) or set(kwargs_aperture) != keys:
            raise ValueError(
                f"kwargs_aperture={kwargs_aperture} is not valid for aperture_type "
                f"{aperture_type}; it must be a dict with keys {sorted(keys)}."
            )

        r2 = xbin**2 + ybin**2
        if aperture_type == "circular":
            radius = kwargs_aperture["radius"]
            if not radius > 0:
                raise ValueError(f"radius={radius} must be positive.")
            return r2 <= radius**2
        r_in, r_out = kwargs_aperture["r_in"], kwargs_aperture["r_out"]
        if not 0 <= r_in < r_out:
            raise ValueError(
                f"r_in={r_in} and r_out={r_out} must be 0 <= r_in < r_out."
            )
        return (r2 >= r_in**2) & (r2 <= r_out**2)

    @staticmethod
    def flux_weighted_model(model, dimensionless_model, flux, mask):
        """Flux-weighted V_rms or J of the bins within an aperture.

        V_rms: sqrt(sum(flux * V_rms^2) / sum(flux)). J, which is
        proportional to V_rms^2: sum(flux * J) / sum(flux), i.e. J of the
        flux-weighted V_rms.

        :param model: V_rms [km/s] or J of each bin
        :param dimensionless_model: True if model is J, False if V_rms
        :type dimensionless_model: bool
        :param flux: flux of each bin
        :param mask: boolean mask of bins within the aperture
        :return: V_rms within the aperture [km/s] or J within the aperture;
            nan, with a warning, if mask selects no bin
        """
        if not np.any(mask):
            warnings.warn(
                "No bin of the map is within the aperture; returning nan. Use a "
                "larger aperture or a finer grid.",
                category=UserWarning,
                stacklevel=2,
            )
            return np.nan
        flux, model = flux[mask], model[mask]
        if dimensionless_model:
            return np.sum(flux * model) / np.sum(flux)
        return np.sqrt(np.sum(flux * model**2) / np.sum(flux))

    @staticmethod
    def vrms_to_J(vrms, dds, ds):
        """Dimensionless kinematics J = V_rms^2 D_ds / (D_s c^2).

        :param vrms: V_rms [km/s]
        :param dds: angular diameter distance between deflector and
            source
        :param ds: angular diameter distance to the source
        :return: J
        """
        return vrms**2 * dds / ds / constants.c**2 * 1e6

    @staticmethod
    def J_to_vrms(J, dds, ds):
        """Inverse of vrms_to_J.

        :param J: dimensionless kinematics J
        :param dds: angular diameter distance between deflector and
            source
        :param ds: angular diameter distance to the source
        :return: V_rms [km/s]
        """
        return np.sqrt(J * ds / dds) * constants.c / 1000
