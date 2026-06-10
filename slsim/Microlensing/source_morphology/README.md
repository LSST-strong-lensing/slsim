# Source Morphology Models

This directory provides 2D source morphology kernels for microlensing simulations. Each kernel represents the projected brightness profile of a source and is convolved with the magnification map in `lightcurve.py` to produce realistic microlensing light curves.

Each kernel is a **unit-sum probability distribution over the source plane** — it encodes only source shape, never absolute brightness. When convolved with the magnification map it returns the flux-weighted average magnification of the source. Absolute brightness at each epoch is handled entirely by the lens class (intrinsic SED + macro-model magnification). Kernels are renormalised to unit sum at every time step and after every spatial rescaling to ensure this separation is maintained throughout.

---

## Base Class (`source_morphology.py`)

`SourceMorphology` is the common base for all morphology types. It handles two
operating modes transparently:

- **Static** — a single kernel is computed once via `get_kernel_map()` and cached
  behind the `kernel_map` property.
- **Time-varying** — a sparse sequence of anchor kernels is supplied at
  initialization through `user_snapshots` (keys: `times`, `kernels`,
  `pixel_scales_m`). The base class sorts the anchors, pads all kernels to a common
  shape, stacks them into a 3D array, and exposes fully vectorised linear temporal
  interpolation via `_interpolate_snapshots()`. The public API for time-varying
  sources is `get_time_dependent_kernel_maps(times)`.

All subclasses must implement `get_kernel_map()`. Or if it's a time-varying morphology, they must provide `user_snapshots` at initialization and implement `_continuous_monochromatic_morphology()` to compute the anchor kernels. The base class handles all temporal interpolation and normalisation.

---

## Gaussian (`gaussian.py`)

A symmetric 2D Gaussian — the simplest possible source model, suitable as a
point-source proxy or for compact featureless sources.

$$K(x,y) \propto \exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right),
\qquad \sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}}$$

The kernel is evaluated on a grid of `(num_pix_x, num_pix_y)` pixels spanning
`(length_x, length_y)` arcseconds and normalised to unit sum.

**Key parameter:** `source_size` — the FWHM of the Gaussian in arcseconds.

---

## AGN Accretion Disk (`agn.py`)

Models the projected emission of a geometrically thin, optically thick accretion
disk (Shakura-Sunyaev thin-disk model) evaluated at a single observer-frame
wavelength or the effective wavelength of a photometric band.

The physical pixel scale is derived analytically from the gravitational radius
without needing to instantiate the 2D array first:

$$r_g = \frac{GM_{\rm BH}}{c^2},
\qquad \Delta x = \frac{2\,r_{\rm out}\,r_g}{2\,r_{\rm res}}$$

**Operating modes:**

- **Static** (default) — a single emission map is computed at the specified
  wavelength and cached. This is the common case for quasar microlensing.
- **Time-varying via `user_snapshots`** — the caller supplies pre-computed kernel
  grids. The base-class vectorised interpolator handles temporal queries. Per-step
  pixel scales are taken from the snapshots rather than derived analytically.
  Accessing the `kernel_map` property in this mode raises `AttributeError`; use
  `get_time_dependent_kernel_maps()` instead.

> **Note:** Analytical time-varying AGN generation (without external grids) is not
> yet implemented and raises `NotImplementedError`.

**Key parameters:** `black_hole_mass_exponent`, `r_out` (outer disk radius in
units of $r_g$), `r_resolution`, `inclination_angle`, `black_hole_spin`,
`eddington_ratio`, and either `observer_frame_wavelength_in_nm` or
`observing_wavelength_band`.

> Thin-disk SED model: Shakura & Sunyaev (1973),
> [A&A 24, 337](https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S).  
> Implementation via `slsim.Util.astro_util.calculate_accretion_disk_emission`.

---

## Type Ia Supernova (`supernovae.py`)

Builds a time-series of SED-weighted 2D spatial kernels for an expanding SN Ia
photosphere. The model uses semi-analytical heuristics to capture the essential
physics without expensive 3D radiative transfer. Because the photosphere expands as
$R = v \cdot t$, this morphology is **inherently time-varying**.

Kernels are computed analytically at sparse anchor times and interpolated at
arbitrary query times by the base-class vectorised interpolator.

### sncosmo usage

`sncosmo` provides exactly three inputs to the morphology:

1. **Time axis** — `model.mintime()` / `model.maxtime()` define the range over
   which anchor kernels are computed.
2. **Bandpass** — `sncosmo.get_bandpass(band)` supplies the filter wavelength
   array and transmission curve $T_\lambda$.
3. **Time-evolving SED** — `model.flux(t, wavelengths)` provides $F_\lambda(t)$
   to weight the spatial profiles during broadband integration.

The spatial morphology itself (photosphere radius, limb darkening, ring effect) is
computed entirely within `_continuous_monochromatic_morphology()` with no
`sncosmo` involvement.

**Model consistency:** In normal pipeline usage the sncosmo model instance is
injected directly from `SupernovaEvent` (via `sn_model_instance`) so the morphology
uses the exact same SN realisation — same template, stretch $x_1$, and colour $c$
— as the lightcurve. Two fallback paths exist for standalone use: a custom
`sn_modeldir` (SALT3), or a built-in template by name (default: `"hsiao"`).

### Physics

**1. Chromatic photosphere radius**

Iron-group line blanketing creates a wavelength-dependent opacity wall that is
saturated in the UV and falls toward the electron-scattering floor in the IR.
The expansion opacity as a function of wavelength is illustrated in Figure 1 of
Kasen et al. (2006). We proxy this with an exponential function $\Pi(\lambda)$
that interpolates between an inner optical/IR photosphere velocity $v_{\rm base}$
and an outer UV shell velocity $v_{\rm UV}$:

$$\Pi(\lambda) = \exp\!\left(-\frac{\lambda - 3000\,\text{\AA}}{2000\,\text{\AA}}\right)$$

$$v_{\rm phot}(\lambda) = v_{\rm base} + (v_{\rm UV} - v_{\rm base})\,\Pi(\lambda)$$

$$R_{\rm phot}(\lambda,\,t) = v_{\rm phot}(\lambda)\cdot t$$

> Kasen, D. et al. (2006), Figure 1 — expansion opacity of iron-group ejecta as a
> function of wavelength.
> [arXiv:astro-ph/0606449](https://arxiv.org/abs/astro-ph/0606449)

**2. Limb darkening**

We apply the standard linear limb darkening law:

$$\mu = \sqrt{1 - (R_{\rm eff}/R_{\rm phot})^2},
\qquad I_{\rm cont} = 1 - u_{\rm limb}(1-\mu)$$

The darkening coefficient $u_{\rm limb}$ is varied linearly with wavelength between
$u_{\rm UV} = 0.8$ (steep darkening in the UV, where the dense line blanket creates
an extended scattering atmosphere) and $u_{\rm IR} = 0.2$ (flatter profile in the
IR where the ejecta are more transparent):

$$u_{\rm limb}(\lambda) = u_{\rm UV}
  - (u_{\rm UV} - u_{\rm IR})\,\frac{\lambda - 3000\,\text{\AA}}{7000\,\text{\AA}}$$

$u_{\rm limb}$ is clipped to $[u_{\rm IR},\, u_{\rm UV}]$ to prevent unphysical
extrapolation.

> The linear limb darkening law is standard stellar-atmosphere theory. The
> chromatic variation of $u_{\rm limb}$ with wavelength is a **custom heuristic**
> motivated by the qualitative opacity behaviour shown in Kasen et al. (2006).
> There is no published source that prescribes these specific values for SN Ia;
> users should treat them as tunable parameters.

**3. Bound-bound line opacity — the "ring" effect**

At strong P-Cygni absorption troughs, material expanding toward the observer
absorbs photons from the photospheric core while material expanding sideways
re-emits them, producing a dark-centre / bright-ring geometry. Rather than a full
Sobolev integration, we use a fast Gaussian annulus peaked at $0.8\,R_{\rm phot}$:

$$I_{\rm ring} = \left(\frac{R_{\rm eff}}{R_{\rm phot}}\right)^2
\exp\!\left[-\left(\frac{R_{\rm eff}}{0.8\,R_{\rm phot}}\right)^4\right]$$

The ring fraction $f_{\rm ring}$ is set by a Gaussian proximity to three dominant
absorption features: Ca II H&K (3934 Å), Si II 6355 Å, and Ca II IR triplet
(8542 Å). This is a **custom heuristic proxy** for Sobolev line formation.

**4. Total intensity**

$$I_{\rm total} = (1-f_{\rm ring})\,I_{\rm cont} + f_{\rm ring}\,I_{\rm ring}$$

**5. Broadband SED integration**

The observational kernel is built by integrating the monochromatic spatial profiles
over wavelength, weighted by the sncosmo SED flux $F_\lambda(t)$ and the
instrument bandpass transmission $T_\lambda$:

$$K(x,y) = \int I_{\rm total}(x,y,\lambda)\cdot F_\lambda(t)\cdot T_\lambda\,d\lambda$$

For computational efficiency the integral is evaluated on a 1D radial grid and
projected to 2D via interpolation.

**6. Geometric asymmetry**

A global ellipticity parameter $q$ (default 1.0, i.e. spherical) stretches the
projected photosphere:

$$R_{\rm eff} = \sqrt{x^2 + (y/q)^2}$$

> Wang & Wheeler (2008), [ARA&A 46, 433](https://doi.org/10.1146/annurev.astro.46.060407.145139) —
> SN Ia global asphericity from spectropolarimetry.

**7. Normalisation**

The kernel is normalised to unit sum at three points: after each analytical anchor computation, after base-class temporal interpolation, and after spatial rescaling in `lightcurve.py`. This ensures the kernel always represents a pure spatial distribution and that absolute brightness information comes exclusively from the lens class calculations.

### Parameter summary

| Parameter                   | Default   | Description                                              |
| --------------------------- | --------- | -------------------------------------------------------- |
| `sn_model_instance`         | `None`    | Injected sncosmo model from `SupernovaEvent` (preferred) |
| `sn_modeldir`               | `None`    | Path to SALT3.NIR_WAVEEXT model files (fallback)         |
| `sn_model_name`             | `"hsiao"` | Built-in sncosmo template name (fallback)                |
| `observing_wavelength_band` | required  | Photometric band for SED weighting                       |
| `v_base_km_s`               | 10000     | Optical/IR photosphere velocity [km/s]                   |
| `v_uv_km_s`                 | 18000     | UV shell velocity [km/s]                                 |
| `u_limb_uv`                 | 0.8       | Limb darkening coefficient at 3000 Å                     |
| `u_limb_ir`                 | 0.2       | Limb darkening coefficient at 10000 Å                    |
| `ellipticity`               | 1.0       | Projected axis ratio $q$ (1.0 = spherical)               |
| `grid_pixels`               | 300       | Kernel map resolution (pixels per side)                  |
| `anchor_spacing_days`       | 5.0       | Source-frame days between analytical evaluations         |
| `user_snapshots`            | `None`    | Pre-computed kernel grids (bypasses all sncosmo calls)   |