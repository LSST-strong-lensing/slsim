## Type Ia Supernova Source Morphology (`supernovae.py`)

This module builds SED-weighted 2D spatial kernels for Type Ia supernovae. To keep microlensing simulations fast, we use semi-analytical heuristics instead of computationally expensive 3D radiative transfer, preserving only the essential physics. The key features of our SN Ia source morphology are:

* **Geometric Asymmetry (Ellipticity, $q$):** Normal SNe Ia show minor (~10%) global asphericity. We model this by projecting the explosion as a simple ellipsoid with ellipticity $q$. Defaults to $q = 1$ (spherical), but users can adjust it to explore the impact of asymmetry on microlensing signatures.
  $$R_{\text{eff}} = \sqrt{x^2 + (y/q)^2}$$
  > *Refs: Wang & Wheeler (2008)*

* **Wavelength-Dependent Velocity Shells (Chromatic Radius):** Because UV photons scatter further out in the ejecta than IR photons, the effective source size depends on wavelength ($R = v \cdot t$). We interpolate between an inner optical/IR velocity shell ($v_{\text{base}}$) and an outer UV shell ($v_{\text{UV}}$) using an exponential proxy, $\Pi(\lambda)$. This mimics the iron-group "opacity wall"—staying saturated in the UV ($\lambda \le 3000 \text{ \AA}$) and dropping to the electron-scattering floor in the IR.
  $$\Pi(\lambda) = \exp\left(-\frac{\lambda - 3000}{2000}\right)$$
  $$v_{\text{phot}}(\lambda) = v_{\text{base}} + (v_{\text{UV}} - v_{\text{base}}) \cdot \Pi(\lambda)$$
  $$R_{\text{phot}} = v_{\text{phot}}(\lambda) \cdot t$$
  > *Refs: opacity curve shape based on Kasen (2006, Fig 1).*

* **Wavelength-Dependent Limb Darkening:** The dense UV line blanket creates an extended, "foggy" atmosphere with steep limb darkening, while the more transparent IR yields a flatter intensity profile. To proxy this, the linear darkening coefficient ($u_{\text{limb}}$) slides smoothly between user-defined limits for the steep UV ($u_{\text{uv}}$) and flatter IR ($u_{\text{ir}}$).
  $$u_{\text{limb}} = u_{\text{uv}} - (u_{\text{uv}} - u_{\text{ir}}) \cdot \frac{\lambda - 3000}{7000}$$
  *(Note: $u_{\text{limb}}$ is safely clipped between $u_{\text{uv}}$ and $u_{\text{ir}}$)*
  $$I_{\text{cont}} = 1 - u_{\text{limb}} \left(1 - \mu \right)$$
  > *Refs: Chromatic profiles based on 3D SN Ia simulations (Bulla et al. 2015) and extended scattering atmospheres (Kasen et al. 2003).*

* **Bound-Bound Line Opacity (The "Ring" Effect):** At strong P-Cygni absorption troughs, material moving toward us absorbs the core light, while sideways-expanding material emits it, creating a "dark center, bright ring" geometry. To avoid slow Sobolev integrations, we use a fast 2D Gaussian ring proxy centered at $0.8 R_{\text{phot}}$ to approximate this high-opacity velocity shell.
  $$I_{\text{ring}} = \left(\frac{R_{\text{eff}}}{R_{\text{phot}}}\right)^2 \exp\left[-\left(\frac{R_{\text{eff}}}{0.8 R_{\text{phot}}}\right)^4\right]$$
  $$I_{\text{total}} = (1 - f_{\text{ring}})I_{\text{cont}} + f_{\text{ring}}I_{\text{ring}}$$
  > *Refs: Sobolev line formation geometry (Kasen et al. 2003) and Type Ia resonance scattering (Thomas, Nugent, & Meza 2011). Custom heuristic proxy.*

* **Broadband SED Integration:** The final observational kernel ($K$) is built by integrating these spatial profiles over wavelength, weighted by the `sncosmo` flux ($F_\lambda$) and instrument transmission ($T_\lambda$).
  $$K(x,y) = \int I_{\text{total}}(x,y,\lambda) \cdot F_\lambda(t) \cdot T_\lambda \, d\lambda$$
  > *Refs: Huber et al. (2019)*