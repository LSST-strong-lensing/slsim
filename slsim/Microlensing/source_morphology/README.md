## Supernova Source Morphology (`supernovae.py`)

This module generates time-dependent, SED-weighted 2D spatial kernels for supernovae. It utilizes semi-analytical heuristics to bypass the computational cost of full 3D radiative transfer simulations while maintaining physical rigor.

### Physical Models & Governing Equations

* **Expansion:** The base photospheric radius ($R_{\text{base}}$) scales by $v = 10^4 \text{ km/s}$, the characteristic expansion velocity for Type Ia ejecta.
  $$R_{\text{base}} = v \cdot t$$
  > *Refs: Dobler & Keeton (2006)*

* **Geometric Asymmetry (Ellipticity, $q$):** Accounts for intrinsic global asphericity observed via spectropolarimetry.
  $$R_{\text{eff}} = \sqrt{x^2 + (y/q)^2}$$
  > *Refs: Wang & Wheeler (2008)*

* **Chromatic Radius (UV Line Blanketing):** Models the "nested onion" structure of the SN atmosphere. The radius is anchored at $0.8 R_{\text{base}}$ to represent the deep photosphere visible in transparent IR, with a $0.5$ inflation term to mimic the UV "opacity wall" ($R_{\text{UV}} \approx 1.6 R_{\text{IR}}$). The expansion transitions at $3000 \text{ \AA}$ (iron-group forest onset) with a $2000 \text{ \AA}$ decay constant for a smooth optical/IR retreat.
  $$R_{\text{phot}} = R_{\text{base}} \left[0.8 + 0.5 \cdot \max\left(0, \exp\left(-\frac{\lambda - 3000}{2000}\right)\right)\right]$$
  > *Refs: Pinto & Eastman (2000, Fig 1)*

* **Wavelength-Dependent Limb Darkening:** The coefficient $u_{\text{limb}}$ is a linear function of $\lambda$. It is anchored at $0.8$ for UV ($3000 \text{ \AA}$) and $0.2$ for IR ($10,000 \text{ \AA}$), reflecting the deeper, hotter layers probed by UV photons.
  $$u_{\text{limb}} = \min\left(0.8, \max\left(0.2, 0.8 - 0.6 \cdot \frac{\lambda - 3000}{7000}\right)\right)$$
  $$I_{\text{cont}} = 1 - u_{\text{limb}} \left(1 - \mu \right)$$
  > *Refs: Bulla et al. (2015); Eastman et al. (1996)*

* **Bound-Bound Line Opacity (The "Ring" Effect):** At major absorption lines (e.g., Si II $\lambda 6355$), intensity is shifted to the edges as the core is obscured. The Gaussian ring is centered at $0.8 R_{\text{phot}}$ to match the projected velocity shell where line opacity dominates.
  $$I_{\text{ring}} = \left(\frac{R_{\text{eff}}}{R_{\text{phot}}}\right)^2 \exp\left[-\left(\frac{R_{\text{eff}}}{0.8 R_{\text{phot}}}\right)^4\right]$$
  > *Refs: Kasen et al. (2003, Fig 4)*

* **Broadband SED Integration:** Integrates monochromatic profiles ($I_{\text{total}}$) weighted by `sncosmo` flux ($F_\lambda$) and instrument transmission ($T_\lambda$) to produce the final observational kernel ($K$).
  $$K(x,y) = \int I_{\text{total}}(x,y,\lambda) \cdot F_\lambda(t) \cdot T_\lambda \, d\lambda$$
  > *Refs: Huber et al. (2019)*