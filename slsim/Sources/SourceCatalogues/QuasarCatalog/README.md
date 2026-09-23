# Quasar Host Matching

This document describes how quasars drawn from the Oguri & Marshall (2010) luminosity function are assigned a host galaxy from a large galaxy catalog (e.g., SkyPy), along with a black hole mass and an Eddington ratio.

The quasar catalog fixes the number of quasars and their luminosities. The matcher draws the remaining variables—host galaxy, black hole mass ($M_{\rm BH}$), and Eddington ratio ($\lambda_{\rm Edd}$)—from their joint distribution *conditioned on* the quasar's predetermined luminosity. Because the scaling relations and distributions are Gaussian in log-space, this conditional distribution has a closed form and is sampled exactly without grid searches.

## The Matching Algorithm

For a single quasar with redshift $z$ and absolute magnitude $M_i(z=2)$:

**1. Calculate Target Bolometric Luminosity**
The magnitude fixes the 3000 Å luminosity, which is converted to bolometric luminosity ($L_{\rm bol}$) using a single zero point and the Richards et al. (2006) bolometric correction:
$$\log_{10} \lambda L_\lambda(3000\,\text{\AA}) = 35.27 - 0.4\,M_i(z=2)$$
$$L_{\rm bol} = 5.15\,\lambda L_\lambda(3000\,\text{\AA})$$
This gets a scatter of 0.1 dex. For the matching math, we normalize this to the Eddington luminosity of one solar mass ($L_{\rm Edd,1} = 1.257\times10^{38}\,\text{erg s}^{-1}$) to define the required target variable $c$:
$$c = \log_{10} L_{\rm bol} - \log_{10} L_{\rm Edd,1}$$

**2. Select Candidate Hosts**
Galaxies within a redshift slice $z \pm \Delta z$ are selected. The slice widens geometrically until it holds at least `min_candidates`.

**3. Define the Black Hole Mass Prior**
Each candidate galaxy gets a Gaussian prior for its expected black hole mass, with mean $m_k$ and intrinsic scatter $s_k$, based on its `galaxy_type`:
* **Red galaxies (Bulges):** $m_k = \log_{10}[0.309 \times 10^9\,(\sigma_e/200\,\text{km s}^{-1})^{4.38}]$ with $s_k = 0.29$ dex.
* **Blue galaxies (Disks):** $m_k = 7.45 + 1.05\,\log_{10}(M_\star/10^{11} M_\odot)$ with $s_k = 0.24$ dex.

**4. Define the Eddington Ratio Prior**
The population Eddington ratio ($x = \log_{10}\lambda_{\rm Edd}$) is modeled as a universal Gaussian with mean $\mu = -1.15$ and width $\sigma_\lambda = 0.30$ dex.

**5. Match and Draw (Bayesian Updating)**
To produce the required luminosity $c$, the drawn Eddington ratio and black hole mass must exactly satisfy $c = x + \log_{10} M_{\rm BH}$. 

First, we calculate the probability that candidate $k$ can host this quasar by evaluating the sum of the two priors at $c$:

$$p(k \mid c) \propto \frac{1}{\sqrt{s_k^2+\sigma_\lambda^2}}\exp\!\left[-\frac{(c-m_k-\mu)^2}{2(s_k^2+\sigma_\lambda^2)}\right]$$

A host is drawn using these weights. Then, the specific Eddington ratio $x$ is drawn from a conditional Gaussian. This conditional distribution represents the statistical compromise between the population average ($\mu$) and what the specific galaxy needs ($c - m_k$), weighted by their precisions (inverse variances):

$$V_k=\left(\sigma_\lambda^{-2}+s_k^{-2}\right)^{-1}, \qquad \bar{x}_k=V_k\left[\frac{\mu}{\sigma_\lambda^2}+\frac{c-m_k}{s_k^2}\right]$$

Once $x$ is drawn from $\mathcal{N}(\bar{x}_k, V_k)$, the black hole mass is deterministically set to $\log_{10} M_{\rm BH} = c - x$ to conserve energy perfectly. 

**6. Rejection**
A quasar is dropped if every candidate is more than `max_offset_sigma` combined standard deviations ($\sqrt{s_k^2+\sigma_\lambda^2}$) away from producing it. Dropped quasars trigger a warning.

---

## Scientific Justifications & Parameter Choices

* **Luminosity Zero Point:** The 35.27 zero point is fitted to the H$\beta$/Mg II sample of Wu & Shen (2022) to ensure the mock sits in the same $L_{3000}$ system as the virial masses. (A pure $\alpha_\nu = -0.5$ continuum would put it at 35.20; the data sit 0.07 dex higher).
* **Bolometric Correction Scatter:** 0.1 dex matches the per-object spread in Runnoe, Brotherton & Shang (2012). Following standard SDSS catalogs, no further disk anisotropy penalty is applied.
* **Eddington Ratio Distribution:** Bounded power-law distributions (like Korytov et al. 2019) sit too high for objects drawn from a luminosity function. The chosen Gaussian ($\mu = -1.15$, $\sigma_\lambda = 0.30$) is fitted so the mock reproduces the observed median of Wu & Shen (2022) once flux limits are applied.
* **Mass Scaling Relations:** M–$\sigma$ (Kormendy & Ho 2013) is calibrated on bulges and applied to red galaxies. Disk/blue galaxies use the total-stellar-mass relation (Reines & Volonteri 2015), whose normalization sits over a dex lower. No morphology cut is applied.

## Validation

Compared with the H$\beta$/Mg II subsample of Wu & Shen (2022) over $0.7 < z < 2$ and $i < 20.5$, medians of mock minus DR16Q:

| $\log M_{\rm BH}$ | $\log \lambda_{\rm Edd}$ | $\log L_{\rm bol}$ | $M_i$ |
|---|---|---|---|
| −0.02 | −0.06 | −0.05 | +0.14 |

The $L_{3000}$ zero point and the Eddington ratio distribution are fitted to this catalog, ensuring agreement by construction. The spread of $\log M_{\rm BH}$ at fixed $L_{\rm bol}$ (0.53–0.59 dex, observed 0.50–0.69) is successfully reproduced. The residual luminosity-dependent tilt in $\log M_{\rm BH}$ arises because the SkyPy velocity dispersions run low, forcing luminous quasars into the extreme tail of available hosts.

## Implementation Details

* **`log_black_hole_mass(galaxy_type, vel_disp, stellar_mass)`**: Returns $m_k$ and $s_k$ based on type (Step 3).
* **`log_bolometric_luminosity(m_i, scatter, rng)`**: Implements Step 1.
* **`QuasarHostMatch.match()`**: Runs steps 2–6 for every quasar. 

*Note: `QuasarRate` uses the tabulated Richards et al. (2006) K-correction normalized to $z = 2$.*

## References
* Kormendy & Ho (2013), ARA&A 51, 511, [arXiv:1304.7762](https://arxiv.org/abs/1304.7762)
* Korytov et al. (2019), ApJS 245, 26, [arXiv:1907.06530](https://arxiv.org/abs/1907.06530)
* Oguri & Marshall (2010), MNRAS 405, 2579, [arXiv:1001.2037](https://arxiv.org/abs/1001.2037)
* Reines & Volonteri (2015), ApJ 813, 82, [arXiv:1508.06274](https://arxiv.org/abs/1508.06274)
* Richards et al. (2006), AJ 131, 2766, [arXiv:astro-ph/0601434](https://arxiv.org/abs/astro-ph/0601434)
* Richards et al. (2006), ApJS 166, 470, [arXiv:astro-ph/0601558](https://arxiv.org/abs/astro-ph/0601558)
* Runnoe, Brotherton & Shang (2012), MNRAS 422, 478, [arXiv:1201.5155](https://arxiv.org/abs/1201.5155)
* Wu & Shen (2022), ApJS 263, 42, [arXiv:2209.03987](https://arxiv.org/abs/2209.03987)