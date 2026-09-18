# Quasar Host Matching

Here we describe how quasars drawn from the Oguri & Marshall (2010) luminosity function are assigned a host galaxy from a large galaxy catalog (e.g. one generated with the SkyPy pipeline), together with a black hole mass and an Eddington ratio.

The quasar catalog already fixes how many quasars there are and how bright each one is. The matcher draws the remaining variables — host galaxy, black hole mass $M_{\rm BH}$ and Eddington ratio $\lambda_{\rm Edd}$ — from their joint distribution *conditioned on* the quasar's luminosity. Because the black hole mass relations and the Eddington ratio distribution are both Gaussian in log space, that conditional distribution has a closed form and is sampled exactly, without any grid or nearest-neighbour search.

## The Matching Algorithm

For a single quasar with redshift $z$ and absolute magnitude $M_i(z=2)$:

1. **Bolometric luminosity.** The magnitude fixes the 3000 Å luminosity through a single zero point, and the Richards et al. (2006) bolometric correction — the one the SDSS quasar catalogs of Shen et al. (2011) and Wu & Shen (2022) use — turns it into $L_{\rm bol}$:

    $$\log_{10} \lambda L_\lambda(3000\,\text{Å}) = 35.27 - 0.4\,M_i(z=2), \qquad L_{\rm bol} = 5.15\,\lambda L_\lambda(3000\,\text{Å}),$$

    with an object-to-object scatter of 0.1 dex (configurable) on the correction, matching the per-object spread in Runnoe, Brotherton & Shang (2012). The zero point is fitted to the H$\beta$/Mg II sample of Wu & Shen (2022; [arXiv:2209.03987](https://arxiv.org/abs/2209.03987)) so that the mock sits in the same $L_{3000}$ system as the virial masses. A pure $\alpha_\nu = -0.5$ continuum in the Richards et al. (2006) $M_i(z=2)$ convention (which gives $M_{1450} = M_i(z=2) + 1.486$, as quoted by Ross et al. 2013) would put it at 35.20; the data sit 0.07 dex higher, of which 0.02 dex is the DR16Q continuum being redder than $\alpha_\nu = -0.5$ between 2500 and 3000 Å. Runnoe et al. give 5.2 ± 0.2 for the correction and recommend a further factor 0.75 for disc anisotropy; neither the SDSS catalogs nor this code apply it.

2. **Select candidate hosts.** Galaxies within a redshift slice $z \pm \Delta z$ of the quasar. The slice is widened geometrically until it holds at least `min_candidates` galaxies, so sparsely populated redshifts are still sampled. A uniform prior over the candidates weights hosts by their number density.

3. **Black hole mass of each candidate.** Each galaxy gets a *distribution* of black hole masses — a mean from the relation appropriate to its `galaxy_type`, and the intrinsic scatter of that relation:

    | `galaxy_type` | relation | scatter | reference |
    |---|---|---|---|
    | `red` | $M_{\rm BH}/10^9 M_\odot = 0.309\,(\sigma_e/200\,\text{km s}^{-1})^{4.38}$ | 0.29 dex | Kormendy & Ho (2013), eq. 7; [arXiv:1304.7762](https://arxiv.org/abs/1304.7762) |
    | `blue` | $\log_{10} M_{\rm BH} = 7.45 + 1.05\,\log_{10}(M_\star/10^{11} M_\odot)$ | 0.24 dex | Reines & Volonteri (2015), eq. 5; [arXiv:1508.06274](https://arxiv.org/abs/1508.06274) |

    M–$\sigma$ is calibrated on bulges, so it is used only for red galaxies; disc galaxies use the total-stellar-mass relation, whose normalisation sits more than a dex lower. No morphology cut is applied — both types are available as hosts at every luminosity.

4. **Eddington ratio distribution.** A Gaussian in $x = \log_{10}\lambda_{\rm Edd}$ with mean $\mu = -1.15$ and width $\sigma_\lambda = 0.30$ dex, fitted so that the mock reproduces Wu & Shen (2022) at fixed $L_{\rm bol}$ once the flux limit is applied (the observed median, $-0.91$, is biased upward by the flux limit and broadened by the 0.4 dex virial-mass errors). For comparison, the intrinsic type-1 ERDF of Schulze et al. (2015; [arXiv:1412.0754](https://arxiv.org/abs/1412.0754)) at $1.1 < z < 2.1$ is a Schechter function with $\log\lambda_* = -1.19$, $\alpha_\lambda = -0.29$, whose mean and width in $\log\lambda$ are $-1.10$ to $-1.0$ (for $10^8$–$10^9 M_\odot$) and 0.39 dex. A power law bounded to $0.1 \le \lambda \le 1$, as in Korytov et al. (2019), describes active and inactive galaxies alike and sits 0.24 dex too high for objects already drawn from a quasar luminosity function.

5. **Draw the host and the Eddington ratio.** Let $c = \log_{10} L_{\rm bol} - \log_{10} L_{\rm Edd,1}$, where $L_{\rm Edd,1} = 1.257\times10^{38}\,\text{erg s}^{-1}$ is the Eddington luminosity per solar mass, and let $m_k$, $s_k$ be the mean and scatter of candidate $k$'s log black hole mass. Marginalising over $x$ gives the host probabilities

    $$p(k \mid c) \propto \frac{1}{\sqrt{s_k^2+\sigma_\lambda^2}}\exp\!\left[-\frac{(c-m_k-\mu)^2}{2(s_k^2+\sigma_\lambda^2)}\right].$$

    After drawing a host, $x$ is drawn from the conditional Gaussian with

    $$V_k=\left(\sigma_\lambda^{-2}+s_k^{-2}\right)^{-1}, \qquad \bar{x}_k=V_k\left[\frac{\mu}{\sigma_\lambda^2}+\frac{c-m_k}{s_k^2}\right],$$

    and the black hole mass is $\log_{10} M_{\rm BH} = c - x$, so that $L_{\rm bol} = \lambda_{\rm Edd}\,L_{\rm Edd}(M_{\rm BH})$ holds exactly for every object. Because the scatter enters as a weight rather than a cut, the assigned masses scatter about the mean relations with their measured dispersion instead of lying on them.

6. **Rejection.** A quasar is dropped if every candidate is more than `max_offset_sigma` combined standard deviations, $\sqrt{s_k^2+\sigma_\lambda^2}$, away from producing it. The dropped quasars are listed in `rejected_indices` and summarised in a warning. On a 2 deg² SkyPy catalog nothing is rejected between $M_i = -27$ and $-18$.

This is repeated for every quasar, giving a catalog in which each quasar is paired with a host galaxy and carries its own `black_hole_mass_exponent`, `eddington_ratio` and `log_bolometric_luminosity`.

## Validation

Compared with the H$\beta$/Mg II subsample of Wu & Shen (2022) over $0.7 < z < 2$ and $i < 20.5$, medians of mock minus DR16Q:

| $\log M_{\rm BH}$ | $\log \lambda_{\rm Edd}$ | $\log L_{\rm bol}$ | $M_i$ |
|---|---|---|---|
| −0.02 | −0.06 | −0.05 | +0.14 |

The $L_{3000}$ zero point and the Eddington ratio distribution are fitted to this catalog, so those two agreements are by construction. The spread of $\log M_{\rm BH}$ at fixed $L_{\rm bol}$ (0.53–0.59 dex between the 16th and 84th percentiles, against an observed 0.50–0.69) and the broad-line FWHM floor are not fitted and are reproduced. The residual is a luminosity-dependent tilt of $\pm 0.2$ dex in $\log M_{\rm BH}$: the SkyPy velocity dispersions are morphology-blind and run low, so the host mass function runs out before the quasars do and luminous quasars are furnished from the extreme tail of the available hosts.

## Implementation in `quasar_host_match.py`

* **`log_black_hole_mass(galaxy_type, vel_disp, stellar_mass)`**: mean log black hole mass and intrinsic scatter of each galaxy, choosing the relation from its type (step 3).
* **`log_bolometric_luminosity(m_i, scatter, rng)`**: $M_i(z=2) \rightarrow \log_{10} L_{\rm bol}$ (step 1).
* **`QuasarHostMatch` class**: its `match()` method runs steps 2–6 for every quasar. Pass an `rng` for a reproducible catalog and `progress=False` to silence the progress bar. `QuasarRate(host_match_kwargs=...)` forwards keyword arguments here, and seeds `rng` from the `seed` of `quasar_sample` unless one is given.

Note that `QuasarRate` now uses the Richards et al. (2006) K-correction as tabulated, normalised to $z = 2$ like the luminosity function; the earlier subtraction of its $z = 0$ value shifted every $M_i$ by 0.6 mag.

The final output is an `astropy` table containing the original quasar information merged with the properties of its assigned host galaxy.

## References

* Kormendy & Ho (2013), ARA&A 51, 511, [arXiv:1304.7762](https://arxiv.org/abs/1304.7762)
* Korytov et al. (2019), ApJS 245, 26, [arXiv:1907.06530](https://arxiv.org/abs/1907.06530)
* Oguri & Marshall (2010), MNRAS 405, 2579, [arXiv:1001.2037](https://arxiv.org/abs/1001.2037)
* Reines & Volonteri (2015), ApJ 813, 82, [arXiv:1508.06274](https://arxiv.org/abs/1508.06274)
* Richards et al. (2006), AJ 131, 2766, [arXiv:astro-ph/0601434](https://arxiv.org/abs/astro-ph/0601434) — luminosity function, K-correction and the $M_i(z=2)$ system
* Richards et al. (2006), ApJS 166, 470, [arXiv:astro-ph/0601558](https://arxiv.org/abs/astro-ph/0601558) — bolometric corrections
* Ross et al. (2013), ApJ 773, 14, [arXiv:1210.6389](https://arxiv.org/abs/1210.6389)
* Runnoe, Brotherton & Shang (2012), MNRAS 422, 478, [arXiv:1201.5155](https://arxiv.org/abs/1201.5155)
* Schulze et al. (2015), MNRAS 447, 2085, [arXiv:1412.0754](https://arxiv.org/abs/1412.0754)
* Shen et al. (2011), ApJS 194, 45, [arXiv:1006.5178](https://arxiv.org/abs/1006.5178)
* Wu & Shen (2022), ApJS 263, 42, [arXiv:2209.03987](https://arxiv.org/abs/2209.03987)
