# Quasar Host Matching

This describes how quasars drawn from a luminosity function are assigned host galaxies from a large galaxy catalog (e.g. one generated with the SkyPy pipeline), together with a black hole mass and an Eddington ratio.

The quasar catalog already fixes *how many* quasars there are and *how bright* they are, because it is sampled from the Oguri & Marshall (2010) / Richards et al. (2006) luminosity function. The job of the matcher is therefore to draw the remaining latent variables — the host galaxy, the black hole mass and the Eddington ratio — from their joint distribution *conditioned on* the quasar's luminosity.

## The model

A galaxy does not have a black hole mass; it has a *distribution* of them, because the black hole mass relations have real intrinsic scatter. That distribution is the only thing tying a host to a quasar. Writing it out, the joint posterior for a host $k$ and an Eddington ratio $\lambda$ is

$$p(k, \lambda) \propto w_k\; p(\lambda)\; \mathcal{N}\!\left(\log M_{\rm BH}^{\rm req}(\lambda)\;\middle|\;\log M_{\rm BH}(k),\; s_k\right),$$

where $s_k$ is the intrinsic scatter of the relation appropriate to galaxy $k$, $w_k$ is the prior over hosts, and

$$M_{\rm BH}^{\rm req}(\lambda) = \frac{L_{\rm bol}}{\lambda\, L_{\rm Edd,1}}$$

is the mass that reproduces the observed luminosity at that Eddington ratio. $L_{\rm Edd,1} = 1.257\times10^{38}\,\text{erg s}^{-1}M_\odot^{-1}$ is the Eddington luminosity **per solar mass**, a constant — not the Eddington luminosity of the black hole, which would make this circular. Since $L_{\rm Edd}$ is strictly linear in mass, $L_{\rm Edd}(M) = M\,L_{\rm Edd,1}$, the definition $\lambda \equiv L_{\rm bol}/L_{\rm Edd}(M_{\rm BH})$ rearranges to the line above, and an erg/s divided by an erg/s per solar mass is a mass.

So the constraint $L_{\rm bol} = \lambda\, M_{\rm BH}\, L_{\rm Edd,1}$ is exact, which is what collapses the black hole mass out of the problem and leaves the Gaussian evaluated at $M_{\rm BH}^{\rm req}$.

## Ingredients

1. **Bolometric luminosity.** The absolute magnitude fixes the monochromatic luminosity through a single calibrated zero point,

    $$\log_{10} \lambda L_\lambda(3000\,\text{Å}) = 35.27 - 0.4\,M_i(z=2),$$

    the slope following from both sides being log luminosities. The zero point is fitted to the H$\beta$/Mg II subsample of Wu & Shen (2022), whose $M_i(z=2)$ comes from SDSS photometry through the same Richards et al. (2006) K-correction the luminosity function uses, so both sides sit in one magnitude system. It holds to 0.02 dex from $M_i = -28$ to $-23$ and 0.05 dex from $z = 0.7$ to $2.5$. The bolometric luminosity then follows from the Runnoe et al. (2012) correction, $L_{\rm bol} = \zeta_{3000}\,\lambda L_\lambda(3000)$ with $\zeta_{3000} = 5.18$, with a configurable object-to-object scatter (default 0.1 dex). Wu & Shen use 5.15 from Richards et al. (2006), which is the same number to 0.003 dex.

    Earlier versions read the 3000 Å flux off a qsogen SED (Temple et al. 2021) anchored to $M_i$, which runs 0.18 dex faint. Roughly half of that was bookkeeping — the reported luminosity was the nominal `LogL3000` normalisation, while the spectrum whose synthetic photometry defined $M_i$ carried 0.095 dex more flux at 3000 Å — and the rest is spectral shape: qsogen's 2500/3000 colour is 0.04 dex bluer than the DR16Q continuum, and its 5100/3000 colour is off by 0.33 dex. The SED is still what generates the broad-band photometry; it is no longer what measures the luminosity.

2. **Candidate hosts.** Galaxies within a thin redshift slice $z \pm \Delta z$ of the quasar. The slice is widened geometrically until it holds at least `min_candidates` galaxies, so the sampling does not degrade in sparsely populated redshift ranges.

3. **Black hole mass relations.** Each relation is applied only where it is calibrated, selected per galaxy by its `galaxy_type`:

    | type | relation | scatter | reference |
    |---|---|---|---|
    | `red` | $M_{\rm BH}/10^9 M_\odot = 0.309\,(\sigma_e/200\,\text{km s}^{-1})^{4.38}$ | 0.29 dex | Kormendy & Ho (2013), eq. 7 |
    | `blue` | $\log M_{\rm BH} = 7.45 + 1.05\,\log(M_\star/10^{11} M_\odot)$ | 0.24 dex | Reines & Volonteri (2015), eq. 5 |

    M–σ is calibrated on ellipticals and classical bulges, and $\sigma_e$ is a *bulge* dispersion. A disc galaxy's dispersion is not one — in a SkyPy catalog it comes from abundance matching on total stellar mass — so applying M–σ there would be unjustified. The Reines & Volonteri relation is measured against *total* stellar mass, which is exactly what the catalog carries, and its normalisation sits more than a dex below the early-type one.

    This is what keeps every galaxy type available as a host while still preferring bulge-dominated hosts for luminous quasars, without any morphology cut being applied. In practice the preference is now close to a cut: 96% of the hosts of $-26 < M_i < -24$ quasars come out red, because only the M–σ branch reaches the masses the observed Eddington ratios demand. See the limitations below.

4. **Eddington ratio.** A Gaussian in $x=\log_{10}\lambda$, which is close to the shape found for broad-line quasars by Kelly & Shen (2013) and Schulze et al. (2015):

    $$\frac{dP}{d\log_{10}\lambda} \propto \exp\!\left[-\frac{(\log_{10}\lambda - \mu)^2}{2\sigma_\lambda^2}\right], \qquad \mu = -1.15,\; \sigma_\lambda = 0.30\,\text{dex},$$

    Both values are fitted to Wu & Shen (2022) *at fixed bolometric luminosity*, which is the comparison a flux limit barely touches: at fixed $L_{\rm bol}$ and $z$ every quasar has essentially the same apparent magnitude, so the selection cuts on luminosity alone and leaves the mass distribution within a bin almost intact.

    The width is not simply the observed one. DR16Q's spread at fixed $L_{\rm bol}$ is an *upper* bound on the intrinsic spread, so $\sigma_\lambda$ is chosen such that 0.30 dex intrinsic plus the catalog's own 0.12 dex mass uncertainties reproduce what is observed, rather than matching it outright.

    **Why not the Korytov et al. (2019) specific accretion rate distribution**, which this replaced. That form, $dP/d\log_{10}\lambda \propto \lambda^{-0.65}$ over $0.1 \le \lambda \le 1$, follows Aird et al. (2018) and describes *galaxies*, active and inactive alike. Applied to objects already drawn from a quasar luminosity function it counts the same selection twice, and it shows: its median $\log\lambda$ of $-0.67$ sits 0.24 dex above the observed $-0.91$, which propagates directly into black hole masses through $\log M_{\rm BH} = \log L_{\rm bol} - \log\lambda - \log L_{\rm Edd,1}$. Lowering its floor to reach the observed median widens it to 0.81 dex against an observed 0.60, because a bounded power law cannot move its centre without also changing its width — and the data want a shift alone.

5. **The draw.** Define $c=\log L_{\rm bol}-\log L_{\rm Edd,1}$ and let $m_k$ and $s_k$ be the mean and scatter of host $k$'s log black-hole mass. Marginalising over $x$ gives the host weights directly:

    $$p(k\mid c) \propto \frac{1}{\sqrt{s_k^2+\sigma_\lambda^2}}\exp\!\left[-\frac{(c-m_k-\mu)^2}{2(s_k^2+\sigma_\lambda^2)}\right].$$

    After drawing the host, $x$ is drawn from its conditional Gaussian,

    $$V_k=\left(\sigma_\lambda^{-2}+s_k^{-2}\right)^{-1}, \qquad
    \bar{x}_k=V_k\left[\frac{\mu}{\sigma_\lambda^2}+\frac{c-m_k}{s_k^2}\right],$$

    and $\log M_{\rm BH}=c-x$. This samples the stated model exactly, without an Eddington-ratio grid, and preserves $L_{\rm bol}=\lambda L_{\rm Edd}(M_{\rm BH})$ exactly.

    Because the scatter enters as a weight rather than as a cut, the resulting $(M_{\rm BH}, \sigma)$ pairs scatter about the mean relation with its measured dispersion instead of lying on it.

6. **Rejection.** A quasar is dropped when every candidate exceeds the combined standardized discrepancy

    $$d_k=\frac{|c-\mu-m_k|}{\sqrt{\sigma_\lambda^2+s_k^2}}$$

    set by `max_offset_sigma`. `n_rejected`, `rejected_indices` and a warning report how many were dropped and where they lie in magnitude and redshift. Rejection can indicate a genuinely absent massive host, but can also reflect finite catalog area, an incomplete high-mass tail, or an inaccurate velocity-dispersion prescription. On a 2 deg² SkyPy catalog nothing is rejected between $M_i = -27$ and $-18$.

## Validation

Against the H$\beta$/Mg II subsample of the SDSS DR16 quasar property catalog of Wu & Shen (2022), on a 2 deg² SkyPy host catalog. The mock is generated over 300 deg² down to $i = 21$, because a survey covering $10^4$ deg² has no useful overlap with a deep pencil beam. Both sides are restricted to $0.7 < z < 2$ — C IV masses above that carry blueshift-dependent systematics of their own — and to $i < 20.5$, and both are required to satisfy the broad-line criterion, which shows up in the data as a hard floor at FWHM = 1399 km/s.

Medians in that window, mock minus DR16Q:

| | $\log M_{\rm BH}$ | $\log \lambda_{\rm Edd}$ | $\log L_{\rm bol}$ | $M_i$ |
|---|---|---|---|---|
| before (Korytov ERDF, SED luminosity) | −0.64 | +0.38 | −0.22 | +0.12 |
| now | −0.01 | −0.06 | −0.06 | +0.17 |

The Eddington ratio distribution and the $L_{3000}$ zero point are fitted to this catalog, so those are a closed loop rather than an independent test. What is *not* fitted, and does hold:

* a single $L_{3000}$ zero point reproduces $\lambda L_\lambda(3000)$ at fixed $M_i$ to 0.006 dex, with 0.014 dex of spread across five magnitude bins from $M_i = -27$ to $-22$;
* the spread of $\log M_{\rm BH}$ at fixed $L_{\rm bol}$, 0.47–0.61 dex between the 16th and 84th percentiles against an observed 0.50–0.69;
* the broad-line criterion, which now removes *no* mock quasar: every one implies an Mg II FWHM above the observed floor, where the old Eddington ratios put 5.6% of them below it.

The residual is a tilt rather than an offset. At fixed $L_{\rm bol}$ the mass offset runs from −0.20 dex at $\log L_{\rm bol} \simeq 45.1$ to +0.15 dex at 46.6, crossing zero near 45.9. No change to the Eddington ratio distribution can remove it, because it comes from the host mass function tilting the draw — see below.

## Known limitations

* **The host black hole mass function runs out before the quasars do, and this is now the dominant error.** Over $0.7 < z < 2$ the candidates' mean-relation $\log M_{\rm BH}$ has a median of 5.35 and a 99.9th percentile of 9.02. The median mass *assigned* to a flux-limited quasar is 8.83 — the 99.83rd percentile of the candidates. Luminous quasars are therefore furnished almost entirely from the extreme tail of the available hosts, which is what produces both the luminosity-dependent tilt above and the 96% red host fraction: only the M–σ branch reaches those masses at all. The matcher is doing the right thing with the catalog it is given; the catalog is the problem.

* **Velocity dispersions are morphology-blind and run low.** In a SkyPy catalog, red and blue galaxies of the same stellar mass are assigned nearly the same dispersion, and the values fall below the observed $\sigma(M_\star)$ of SDSS early types (Zahid et al. 2016), increasingly so at low mass. This is the proximate cause of the point above. It is shared with the galaxy deflector population, so changing it moves lensing predictions too.

* **The fitted Eddington ratio location inherits the virial mass zero point**, of order 0.1–0.2 dex from the virial factor. The $L_{3000}$ zero point does not depend on that calibration and is the firmer of the two.

* **Below $z \simeq 0.9$ the comparison sample is not clean.** DR16Q's target selection is heterogeneous there, and its $i$ band carries host light the K-correction does not model — the photometric $M_i$ runs 0.4 mag bright of the continuum-derived one at $z < 0.5$. Hence the $z > 0.7$ window.

## Why not nearest-neighbour matching?

An earlier version drew an Eddington ratio for every candidate and kept the pair minimising $|M_i - M_{i,\rm pred}|$. That selects the extreme tail of the Eddington ratio distribution rather than sampling from it, and produces a $\sigma$–$L$ relation with no scatter at all, because the best match is by construction the one that lies on the relation. The weighted draw above is the same physics without those artefacts.

## Implementation

`quasar_host_match.py`:

* `black_hole_mass` — mean black hole mass and intrinsic scatter of each galaxy, choosing the relation from its type.
* `bolometric_luminosity` — $M_i(z=2) \rightarrow L_{\rm bol}$ via the calibrated 3000 Å zero point and the Runnoe et al. (2012) correction.
* `QuasarHostMatch` — the matching class described above. Pass an `rng` for a reproducible catalog.

## References

* Aird, Coil & Georgakakis (2018), MNRAS 474, 1225, [arXiv:1705.01132](https://arxiv.org/abs/1705.01132)
* Kelly & Shen (2013), ApJ 764, 45, [arXiv:1209.0477](https://arxiv.org/abs/1209.0477)
* Kormendy & Ho (2013), ARA&A 51, 511, [arXiv:1304.7762](https://arxiv.org/abs/1304.7762)
* Korytov et al. (2019), ApJS 245, 26, [arXiv:1907.06530](https://arxiv.org/abs/1907.06530)
* Lyke et al. (2020), ApJS 250, 8, [arXiv:2007.09001](https://arxiv.org/abs/2007.09001)
* Oguri & Marshall (2010), MNRAS 405, 2579, [arXiv:1001.2037](https://arxiv.org/abs/1001.2037)
* Reines & Volonteri (2015), ApJ 813, 82, [arXiv:1508.06274](https://arxiv.org/abs/1508.06274)
* Richards et al. (2006), AJ 131, 2766, [arXiv:astro-ph/0601434](https://arxiv.org/abs/astro-ph/0601434) — the luminosity function and the $M_i(z=2)$ system
* Richards et al. (2006), ApJS 166, 470, [arXiv:astro-ph/0601558](https://arxiv.org/abs/astro-ph/0601558) — the spectral energy distributions and bolometric corrections
* Runnoe, Brotherton & Shang (2012), MNRAS 422, 478, [arXiv:1201.5155](https://arxiv.org/abs/1201.5155), and its erratum, MNRAS 427, 1800
* Schulze et al. (2015), MNRAS 447, 2085, [arXiv:1412.0754](https://arxiv.org/abs/1412.0754)
* Shen et al. (2011), ApJS 194, 45, [arXiv:1006.5178](https://arxiv.org/abs/1006.5178)
* Temple, Hewett & Banerji (2021), MNRAS 508, 737, [arXiv:2109.04472](https://arxiv.org/abs/2109.04472)
* Wu & Shen (2022), ApJS 263, 42, [arXiv:2209.03987](https://arxiv.org/abs/2209.03987)
* Zahid et al. (2016), ApJ 832, 203, [arXiv:1607.04275](https://arxiv.org/abs/1607.04275)
