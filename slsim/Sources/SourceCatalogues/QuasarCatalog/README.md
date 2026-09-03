# Quasar Host Matching

This describes how quasars drawn from a luminosity function are assigned host galaxies from a large galaxy catalog (e.g. one generated with the SkyPy pipeline), together with a black hole mass and an Eddington ratio.

The quasar catalog already fixes *how many* quasars there are and *how bright* they are, because it is sampled from the Oguri & Marshall (2010) / Richards et al. (2006) luminosity function. The job of the matcher is therefore to draw the remaining latent variables — the host galaxy, the black hole mass and the Eddington ratio — from their joint distribution *conditioned on* the quasar's luminosity.

## The generative model

For a quasar of absolute magnitude $M_i(z=2)$ at redshift $z$:

1. **Bolometric luminosity.** The absolute magnitude is converted to the monochromatic luminosity $\lambda L_\lambda(3000\,\text{Å})$ by rescaling the same qsogen SED (Temple et al. 2021) that generates the broad-band photometry, so the two are exactly consistent. Because the $i$ band at $z=2$ samples the rest-frame continuum at $\sim2500\,\text{Å}$, free of strong lines and host light, this relation is a pure $0.4\,$dex-per-magnitude rescaling and needs only a single reference SED evaluation. The bolometric luminosity then follows from the Runnoe et al. (2012) correction, $L_{\rm bol} = \zeta_{3000}\,\lambda L_\lambda(3000)$ with $\zeta_{3000} = 5.18$, with a configurable object-to-object scatter (default 0.1 dex).

2. **Candidate hosts.** Galaxies within a thin redshift slice $z \pm \Delta z$ of the quasar. The slice is widened geometrically until it holds at least `min_candidates` galaxies, so the sampling does not degrade in sparsely populated redshift ranges.

3. **M–σ relation.** Each candidate has a mean black hole mass from Kormendy & Ho (2013), $M_{\rm BH}/10^9 M_\odot = 0.310\,(\sigma_e/200\,\text{km s}^{-1})^{4.38}$, with an intrinsic scatter of $0.29$ dex. Note this relation is calibrated on ellipticals and classical bulges and $\sigma_e$ is the *bulge* dispersion; set `galaxy_types=["red"]` to restrict the hosts to bulge-dominated galaxies.

4. **Eddington ratio.** The power-law distribution $p(\lambda) \propto \lambda^{\gamma_e}$ with $\gamma_e = -0.65$ from Korytov et al. (2019). Their Eq. (16) carries a $(1+z)/(1+z_0)^{\gamma_z}$ prefactor, but that sets the *fraction of galaxies that are active* rather than the shape of the distribution, and it cancels on normalisation. The redshift dependence of the quasar abundance is already carried by the luminosity function, so nothing is lost.

    The default range is $0.01 \le \lambda \le 1$ rather than the $0.1 \le \lambda \le 1$ of Korytov et al. The lower bound is roughly where a radiatively efficient thin disc gives way to an advection dominated flow, and it matches the lower edge of `agn_bounds_dict` in the variability model. It has to be this low because the luminosity function is sampled far below its knee: an $M_i = -19$ quasar needs a $10^6\,M_\odot$ black hole even at $\lambda = 0.1$, and no galaxy in a typical SkyPy catalog is that small, so every such quasar would otherwise be rejected.

5. **The draw.** A (host $k$, Eddington ratio $\lambda$) pair is sampled from

    $$p(k, \lambda) \propto p(\lambda)\;\mathcal{N}\!\left(\log M_{\rm BH}^{\rm req}(\lambda)\;\middle|\;\log M_{\rm BH}(\sigma_k),\; 0.29\right),$$

    where $M_{\rm BH}^{\rm req}(\lambda) = L_{\rm bol} / (\lambda\, L_{\rm Edd,1})$ is the mass that reproduces the observed luminosity at that Eddington ratio. Candidates enter with a uniform prior, which correctly weights by the galaxy number density in the slice. The reported black hole mass is $M_{\rm BH}^{\rm req}$, so the catalog satisfies $L_{\rm bol} = \lambda\, L_{\rm Edd}(M_{\rm BH})$ exactly, while the $(M_{\rm BH}, \sigma)$ pairs scatter about the M–σ relation with its measured dispersion.

6. **Rejection.** A quasar whose luminosity no candidate host can produce within `max_offset_sigma` times the M–σ scatter is dropped, and the number dropped is reported in `n_rejected`. This is a genuine physical statement — there may be no galaxy massive enough at that redshift — and is preferable to silently assigning an implausible host.

Setting `unique_hosts=True` prevents a galaxy from being assigned to more than one quasar.

## Recommended settings, and why the defaults are not enough

Validated against the SDSS DR7 quasar property catalog of Shen et al. (2011), in the magnitude range where the two overlap ($-26 < M_i < -24$):

| host pool | Eddington ratio distribution | median $\log M_{\rm BH}$ | median $\log \lambda$ |
|---|---|---|---|
| all galaxies | power law | 7.90 | −0.30 |
| all galaxies | lognormal | 8.17 | −0.59 |
| red only | power law | 8.20 | −0.62 |
| **red only** | **lognormal** | **8.38** | **−0.80** |
| Shen et al. (2011) | — | 8.83 | −0.86 |

With the defaults the black hole masses come out $\sim0.9$ dex too low and the Eddington ratios $\sim0.5$ dex too high. Two things cause this.

First, the Korytov et al. power law with $\gamma_e = -0.65$ puts most of its probability *mass* near the Eddington limit, because the mass per unit $\log\lambda$ goes as $\lambda^{\gamma_e+1}$, which rises. Broad-line quasars are observed near $\lambda \sim 0.1$.

Second, and more importantly, **the AGN active fraction here does not depend on host mass**. A SkyPy catalog is overwhelmingly dwarf galaxies — the median velocity dispersion is 30–40 km/s at every redshift, and galaxies above 240 km/s are well under 1% of the catalog. Since candidates enter with a uniform prior (which is the correct *number density* weighting), the draw lands on a numerous $\sigma \approx 150\,$km/s host at high $\lambda$ rather than a rare $\sigma \approx 240\,$km/s host at low $\lambda$. In reality luminous quasars preferentially inhabit massive galaxies; cosmoDC2 encodes this with conditional abundance matching on specific star formation rate, and there is no equivalent term here. Restricting the candidates to red, bulge-dominated galaxies is a coarse stand-in that also happens to be where the Kormendy & Ho M–σ relation is calibrated.

So for science use, pass:

```python
QuasarRate(
    ...,
    host_match_kwargs={
        "galaxy_types": ["red"],
        "eddington_ratio_distribution": "lognormal",
    },
)
```

The residual 0.45 dex in $M_{\rm BH}$ is within the systematic floor of the comparison: single-epoch virial masses carry $\sim0.4$ dex of uncertainty and are biased high at fixed luminosity in a flux-limited sample.

## Why not nearest-neighbour matching?

An earlier version drew an Eddington ratio for every candidate and kept the pair minimising $|M_i - M_{i,\rm pred}|$. That selects the extreme tail of the Eddington ratio distribution rather than sampling from it, and produces a $\sigma$–$L$ relation with no scatter at all, because the best match is by construction the one that lies on the relation. The weighted draw above is the same physics without those artefacts.

## Implementation

`quasar_host_match.py`:

* `l3000_from_absolute_i_magnitude` / `absolute_i_magnitude_from_l3000` — qsogen-anchored luminosity–magnitude conversion.
* `bolometric_luminosity_from_l3000` — Runnoe et al. (2012) bolometric correction, linear or log-log form, with optional anisotropy correction and scatter.
* `black_hole_mass_from_vel_disp` — the M–σ relation, with optional intrinsic scatter.
* `sample_eddington_rate` / `eddington_ratio_grid` — the Eddington ratio distribution.
* `calculate_lsst_magnitude` — a coarse grey-bolometric-correction estimate of the absolute AB magnitude in an LSST band, kept for convenience; the matcher uses the qsogen route instead.
* `QuasarHostMatch` — the matching class described above.

## References

* Kormendy & Ho (2013), [arXiv:1304.7762](https://arxiv.org/abs/1304.7762)
* Korytov et al. (2019), [arXiv:1907.06530](https://arxiv.org/abs/1907.06530)
* Oguri & Marshall (2010), [arXiv:1001.2037](https://arxiv.org/abs/1001.2037)
* Richards et al. (2006), [arXiv:astro-ph/0601434](https://arxiv.org/abs/astro-ph/0601434)
* Runnoe, Brotherton & Shang (2012), [arXiv:1201.5155](https://arxiv.org/abs/1201.5155), and its erratum
* Temple, Hewett & Banerji (2021), [arXiv:2109.04472](https://arxiv.org/abs/2109.04472)
