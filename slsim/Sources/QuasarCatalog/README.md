# Quasar Host Matching

Here we describe the methodology to match Active Galactic Nuclei (AGN) or quasars with their most likely host galaxies from a large galaxy catalog generated with SkyPy pipeline.

The main idea is to use established physical relationships to determine which galaxy could plausibly host a quasar of a given luminosity at a specific redshift. The algorithm essentially works backwards: for a set of candidate galaxies, it calculates the properties of the quasar each one *could* host and matches it to the target quasar from an input catalog.

## The Matching Algorithm

The matching process is driven by the physical connection between a host galaxy's properties (specifically its velocity dispersion) and the potential luminosity of the supermassive black hole (SMBH) at its center.

The overall workflow for matching a single quasar is as follows:

1.  **Select Candidate Hosts**: For a given quasar at redshift `z`, we select a pool of potential host galaxies from the galaxy catalog that are within a small redshift slice (`z ± Δz`) around the quasar's redshift.

2.  **Estimate Black Hole Mass ($M_{BH}$)**: For each candidate galaxy, we estimate the mass of its central SMBH. This is done using the well-known M-$\sigma$ relation, which links the black hole's mass to the stellar velocity dispersion ($\sigma_e$) of the galaxy's bulge (Kormendy & Ho 2013; [arXiv:1304.7762](https://arxiv.org/abs/1304.7762)). The specific relation used is:

    ```math
    \frac{M_{\text{BH}}}{10^9 M_{\odot}} = 0.310_{-0.033}^{+0.037} \left( \frac{\sigma_e}{200 \text{km.s}^{-1}} \right)^{4.38 \pm 0.29}
    ```

    This calculation is performed by the `black_hole_mass_from_vel_disp()` function.

3.  **Model Quasar Accretion (Eddington Ratio)**: A quasar's luminosity is powered by matter accreting onto the SMBH. The efficiency of this process is described by the Eddington Ratio, $\lambda_{\text{edd}} = L_{\text{bol}} / L_{\text{edd}}$, where $L_{\text{bol}}$ is the bolometric (total) luminosity and $L_{\text{edd}}$ is the theoretical maximum luminosity (the Eddington Luminosity).

    Instead of assuming a single value, we draw a random $\lambda_{\text{edd}}$ for each candidate galaxy from a physically-motivated probability distribution that evolves with redshift (Korytov et al. 2019; [arXiv:1907.06530](https://arxiv.org/abs/1907.06530)). This acknowledges the observed diversity in quasar accretion rates. The probability distribution is modeled as:

    ```math
    P(\lambda_{\text{edd}}|z) = A \frac{1+z}{(1+z_0)^{\gamma_z}} \lambda_{\text{edd}}^{\gamma_e}
    ```

    This sampling is implemented in the `sample_eddington_rate()` function.

4.  **Calculate Predicted Quasar Magnitude**: With the black hole mass ($M_{BH}$) and a sampled Eddington ratio ($\lambda_{\text{edd}}$), we can calculate the predicted absolute magnitude of the potential quasar in a specific photometric band (e.g., LSST 'i' band). This involves a few steps, all handled within the `calculate_lsst_magnitude()` function:
    * Calculate Eddington Luminosity: $L_{\text{edd}} \propto M_{BH}$.
    * Calculate Bolometric Luminosity: $L_{\text{bol}} = \lambda_{\text{edd}} \times L_{\text{edd}}$.
    * Convert to Bolometric Magnitude: $M_{\text{bol}}$.
    * Apply a **Bolometric Correction (BC)** to convert $M_{\text{bol}}$ to the magnitude in the desired band, $M_i$ (Runnoe, Brotherton, & Shang 2012; [arXiv:1201.5155](https://arxiv.org/abs/1201.5155)).

5.  **Find the Best Match**: After calculating a predicted i-band absolute magnitude ($M_{i, \text{predicted}}$) for every candidate galaxy, the algorithm compares these values to the actual magnitude of the target quasar ($M_{i, \text{target}}$). The galaxy that yields the predicted magnitude closest to the target magnitude is selected as the best-fit host.

    ```math
    \text{Select galaxy that minimizes } |M_{i, \text{predicted}} - M_{i, \text{target}}|
    ```

This process is repeated for every quasar in the input catalog, resulting in a final catalog where each quasar is paired with a physically plausible host galaxy.

## Implementation in `quasar_host_match.py`

The physical model described above is implemented in the provided Python script.

* **`black_hole_mass_from_vel_disp(sigma_e)`**: Implements the M-$\sigma$ relation (Equation 1) to calculate $M_{BH}$.
* **`sample_eddington_rate(z, ...)`**: Implements inverse transform sampling to draw $\lambda_{\text{edd}}$ values from the redshift-dependent probability distribution (Equation 2).
* **`calculate_lsst_magnitude(...)`**: Converts the physical properties ($M_{BH}$, $\lambda_{\text{edd}}$) into an observable absolute magnitude in a given LSST band.
* **`QuasarHostMatch` class**: This class orchestrates the entire workflow. Its `match()` method iterates through the input quasar catalog, performs the candidate selection, calculates predicted magnitudes, and identifies the best-matching host galaxy for each quasar.

The final output is an `astropy` table containing the original quasar information merged with the properties of its newly assigned host galaxy.