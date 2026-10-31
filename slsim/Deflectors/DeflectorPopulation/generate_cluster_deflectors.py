"""The GeneratedDeflector class implements the analytic subhalo model from Han
et al. (2016). Most other choices are made following Abe et al. (2025).

Model summary:

1. First, the accretion subhalo masses are sampled from the SHMF given in Han et al. (2018)

2. Host halo concentration is set by the Diemer19 mass-concentration relation with log scatter 0.33
The accreted subhalos are placed randomly in the host halo (within R200) tracing the host halo density

3. The evolved subhalo masses are calculated according to eq. 7 in Han 2016, with some subhalos being completely stripped (orphan galaxies)
Subhalo concentration is chosen the same way using the accretion mass
Subhalos use a truncated NFW profile with truncation radius from eq. 6 in Gilman et al. (2016)

4. Galaxy masses are calculated from subhalo accretion mass from the Behroozi SMHM relation, with log scatter 0.2

5. Galaxies are randomly selected as red or blue according to their mass, the halo mass and their distance to the center.
Relations are given from Zheng et al. 2025

6. The closest matching red / blue galaxy in terms of stellar mass and redshift is then selected from the skypy catalog

TODO better truncation radius, eccentricity for subhalos, better relations between satellite galaxies and subhalos,
more accurate subhalo evolution (update concentration parameter or GBNFW. also orphan galaxies may be a numerical artifact
from DM simulations), more realistic satellite distribution (observations show higher concentrations towards cluster center)

References:
Han et al. (2016): https://academic.oup.com/mnras/article/457/2/1208/965286
Han et al. (2018): https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract
Abe et al. (2025): https://arxiv.org/abs/2411.07509
Gilman et al. (2019): https://arxiv.org/abs/1908.06983
Zheng et al. (2025): https://arxiv.org/pdf/2501.00986
"""

import random

import numpy as np
from colossus.halo import concentration, profile_nfw
from scipy.special import erf
from scipy.stats import truncnorm

from slsim.Deflectors.deflector_group import DeflectorGroup
from slsim.Deflectors.deflector_util import set_colossus_cosmo
from slsim.Deflectors.MassLightConnection.galaxy_population import (
    gals_init,
    stellarmass_halomass,
)
from slsim.Halos.halo_population import dNhalodzdlnM_lens
from slsim.Sources.SourcePopulation.galaxies import galaxy_projected_eccentricity
from slsim.Util.param_util import e2epsilon, ellip_from_axis_ratio2epsilon


class GeneratedDeflector:
    def __init__(self, M200h, z, cosmo, verbose=False):
        """
        :param M200h: host halo mass (200c) in solar masses / h
        :param z: deflector redshift
        :param cosmo: astropy cosmology instance
        :param verbose: verbose
        """

        self.verbose = verbose

        self.cosmo_col = set_colossus_cosmo(cosmo)

        c = concentration.concentration(M200h, "200c", z, model="diemer19")
        c = np.random.lognormal(
            mean=np.log(c), sigma=0.33
        )  # concentration distribution 0.16 dex
        self.c = c
        self.p_nfw = profile_nfw.NFWProfile(M=M200h, c=c, z=z, mdef="200c")

        self.M200h = M200h  # solar masses / h
        self.R200h = self.p_nfw.RDelta(z, "200c")  # kpc / h
        self.Rvir = self.p_nfw.RDelta(z, "vir")  # kpc / h

        self.z = z
        self.cosmo = cosmo
        self.angular_diameter_dist = cosmo.angular_diameter_distance(self.z).to_value("kpc")  # distance corresponding to 1 radian

        self.mean_position_angle = np.random.rand() * 2 * np.pi

        mean_ellipticity_host = max( #mass physical
            0.23, 0.094 * np.log10(self.M200h / self.cosmo.h) - 0.95
        )
        self.ellipticity_host = truncnorm.rvs(  # 1 - q
            (0.0 - mean_ellipticity_host) / 0.13,
            (0.9 - mean_ellipticity_host) / 0.13,
            loc=mean_ellipticity_host,
            scale=0.13,
        )

        if self.verbose:
            print(f"M200: {np.log10(self.M200h):.4g}/h, z: {self.z:.4g}")
            print(f"R200: {self.R200h:.4g} kpc/h, projected {self.R200h / self.cosmo.h / self.angular_diameter_dist * 206265:.4g} arcsec")
            print(f"host concentration: {self.c:.4g}")
            print(f"host ellipticity: {1 - self.ellipticity_host:.4g} (q)")

        self._halos = None

    def _accreted_subhalo_density_pdf(self, m_acc_h):
        """Returns accreted subhalo dN / dlnm.

        :param m_acc_h: subhalo accretion mass [solar masses / h]
        """

        # Han et al. 2018, table 1
        a1 = 0.11
        al1 = 0.95
        a2 = 0.20
        al2 = 0.30
        b = 7.6
        beta = 2.1

        mu = m_acc_h / self.M200h

        return (a1 * mu**-al1 + a2 * mu**-al2) * np.exp(-b * mu**beta)

    def _unevolved_spatial_distribution_pdf(self, r):
        """Returns normalized dN / d3r as a function of radius, so the total
        probability over the host halo is 1.

        In the model, when selecting by accretion mass the final position traces host halo density

        :param r: distance to halo center [kpc / h]
        """

        return self.p_nfw.density(r) / self.M200h

    def _evolved_subhalo_mass_fraction_of_accretion(self, r):
        """Returns m_evolved / m_acc.

        :param r: distance to halo center [kpc / h]

        Han 2016
        """

        M = self.M200h / 10**10

        mustar = 0.5 * M**-0.03
        beta = 1.7 * M**-0.04
        sigma = 1.1
        fs = 0.55

        mubar = mustar * (r / self.R200h) ** beta
        mu = np.random.lognormal(mean=np.log(mubar), sigma=sigma)

        return np.where(np.random.rand(*r.shape) < fs, mu, 0)  # 0 mass when stripped

    def _generate_subhalos(self, m_acc_min, m_acc_max):
        """Randomly samples subhalos from m_acc_min to m_acc_max (solar masses
        / h)

        Returns array [(accretion mass [solar masses / h], evolved mass [solar masses / h], distance to center [kpc / h])]
        """

        N_MASS = 1000
        N_RADIUS = 1000

        # Mass bins
        m_acc = np.logspace(np.log10(m_acc_min), np.log10(m_acc_max), num=N_MASS)

        # Radial bin edges
        r = self.R200h * np.linspace(0.001, 1, N_RADIUS)  # TODO splashback radius

        # Radial shell volumes
        d3r = 4 / 3 * np.pi * (r[1:] ** 3 - r[:-1] ** 3)

        # Spatial contribution for each radial bin
        spatial_distribution_multiplier = (
            self._unevolved_spatial_distribution_pdf(r[:-1]) * d3r
        )

        # Expected number for each (logarithmic) mass bin
        dlnm = np.log(m_acc[1]) - np.log(m_acc[0])
        accreted_subhalo_expected = self._accreted_subhalo_density_pdf(m_acc) * dlnm

        # Full (mass, radius) probability matrix, shape (N_MASS, N_RADIUS - 1)
        probability_present = (
            accreted_subhalo_expected[:, None] * spatial_distribution_multiplier[None, :]
        )

        # Random sampling
        rng = np.random.default_rng()
        mask = rng.random(probability_present.shape) < probability_present

        # Indices of sampled subhalos
        mass_indices, radius_indices = np.where(mask)

        mass_acc = m_acc[mass_indices]

        # Randomly interpolate radius within shell
        interpolation_factor = rng.random(len(radius_indices))
        radius = r[radius_indices] * interpolation_factor + r[radius_indices + 1] * (
            1 - interpolation_factor
        )

        mass_evolved = mass_acc * self._evolved_subhalo_mass_fraction_of_accretion(
            r[radius_indices]
        )

        # Shape (N_subhalos, 3)
        return np.column_stack((mass_acc, mass_evolved, radius))

    def _generate_deflector_population(
        self,
        min_subhalo_accretion_mass=None,
        crop_subhalo_dist=100,
    ):
        """Generates the host halo and its subhalos / galaxies

        :param min_subhalo_accretion_mass: min subhalo accretion mass [solar masses / h]
        :param crop_subhalo_dist: only return subhalos closer than this distance to the BCG [arcsecs]

        Returns a list of halo info

        [{
            "subhalo_m_evolved": subhalo_m_evolved,
            "dist_to_center": dist_to_center,
            "concentration": concentrations[i],
            "pos_2d": pos_2d,
            "galaxy_mass": galaxy_mass,
            "red": red,
            "light_position_angle": light_position_angle,
        }]
        """

        if min_subhalo_accretion_mass is None:
            min_subhalo_accretion_mass = 10**-3 * self.M200h

        paramc, params = gals_init()
        
        # central halo first, then the sampled subhalos
        halos = np.vstack(
            (
                np.array([[self.M200h, self.M200h, 0]]),
                self._generate_subhalos(min_subhalo_accretion_mass, self.M200h),
            )
        )

        m_accs = halos[:, 0]
        m_evos = halos[:, 1]

        concentrations = np.zeros_like(m_evos)

        c = concentration.concentration(
            m_accs[m_evos > 0], "200c", self.z, model="diemer19"
        )
        c = np.random.lognormal(
            mean=np.log(c), sigma=0.33
        )  # concentration distribution 0.16 dex
        concentrations[m_evos > 0] = c

        concentrations[0] = self.c #host

        halo_data = []

        for i, (subhalo_m_acc, subhalo_m_evolved, dist_to_center) in enumerate(halos):
            # Mo/h, Mo/h, kpc/h

            pos = np.random.normal(size=3)
            pos /= np.linalg.norm(pos)
            pos_2d = (
                pos[:2] * dist_to_center / self.cosmo.h / self.angular_diameter_dist * 206265
            )  # 2d coordinate in arcsecs

            pos_2d[0] *= np.sqrt(1 - self.ellipticity_host)
            pos_2d[1] /= np.sqrt(1 - self.ellipticity_host)

            pos_2d = [  # not sure why this has to be negative
                pos_2d[0] * np.cos(-self.mean_position_angle)
                - pos_2d[1] * np.sin(-self.mean_position_angle),
                pos_2d[0] * np.sin(-self.mean_position_angle)
                + pos_2d[1] * np.cos(-self.mean_position_angle),
            ]

            if not (
                (-crop_subhalo_dist < pos_2d[0] < crop_subhalo_dist)
                and (-crop_subhalo_dist < pos_2d[1] < crop_subhalo_dist)
            ):
                continue

            # compute stellar mass from halo mass, use paramc if host halo, else params for SMHM relation
            galaxy_mass = stellarmass_halomass(
                subhalo_m_acc, self.z, paramc if dist_to_center == 0 else params
            )  # solar masses / h
            galaxy_mass = np.random.lognormal(
                mean=np.log(galaxy_mass), sigma=0.2
            )  # scatter

            # determine if galaxy is red or blue, different for central and satellite
            if dist_to_center > 0:
                red = (
                    self._fraction_red_galaxies(dist_to_center, galaxy_mass)
                    > np.random.rand()
                )
            else:
                red = self._fraction_central_red(galaxy_mass) > np.random.rand()

            # random position angle of the galaxy light, used for its eccentricity
            light_position_angle = np.random.normal(
                loc=self.mean_position_angle, scale=35.4 * np.pi / 180
            )

            halo_data.append(
                {
                    "subhalo_m_evolved": subhalo_m_evolved,
                    "dist_to_center": dist_to_center,
                    "concentration": concentrations[i],
                    "pos_2d": pos_2d,
                    "galaxy_mass": galaxy_mass,
                    "red": red,
                    "light_position_angle": light_position_angle,
                }
            )

        return halo_data

    def get_deflector_data(
        self,
        red_galaxies,
        blue_galaxies,
        min_subhalo_accretion_mass=None,
        crop_subhalo_dist=100,
        #subhalo_profile="TNFW",  # TNFW, GBNFW
        kwargs_host_mass=None,
    ):
        """Generate deflector mass and light dicts

        :param red_galaxies: catalog of red galaxies to match to
        :param blue_galaxies: catalog of blue galaxies to match to
        :param min_subhalo_accretion_mass: min subhalo accretion mass [solar masses / h]
        :param crop_subhalo_dist: only return subhalos closer than this distance to the BCG [arcsecs]
        
        To make the host halo GNFW, specify kwargs_host_mass={"gamma_in": gamma}

        returns dict with kwargs_mass_list, kwargs_light_list, center_x_deflector_list, center_y_deflector_list to be passed into DeflectorGroup
        """

        if self._halos is None:
            self._halos = self._generate_deflector_population(
                min_subhalo_accretion_mass=min_subhalo_accretion_mass,
                crop_subhalo_dist=crop_subhalo_dist,
            )

        halo_data = self._halos
        ellipticity_host = self.ellipticity_host
        mean_position_angle = self.mean_position_angle

        light_dicts = []
        mass_dicts = []
        center_x_list = []
        center_y_list = []

        log_skypy_red_galaxy_masses = np.log10(red_galaxies["stellar_mass"])
        log_skypy_blue_galaxy_masses = np.log10(blue_galaxies["stellar_mass"])

        for halo in halo_data:
            subhalo_m_evolved = halo["subhalo_m_evolved"]
            dist_to_center = halo["dist_to_center"]
            pos_2d = halo["pos_2d"]
            galaxy_mass = halo["galaxy_mass"]
            red = halo["red"]

            # TODO remove weighing, make skypy generate more massive galaxies
            if red:
                # Find sample galaxy with closest redshift and stellar mass
                # skypy returns in physical mass
                # weigh redshift higher
                closest_real_galaxy_index = np.argmin(
                    np.hypot(
                        5 * (red_galaxies["z"] - self.z),
                        log_skypy_red_galaxy_masses
                        - np.log10(galaxy_mass / self.cosmo.h),
                    )
                )

                light_dict = dict(red_galaxies[closest_real_galaxy_index])
            else:  # blue
                # Find sample galaxy with closest redshift and stellar mass
                # skypy returns in physical mass
                # weigh redshift higher
                closest_real_galaxy_index = np.argmin(
                    np.hypot(
                        5 * (blue_galaxies["z"] - self.z),
                        log_skypy_blue_galaxy_masses
                        - np.log10(galaxy_mass / self.cosmo.h),
                    )
                )

                light_dict = dict(blue_galaxies[closest_real_galaxy_index])

            del light_dict["z"]
            light_dict["extended_source_type"] = "single_sersic"  # TODO hernquist or sersic better
            light_dict["n_sersic"] = 4  # TODO elliptical / spiral

            # eccentricity
            # galaxy_projected_eccentricity wants (1 - q^2) / (1 + q^2)
            light_dict["e1"], light_dict["e2"] = galaxy_projected_eccentricity(
                e2epsilon(light_dict["ellipticity"]),
                halo["light_position_angle"],
            )

            if subhalo_m_evolved > 0:
                # concentration was chosen using the accretion mass
                # slsim wants physical masses, not / h
                mass_dict = {
                    "mass_type": "NFW_HERNQUIST",
                    "halo_mass": subhalo_m_evolved / self.cosmo.h,
                    "concentration": halo["concentration"],
                    "e1": 0,
                    "e2": 0,
                }

                if dist_to_center != 0:  # subhalo
                    #if subhalo_profile == "TNFW":
                    mass_dict["truncation_radius"] = (
                        1.4
                        * (subhalo_m_evolved / self.cosmo.h / 10**7) ** (1 / 3)
                        * (dist_to_center / self.cosmo.h / 50) ** (2 / 3)
                    )
                    #elif subhalo_profile == "GBNFW":
                    #    raise ValueError()  # TODO
                    #else:
                    #    raise ValueError()
                else:  # host halo
                    # galaxy_projected_eccentricity wants (1 - q^2) / (1 + q^2)
                    e1, e2 = galaxy_projected_eccentricity(
                        ellip_from_axis_ratio2epsilon(ellipticity_host),
                        mean_position_angle,
                    )

                    mass_dict["e1"] = e1
                    mass_dict["e2"] = e2

                    if kwargs_host_mass is not None:
                        mass_dict.update(kwargs_host_mass)
            else:  # subhalo is completely stripped
                mass_dict = {"mass_type": "HERNQUIST"}

            light_dicts.append(light_dict)
            mass_dicts.append(mass_dict)
            center_x_list.append(pos_2d[0])
            center_y_list.append(pos_2d[1])

        return {
            "kwargs_mass_list": mass_dicts,
            "kwargs_light_list": light_dicts,
            "center_x_deflector_list": center_x_list,
            "center_y_deflector_list": center_y_list,
        }

    def get_deflector(
        self,
        red_galaxies,
        blue_galaxies,
        min_subhalo_accretion_mass=None,
        crop_subhalo_dist=100,
        #subhalo_profile="TNFW",  # TNFW, GBNFW not implemented yet
        kwargs_host_mass=None,
    ):
        """
        :param red_galaxies: catalog of red galaxies to match to
        :param blue_galaxies: catalog of blue galaxies to match to
        :param min_subhalo_accretion_mass: min subhalo accretion mass [solar masses / h]
        :param crop_subhalo_dist: only return subhalos closer than this distance to the BCG [arcsecs]

        returns DeflectorGroup
        """

        return DeflectorGroup(
            self.z,
            **self.get_deflector_data(
                red_galaxies,
                blue_galaxies,
                min_subhalo_accretion_mass,
                crop_subhalo_dist,
                #subhalo_profile=subhalo_profile,
                kwargs_host_mass=kwargs_host_mass,
            ),
        )

    def _fraction_red_galaxies(self, r, m):
        """Returns probability that a galaxy is red at a distance r (kpc / h)
        with stellar mass m (solar masses / h)
        
        Zheng 2025"""

        paramc, params = gals_init()

        central_stellar_mass = stellarmass_halomass(
            self.M200h, self.z, paramc
        )  # solar masses / h

        log_Mc = np.log10(central_stellar_mass)
        log_Ms = np.log10(m)

        A1 = 0.09
        eta1 = 1.07
        nu1 = 1.67

        A2 = 0.05
        eta2 = 0.81
        nu2 = 1.80

        A = 1.0 - A1 * (13.0 - log_Mc) ** eta1 * (11.2 - log_Ms) ** nu1
        k = -A2 * (13.0 - log_Mc) ** eta2 * (11.2 - log_Ms) ** nu2

        return np.where(
            log_Ms > 11.2, 1, np.clip(A * (r / self.Rvir) ** k, 0, 1)
        )

    def _fraction_central_red(self, central_stellar_mass):
        """Probability that central galaxy is red. Zheng 2025

        :param central_stellar_mass: solar masses / h
        """

        mu = 10.02
        sigma = 0.64

        return 0.5 * (
            1 + erf((np.log10(central_stellar_mass) - mu) / (np.sqrt(2) * sigma))
        )


class GeneratedDeflectorPopulation:
    def __init__(
        self,
        Mmin,
        Mmax,
        zmin,
        zmax,
        red_galaxies,
        blue_galaxies,
        sky_area,
        cosmo,
        crop_subhalo_dist=100,
        min_subhalo_accretion_mass=None,
        verbose=False,
    ):
        """
        :param Mmin: Minimum host halo mass (solar masses / h)
        :param Mmax: Maximum host halo mass (solar masses / h)
        :param zmin: min redshift
        :param zmax: max redshift
        :param red_galaxies: list of red galaxies from skypy
        :param blue_galaxies: list of blue galaxies from skypy
        :param sky_area: sky area over which to draw deflectors, set very high to get a realistic population
        :type sky_area: astropy.Quantity
        :param cosmo: astropy cosmology instance
        :param crop_subhalo_dist: maximum distance in arcsecs from the center that satellites will be generated
        :param min_subhalo_accretion_mass: minimum mass of accreted subhalos to sample (solar masses / h). Default 10^-3 times the host halo mass, but can be set explicitly if generating smaller groups
        :param verbose: verbose
        """

        self.Mmin = Mmin
        self.Mmax = Mmax
        self.zmin = zmin
        self.zmax = zmax
        self.red_galaxies = red_galaxies
        self.blue_galaxies = blue_galaxies
        self.crop_subhalo_dist = crop_subhalo_dist
        self.min_subhalo_accretion_mass = min_subhalo_accretion_mass
        self.cosmo = cosmo
        self.sky_area = sky_area
        self.verbose = verbose

        self.deflectors = []

        cosmo_col = set_colossus_cosmo(cosmo)

        MM_h = np.logspace(np.log10(Mmin), np.log10(Mmax), 1000)
        dlnm = np.log(MM_h[1]) - np.log(MM_h[0])
        sky_area_deg2 = sky_area.to_value("deg2")

        dz = 0.001
        for z in np.arange(zmin, zmax, dz):
            counts = np.random.poisson(
                dNhalodzdlnM_lens(MM_h, z, cosmo_col, mdef="200c", model="tinker08")
                * sky_area_deg2
                * dlnm
                * dz
            )

            for m, count in zip(MM_h, counts):
                if count > 0:
                    self.deflectors += [(m, z)] * count

    def draw_deflector(self, kwargs_host_mass=None):
        """Draw a random deflector.

        Returns DeflectorGroup object

        :param kwargs_host_mass: Specifying kwargs_host_mass={"gamma_in": gamma} makes the
        host halo a GNFW profile. gamma=1 corresponds to NFW, with lower values being shallower.
        """

        M, z = random.choice(self.deflectors)

        if self.verbose:
            print(f"Drew log10(M200h) {np.log10(M):.4f} z {z:.4f}")

        generated_deflector = GeneratedDeflector(
            M, z, self.cosmo, verbose=self.verbose
        )

        return generated_deflector.get_deflector(
            self.red_galaxies,
            self.blue_galaxies,
            crop_subhalo_dist=self.crop_subhalo_dist,
            min_subhalo_accretion_mass=self.min_subhalo_accretion_mass,
            kwargs_host_mass=kwargs_host_mass,
        )

    def deflector_number(self):
        """Returns the number of distinct deflectors in the population."""

        return len(self.deflectors)