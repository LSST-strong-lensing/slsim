"""The GeneratedDeflector class implements the analytic subhalo model from Han
et al. (2016) Some other choices are made following Abe et al. (2025)

Model summary:

1. First, the accretion subhalo masses are sampled from the SHMF given in Han et al. (2018), table 1, fit to the Millenium 2 simulation
The SHMF given in the original paper is inaccurate for higher subhalo masses

2. Host halo concentration is set by the Diemer19 mass-concentration relation with log scatter 0.33
The accreted subhalos are placed randomly in the host halo (within R200) tracing the host halo density

3. The evolved subhalo masses are calculated according to eq. 7 in Han 2016, with some subhalos being completely stripped
Subhalo concentration is chosen the same way using the accretion mass
Subhalos use a truncated NFW profile with truncation radius from eq. 6 in Gilman et al. (2016)

4. Galaxy masses are calculated from subhalo accretion mass from the Behroozi SMHM relation, with log scatter 0.2

5. Galaxies are randomly selected as red or blue according to their distance to the center. Hennig et al. (2017) describes the density of galaxies
in the cluster as NFW profiles with red galaxies being more concentrated in the center, as well as the average fraction of red galaxies
which is a function of host mass and redshift.

6. The closest matching red / blue galaxy in terms of stellar mass and redshift is then selected from the skypy catalog

TODO better truncation radius, different subhalo profile, eccentricity for halos

References:
Han et al. (2016): https://academic.oup.com/mnras/article/457/2/1208/965286
Han et al. (2018): https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract
Abe et al. (2025): https://arxiv.org/abs/2411.07509
Gilman et al. (2019): https://arxiv.org/abs/1908.06983
Hennig et al. (2017): https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.4015H/abstract
"""

from colossus.halo import profile_nfw
from colossus.halo import concentration

import numpy as np

# from scipy.stats import truncnorm

from slsim.Deflectors.MassLightConnection.galaxy_population import (
    gals_init,
    stellarmass_halomass,
)
from slsim.Deflectors.deflector_util import set_colossus_cosmo

from slsim.Deflectors.deflector_group import DeflectorGroup

from slsim.Sources.SourcePopulation.galaxies import galaxy_projected_eccentricity

from slsim.Halos.halo_population import dNhalodzdlnM_lens

import random


class GeneratedDeflector:
    def __init__(self, M200h, z, cosmo):
        """
        :param M: host halo mass (200c) in solar masses / h
        :param z: deflector redshift
        :param cosmo: astropy cosmology instance
        """

        set_colossus_cosmo(cosmo)

        c = concentration.concentration(M200h, "200c", z, model="diemer19")
        c = np.random.lognormal(
            mean=np.log(c), sigma=0.33
        )  # concentration distribution 0.16 dex
        self.p_nfw = profile_nfw.NFWProfile(M=M200h, c=c, z=z, mdef="200c")

        self.M200h = M200h  # solar masses / h
        self.R200h = self.p_nfw.RDelta(z, "200c")  # kpc / h

        self.z = z
        self.cosmo = cosmo

        # fraction of red galaxies, function of redshift and mass
        f_red_200 = (
            0.68
            * (self.M200h / self.cosmo.h / (6 * 10**14)) ** -0.10
            * ((1 + self.z) / (1 + 0.46)) ** -0.65
        )
        self.f_red_200 = np.random.normal(loc=f_red_200, scale=0.14 * f_red_200)

    def accreted_subhalo_density_pdf(self, m_acc_h):
        """Returns accreted subhalo dN / dlnm.

        :param m_acc: subhalo accretion mass [solar masses / h]
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

    def unevolved_spatial_distribution_pdf(self, r):
        """
        r: kpc/h

        In the model, when selecting by accretion mass the final position traces host halo density
        This function returns normalized dN / d3r as a function of radius, so the total probability over the host halo is 1
        """

        return self.p_nfw.density(r) / self.M200h

    def evolved_subhalo_mass_fraction_of_accretion(self, r):
        """Returns m_evolved / m_acc.

        :param r: distance to halo center [kpc / h]
        """

        M = self.M200h / 10**10

        mustar = 0.5 * M**-0.03  # stripping function amplitude
        beta = 1.7 * M**-0.04  # stripping function slope
        sigma = 1.1  # log scatter of m / m_acc
        fs = 0.55  # fraction of survived subhaloes

        mubar = mustar * (r / self.R200h) ** beta  # average evolved mass fraction
        mu = np.random.lognormal(mean=np.log(mubar), sigma=sigma)  # with scatter

        return np.where(np.random.rand(*r.shape) < fs, mu, 0)  # 0 mass when stripped

    def generate_subhalos(self, m_acc_min, m_acc_max):
        """Randomly samples subhalos from m_acc_min to m_acc_max (solar masses
        / h)

        Returns list of tuples (accretion mass [solar masses / h],
        evolved mass [solar masses / h], distance to center [kpc / h])
        """

        m_acc = np.logspace(np.log10(m_acc_min), np.log10(m_acc_max), num=1000, base=10)
        r = self.R200h * np.linspace(0.001, 1, 1000)

        d3r = 4 / 3 * np.pi * (r[1:] ** 3 - r[:-1] ** 3)

        # probability that the accreted subhalo ended up at each radius (traces halo density profile)
        spatial_distribution_multiplier = (
            self.unevolved_spatial_distribution_pdf(r[:-1]) * d3r
        )

        dlnm = np.log(m_acc[1]) - np.log(m_acc[0])

        # mass_acc, mass_evolved, radius
        subhalos = []

        for m in m_acc:
            # expected number of accreted subhalos of this mass
            accreted_subhalo_expected = self.accreted_subhalo_density_pdf(m) * dlnm

            # function of radius, expected number of subhalos with this accretion mass at this radius
            probability_present = (
                spatial_distribution_multiplier * accreted_subhalo_expected
            )

            # poisson unnecessary
            present_r_indices = np.where(
                np.random.rand(*probability_present.shape) < probability_present
            )[0]

            mass_acc = [m] * len(present_r_indices)
            mass_evolved = m * self.evolved_subhalo_mass_fraction_of_accretion(
                r[present_r_indices]
            )
            radius = r[present_r_indices]

            subhalos += list(zip(mass_acc, mass_evolved, radius))

        return subhalos

    def get_deflector_data(
        self,
        red_galaxies,
        blue_galaxies,
        min_subhalo_accretion_mass=None,
        crop_subhalo_dist=1000,
    ):
        """
        :param galaxy_list: list of galaxies to assign as deflectors
        :param min_subhalo_accretion_mass: min subhalo accretion mass [solar masses / h]
        :param crop_subhalo_dist: only return subhalos closer than this distance to the BCG [arcsecs]

        returns kwargs_mass_list, kwargs_light_list, center_x_deflector_list, center_y_deflector_list to be passed into DeflectorGroup
        """

        if min_subhalo_accretion_mass is None:
            min_subhalo_accretion_mass = 10**-3 * self.M200h

        paramc, params = gals_init()
        angular_diameter_dist = self.cosmo.angular_diameter_distance(self.z).to_value(
            "kpc"
        )  # distance corresponding to 1 radian

        mean_position_angle = np.random.rand() * 2 * np.pi

        light_dicts = []
        mass_dicts = []
        center_x_list = []
        center_y_list = []

        log_skypy_red_galaxy_masses = np.log10(red_galaxies["stellar_mass"])
        log_skypy_blue_galaxy_masses = np.log10(blue_galaxies["stellar_mass"])

        halos = [(self.M200h, self.M200h, 0)] + self.generate_subhalos(
            min_subhalo_accretion_mass, self.M200h
        )
        for subhalo_m_acc, subhalo_m_evolved, dist_to_center in halos:
            # Mo/h, Mo/h, kpc/h

            # place randomly in 3d and cast to 2d
            pos = np.random.normal(size=(3))
            pos /= np.linalg.norm(pos)
            pos_2d = (
                pos[:2] * dist_to_center / self.cosmo.h / angular_diameter_dist * 206265
            )  # 2d coordinate in arcsecs

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

            # TODO remove weighing, make skypy generate more massive galaxies
            if np.random.rand() < self.fraction_red_galaxies(dist_to_center):  # red
                # Find sample galaxy with closest redshift and stellar mass                                #skypy returns in physical mass
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
                # Find sample galaxy with closest redshift and stellar mass                                #skypy returns in physical mass
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
            light_dict["extended_source_type"] = "hernquist"

            # eccentricity
            light_dict["e1"], light_dict["e2"] = galaxy_projected_eccentricity(
                light_dict["ellipticity"],
                np.random.normal(loc=mean_position_angle, scale=35.4 * np.pi / 180),
            )

            if subhalo_m_evolved > 0:
                # use accretion mass for concentration
                c = concentration.concentration(
                    subhalo_m_acc, "200c", self.z, model="diemer19"
                )
                c = np.random.lognormal(
                    mean=np.log(c), sigma=0.33
                )  # concentration distribution 0.16 dex

                # slsim wants physical masses, not / h
                mass_dict = {
                    "mass_type": "NFW_HERNQUIST",
                    "halo_mass": subhalo_m_evolved / self.cosmo.h,
                    "concentration": c,
                    "e1": 0,
                    "e2": 0,
                }

                if dist_to_center != 0:  # subhalo
                    mass_dict["truncation_radius"] = (
                        1.4
                        * (subhalo_m_evolved / self.cosmo.h / 10**7) ** (1 / 3)
                        * (dist_to_center / self.cosmo.h / 50) ** (2 / 3)
                    )
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
        crop_subhalo_dist=1000,
    ):
        """
        :param galaxy_list: list of galaxies to assign as deflectors
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
            )
        )

    def fraction_red_galaxies(self, r):
        """Returns probability that a galaxy is red at a distance r (kpc / h)

        Red and blue galaxy densities in the cluster are modeled as NFW
        profiles with different concentrations

        f_red_200 is calculated earlier as a function of mass and
        redshift

        Data taken from Hennig et al. (2017)
        """

        # From f_red_200, we need to calculate the NFW densities describing the galaxy distributions from their concentrations

        c_red = 5.37
        c_blue = 1.38

        def nfw(x, c):
            y = c * x
            y = max(y, 0.0001)
            return 1 / (y * (1 + y) ** 2)

        def enclosed(y):
            y = max(y, 0.0001)
            return np.log(1 + y) - y / (1 + y)

        I_red = enclosed(c_red) / c_red**3
        I_blue = enclosed(c_blue) / c_blue**3

        A_red = self.f_red_200 / I_red
        A_blue = (1 - self.f_red_200) / I_blue

        rho_red = A_red * nfw(r / self.R200h, c_red)
        rho_blue = A_blue * nfw(r / self.R200h, c_blue)

        return rho_red / (rho_red + rho_blue)


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
        :param min_subhalo_accretion_mass: minimum mass of accreted subhalos to sample (solar masses / h). Default 10^-3 M_hh, but can be set explicitly if generating smaller groups
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

        self.deflectors = []

        cosmo_col = set_colossus_cosmo(cosmo)

        MM_h = np.logspace(np.log10(Mmin), np.log10(Mmax), 1000)
        dlnm = np.log(MM_h[1]) - np.log(MM_h[0])

        dz = 0.001
        for z in np.arange(zmin, zmax, dz):
            counts = np.random.poisson(
                dNhalodzdlnM_lens(MM_h, z, cosmo_col, mdef="200c", model="tinker08")
                * sky_area.to_value("deg2")
                * dlnm
                * dz
            )

            for m, count in zip(MM_h, counts):
                if count > 0:
                    self.deflectors += [(m, z)] * count

    def draw_deflector(self):
        """Draw a random deflector.

        Returns DeflectorGroup object
        """

        M, z = random.choice(self.deflectors)

        # print(f"Drew M {np.log10(M):.4f}/h z {z:.4f}")

        generated_deflector = GeneratedDeflector(M, z, self.cosmo)

        return generated_deflector.get_deflector(
            self.red_galaxies,
            self.blue_galaxies,
            crop_subhalo_dist=self.crop_subhalo_dist,
            min_subhalo_accretion_mass=self.min_subhalo_accretion_mass,
        )

    def deflector_number(self):
        return len(self.deflectors)
