import os
import pandas as pd
import numpy as np
from lenstronomy.Util import constants

import solve_lenseq
import lens_gals
import lens_halo
# import lens_subhalo


class SLHammocksPipeline:
    """Class for skypy configuration."""

    def __init__(self, slhammocks_config=None, sky_area=None, cosmo=None):
        """
        :param slhammmocks_config: path to SkyPy configuration yaml file.
                            If None, uses 'data/SkyPy/lsst-like.yml'.
        :type slhammmocks_config: string or None
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled.
                                Must be in units of solid angle.
        :param filters: filters for SED integration
        :type filters: list of strings or None
        :param cosmo: An instance of an astropy cosmology model
                        (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
        :type cosmo: astropy.cosmology instance or None
        """
        if slhammocks_config!=None:
            # path = os.getcwd()
            # if slhammocks_config is None:
            #     slhammocks_config = os.path.join(path, "../data/SL-Hammocks/gal_pop_Salpeter_10deg2_zl2.csv")

            df = pd.read_csv(slhammocks_config)
            # mlens_halo = np.where(df["mass_sh"]==-1, df["mass_hh"], df["mass_sh"])
            # mlens_gal = np.where(df["mass_sh"]==-1, df["mass_cen"], df["mass_sat"])
            # lens_con = np.where(df["mass_sh"]==-1, df["param1_hh"], df["param1_sh"])
            # lens_tb = np.where(df["mass_sh"]==-1, df["param1_cen"], df["param1_sat"])
            # elip_lens_gal = np.where(df["mass_sh"]==-1, df["elip_cen"], df["elip_sat"])
            # df["mlens_halo"]=mlens_halo
            # df["mlens_gal"]=mlens_gal
            # df["lens_tb"]=lens_tb
            # df["lens_con"]=lens_con
            # df["elip_lens_gal"] = elip_lens_gal
            # df = df.rename(columns={'zl_hh': 'z'})
            df = df.rename(columns={'zl': 'z'})
            df = df.rename(columns={'con': 'concentration'})
            df = df.rename(columns={'m_g': 'stellar_mass'})
            df = df.rename(columns={'m_h': 'halo_mass'})
            df['angular_size']=df['tb']/0.551*constants.arcsec

            data_area = 10.0 #deg2
            if(sky_area.value>10.0):
                print("Now sky_area should be lower than 10. Now we set sky_area=10.0.")
            thinp = int((data_area/sky_area).value)

            self._pipeline = df[::thinp]
        else:
            kwargs_population_base = {
                'z_min': 0.01,
                'z_max': 5.0,
                'log10host_halo_mass_min': 11.,
                'log10host_halo_mass_max': 16.,
                'log10subhalo_mass_min' : 10.,
                # Intrinsic scatter (https://arxiv.org/abs/astro-ph/0608157)
                'sigma_host_halo_concentration' : 0.33,
                'sigma_subhalo_concentration' : 0.33,
                # Intrinsic scatter (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
                'sigma_central_galaxy_mass' : 0.2,
                'sigma_satellite_galaxy_mass' : 0.2,
                'sig_tb': 0.46,
                'TYPE_GAL_SIZE': 'vdW23',
                'frac_SM_IMF': 1.715,
                'TYPE_SMHM': 'true',
                'switch_sub': False
                }
            from colossus.cosmology import cosmology
            # kwargs_population = update_dict_values(kwargs_population_base, **kwargs_population)
            cosmo_col = cosmology.setCosmology('planck18')

            # population_dict = compound_model_population_dictionary(kwargs_population)
            self._pipeline = halo_galaxy_population(sky_area,cosmo_col,**kwargs_population_base)

    # @property
    # def central_galaxies(self):

    #     """Skypy pipeline for blue galaxies.

    #     :return: list of blue galaxies
    #     :rtype: list of dict
    #     """
    #     return self._pipeline[self._pipeline["mass_sh"]==-1]

    # @property
    # def satellite_galaxies(self):
    #     """Skypy pipeline for red galaxies.

    #     :return: list of red galaxies
    #     :rtype: list of dict
    #     """
    #     return self._pipeline[self._pipeline["mass_sh"]!=-1]

def update_dict_values(original_dict, **kwargs):
    for key, value in kwargs.items():
        if key in original_dict:
            original_dict[key] = value
        else:
            print(f"Key '{key}' not found in the dictionary")
    return original_dict

def halo_galaxy_population(sky_area,cosmo_col,z_min,z_max,log10host_halo_mass_min,log10host_halo_mass_max,
                           log10subhalo_mass_min, sigma_host_halo_concentration, sigma_subhalo_concentration,
                           sigma_central_galaxy_mass, sigma_satellite_galaxy_mass, sig_tb,
                           TYPE_GAL_SIZE,frac_SM_IMF, TYPE_SMHM, switch_sub,**kwargs):
    dz = 0.001
    dlogMh = 0.001
    dlnMh = np.log(10**dlogMh)
    # cosmological parameters
    area = sky_area.value
    halo_gal_pop_array = np.array([])
    halo_gal_pop_array = np.empty((0,10), float)
    zz = np.arange(z_min, z_max + dz,dz)
    Mh_min = 10**log10host_halo_mass_min
    Mh_max = 10**log10host_halo_mass_max
    MMh = 10**np.arange(np.log10(Mh_min),
                        np.log10(Mh_max), dlogMh)
    min_Msh =10**log10subhalo_mass_min
    paramc, params = lens_gals.gals_init(TYPE_SMHM)
    sig_c = sigma_host_halo_concentration
    sig_csh = sigma_subhalo_concentration
    sig_mcen = sigma_central_galaxy_mass
    sig_msat = sigma_satellite_galaxy_mass

    for z in zz:
        zz2 = np.full(len(MMh), z)
        NNh = area * \
            lens_halo.dNhalodzdlnM_lens(MMh, zz2, cosmo_col) * dlnMh * dz
        Nh = np.random.poisson(NNh)
        indices = np.nonzero(Nh)[0]
        if len(indices) == 0:
            continue

        zl_tab = np.repeat(zz2[indices], Nh[indices])
        Mhosthl_tab  = np.repeat(MMh[indices], Nh[indices])
        conhl_tab = lens_halo.concent_m_w_scatter(Mhosthl_tab, z, sig_c)
        # in physical [Mpc/h]
        eliphl_tab, polarhl_tab = solve_lenseq.gene_e_ang_halo(Mhosthl_tab)

        mshsat_tot = 0
        Mhosthl_tab_re = Mhosthl_tab

        Mcenl_ave = lens_gals.stellarmass_halomass(Mhosthl_tab_re / (cosmo_col.H0 / 100.), zl_tab, paramc, frac_SM_IMF) * (cosmo_col.H0 / 100.)
        Mcenl_scat = np.random.lognormal(0.0, sig_mcen, size=Mhosthl_tab_re.shape)
        Mcenl_tab = Mcenl_ave * Mcenl_scat

        elipcenl, polarcenl = lens_gals.set_gals_param(polarhl_tab)
        tb_cen = lens_gals.galaxy_size(Mhosthl_tab_re, Mcenl_tab/frac_SM_IMF, zl_tab, cosmo_col, model=TYPE_GAL_SIZE, scatter=True, sig_tb=sig_tb)
        halogal_par_mat = np.hstack((zl_tab.reshape(-1, 1), Mhosthl_tab_re.reshape(-1, 1), np.zeros_like(Mhosthl_tab).reshape(-1, 1), eliphl_tab.reshape(-1, 1), polarhl_tab.reshape(-1, 1), conhl_tab.reshape(-1, 1),
                                        Mcenl_tab.reshape(-1, 1), elipcenl.reshape(-1, 1), polarcenl.reshape(-1, 1), tb_cen.reshape(-1, 1)))

        halo_gal_pop_array = np.append(halo_gal_pop_array, halogal_par_mat, axis=0)

        halo_gal_pop_array = np.append(halo_gal_pop_array, halogal_par_mat, axis=0)
    columns_pop = ["z", "halo_mass", "halo_mass_acc", "e_h" , "p_h", "concentration", "stellar_mass", "e_g", "p_g", "tb"]
    data_frame_pop = pd.DataFrame(halo_gal_pop_array)
    data_frame_pop.columns =columns_pop

    return data_frame_pop
