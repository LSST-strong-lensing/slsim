import os
import tempfile
import slsim
import pandas as pd
import numpy as np

# import solve_lenseq
# import lens_gals
# import lens_halo
# import lens_subhalo
# import global_value as g


class SLHammocksPipeline:
    """Class for skypy configuration."""

    def __init__(self, slhammocks_config=None, sky_area=None, cosmo=None, pop_generate=False):
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
        if pop_generate:
            print("generating population of lensing candidates")
            self._pipeline = population_generator(area,cosmo)
        else:
            path = os.getcwd()
            if slhammocks_config is None:
                slhammocks_config = os.path.join(path, "../data/SL-Hammocks/gal_pop_Salpeter_10deg2_zl2.csv")

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
            df = df.rename(columns={'m_g': 'm_gal'})
            df = df.rename(columns={'m_h': 'm_halo'})

            data_area = 10.0 #deg2
            if(sky_area.value>10.0):
                print("Now sky_area should be lower than 10. Now we set sky_area=10.0.")
            thinp = int((data_area/sky_area).value)

            self._pipeline = df[::thinp]

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

# def population_generator(area,cosmo,zmin=0.01,zmax=5.0,Mhmin=1e11,Mhmax=1e16,switch_sub=False):
#     result_halogal_par = []
#     dz = 0.001
#     zz_ar = np.arange(zmin, zmax + dz, dz)
#     dlogMh = 0.001
#     dlnMh = np.log(10**dlogMh)
#     MMh = 10**np.arange(np.log10(Mhmin),
#                         np.log10(Mhmax), dlogMh)
#     min_Msh =Mhmin/10.
#     if switch_sub:
#         n_bins = 100
#         output_length = n_bins-1
#         interp_dnsh, interp_msh_acc_Mh = lens_subhalo.create_interp_dndmsh(0, zmax+0.1, Mhmin/2., Mhmax*2., min_Msh, cosmo, n_bins=n_bins)

#     # Start for loop-1 of redshift
#     for zz in zz_ar:
#         zz2 = np.full(len(MMh), zz)
#         NNh = area * \
#             lens_halo.dNhalodzdlnM_lens(MMh, zz2, cosmo) * dlnMh * dz
#         Nh = np.random.poisson(NNh)
#         indices = np.nonzero(Nh)[0]
#         if len(indices) == 0:
#             continue

#         zl_tab = np.repeat(zz2[indices], Nh[indices])
#         Mhosthl_tab  = np.repeat(MMh[indices], Nh[indices])
#         conhl_tab = lens_halo.concent_m_w_scatter(Mhosthl_tab, zz, g.sig_c)
#         # in physical [Mpc/h]
#         eliphl_tab, polarhl_tab = solve_lenseq.gene_e_ang_halo(Mhosthl_tab)

#         mshsat_tot = 0
#         Mhosthl_tab_re = Mhosthl_tab

#         if switch_sub == True:
#             Mhl_zl_tab_vec = np.vstack((np.log10(Mhosthl_tab),  zl_tab)).T
#             dnshp = interp_dnsh(Mhl_zl_tab_vec).reshape(
#                 len(Mhosthl_tab), output_length)
#             msh_acc_Mh = interp_msh_acc_Mh(Mhl_zl_tab_vec).reshape(
#                 len(Mhosthl_tab), output_length)
#             dnsh = np.where((msh_acc_Mh > 0.5) | (dnshp < 1.0e-4), 0., dnshp)
#             mmsh_acc = msh_acc_Mh*Mhosthl_tab.reshape(len(Mhosthl_tab), -1)
#             mmsh = np.logspace(np.log10(min_Msh), np.log10(
#                 Mhosthl_tab/2.), n_bins).T[:, 1:]  # in [Modot/h]
#             NNsh = np.random.poisson(dnsh)

#             # Start: for-loop-2 of host halos
#             for j, (mh, zl) in enumerate(zip(Mhosthl_tab, zl_tab)):
#                 indices_sh = np.nonzero(NNsh[j])[0]
#                 cut_Nsh = NNsh[j][indices_sh]
#                 cut_msh = mmsh[j][indices_sh]
#                 cut_msh_acc = mmsh_acc[j][indices_sh]
#                 # Subhalo mass
#                 msh_tab = np.repeat(cut_msh, cut_Nsh)
#                 msh_acc_tab = np.repeat(cut_msh_acc, cut_Nsh)
#                 zsub_tab = np.full(len(msh_tab), zz)
#                 mshsat_tot = 0
#                 # Start: If at least one subhalo exists in loop-2 of host halos
#                 if len(msh_tab) != 0:
#                     # Satellite gals
#                     msat_ave = lens_gals.stellarmass_halomass(
#                         msh_acc_tab/(cosmo.H0/100.), zl, g.params, g.frac_SM_IMF)*(cosmo.H0/100.)  # KA
#                     msat_scat = np.random.lognormal(0.0, g.sig_msat, len(msh_acc_tab))
#                     msat_tab = msat_ave*msat_scat

#                     mshsat_tot = sum(msh_tab)+sum(msat_tab)
#                     elipsh_tab, polarsh_tab = solve_lenseq.gene_e_ang_halo(msh_acc_tab)
#                     con_sh_ave = np.where(lens_subhalo.concent_m_sub_ando(msh_tab, zl, cosmo) > lens_halo.concent_m(msh_tab, zl),
#                                       lens_subhalo.concent_m_sub_ando(msh_tab, zl, cosmo), lens_halo.concent_m(msh_tab, zl))
#                     cor_con_sh = mass_so.M_to_R(
#                         msh_acc_tab, zl, 'vir') / mass_so.M_to_R(msh_tab, zl, 'vir')
#                     con_sh_tab = con_sh_ave * \
#                         cor_con_sh * np.random.lognormal(0.0, g.sig_c_sh, len(msh_tab))
#                     elipsat_tab, polarsat_tab = lens_gals.set_gals_param(polarsh_tab)
#                     tb_sat_tab = lens_gals.galaxy_size(
#                          msh_acc_tab, msat_tab/g.frac_SM_IMF, zz, cosmo, model=g.TYPE_GAL_SIZE, scatter=True, sig_tb=g.sig_tb)
#                     halogal_par_mat = np.hstack((zsub_tab.reshape(-1, 1), msh_tab.reshape(-1, 1),  msh_acc_tab.reshape(-1, 1), elipsh_tab.reshape(-1, 1), polarsh_tab.reshape(-1, 1), con_sh_tab.reshape(-1, 1), msat_tab.reshape(-1, 1), elipsat_tab.reshape(-1, 1), polarsat_tab.reshape(-1, 1), tb_sat_tab.reshape(-1, 1)))

#                     result_halogal_par.append(halogal_par_mat)
#                 # End: If at least one subhalo exists in loop-2 of host halos

#                 Mhosthl_tab_re[j] = mh-mshsat_tot

#         Mcenl_ave = lens_gals.stellarmass_halomass(Mhosthl_tab_re / (cosmo.H0 / 100.), zl_tab, g.paramc, g.frac_SM_IMF) * (cosmo.H0 / 100.)
#         Mcenl_scat = np.random.lognormal(0.0, g.sig_mcen, size=Mhosthl_tab_re.shape)
#         Mcenl_tab = Mcenl_ave * Mcenl_scat

#         elipcenl, polarcenl = lens_gals.set_gals_param(polarhl_tab)
#         tb_cen = lens_gals.galaxy_size(Mhosthl_tab_re, Mcenl_tab/g.frac_SM_IMF, zl_tab, cosmo, model=g.TYPE_GAL_SIZE, scatter=True, sig_tb=g.sig_tb)
#         halogal_par_mat = np.hstack((zl_tab.reshape(-1, 1), Mhosthl_tab_re.reshape(-1, 1), np.zeros_like(Mhosthl_tab).reshape(-1, 1), eliphl_tab.reshape(-1, 1), polarhl_tab.reshape(-1, 1), conhl_tab.reshape(-1, 1),
#                                      Mcenl_tab.reshape(-1, 1), elipcenl.reshape(-1, 1), polarcenl.reshape(-1, 1), tb_cen.reshape(-1, 1)))

#         result_halogal_par.append(halogal_par_mat)

#     return result_halogal_par
