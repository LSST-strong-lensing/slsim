import os
import tempfile
import slsim
import pandas as pd
import numpy as np

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
