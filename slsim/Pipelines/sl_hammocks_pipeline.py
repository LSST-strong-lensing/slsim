import numpy as np
import os
from astropy.table import Table, hstack
from colossus.cosmology import cosmology
from colossus.halo import mass_defs
import slsim.Deflectors.MassLightConnection.galaxy_population as galaxy_population
import slsim
import slsim.Halos.halo_population as halo_population
import slsim.Util.param_util as util
import tempfile
from skypy.pipeline import Pipeline


class SLHammocksPipeline:
    """SLHammocksPipeline is a class that generate galaxy populations using a
    halo-model approach. It supports either loading pre-generated galaxy
    population data from a CSV file or generating the population on-the-fly
    based on halo occupation and galaxy property models.

    The pipeline integrates halo properties (mass, redshift) with
    stellar masses and computes photometric magnitudes using SkyPy. For
    this we need to execute skypy pipeline with in this class.
    """

    def __init__(
        self,
        skypy_config=None,
        slhammocks_config=None,
        sky_area=None,
        cosmo=None,
        z_min=None,
        z_max=None,
        loghm_min=11.5,
        loghm_max=16.0,
    ):
        """
        :param skypy_config: path to the SkyPy configuration file.
        :type skypy_config: string or None
        :param slhammocks_config: path to the deflector population csv file for 'halo-model'
                            If None, generate the population. Not supported at this time.
        :type slhammocks_config: string or None
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled.
                                Must be in units of solid angle.
        :param filters: filters for SED integration
        :type filters: list of strings or None
        :param cosmo: An instance of an astropy cosmology model
                        (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
        :type cosmo: astropy.cosmology instance or None
        :param z_min: minimum galaxy redshift
        :param z_max: maximum galaxy redshift
        :param loghm_min: minimum halo mass in log unit
        :param loghm_max: maximum halo mass in log unit
        """
        path = os.path.dirname(slsim.__file__)
        module_path, _ = os.path.split(path)
        if slhammocks_config is not None:
            table = Table.read(slhammocks_config, format="csv")
            table = table_translator_for_slsim(table, cosmo)

            data_area = 0.001  # deg2 #TODO prepare the data with larger sky area
            if sky_area.value > data_area:
                raise ValueError(
                    "Now sky_area should be lower than the sky_area size of prepared data file"
                )
            else:
                thinp = int((data_area / sky_area).value)

            self._pipeline = table[::thinp]
        else:
            """
            - 'z_min' : Lower limit of the redshift range when generating delector population.
            - 'z_max' : Upper limit of the redshift range when generating delector population.
            - 'log10host_halo_mass_min' : Lower limit of host halo mass (in log10 scale).
            - 'log10host_halo_mass_max' : Upper limit of host halo mass (in log10 scale).
            - 'sigma_host_halo_concentration' : Intrinsic scatter of the concentration parameter of dark matter halos.
                    Represents the variance value in a log-normal distribution.
                    (https://arxiv.org/abs/astro-ph/0608157)
            - 'sigma_central_galaxy_mass' : Intrinsic scatter of the stellar-mass-halo-mass relation.
                    Represents the variance value in a log-normal distribution.
                    (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
            - 'sig_tb' : Scatter of the galaxy-mass-size relation only used when you set 'oguri20' in "TYPE_GAL_SIZE".
                            Represents the variance value in a log-normal distribution.
            - 'TYPE_GAL_SIZE' : Type of galaxy size model to use.
                            (Now three available options ['vdw23', 'oguri20', 'karmakar23'])
            - 'frac_SM_IMF' : Fraction of stellar M/L ratio against the Chabrier initial mass function(IMF).
                            (e.g., 1.0 for Chabrier IMF, 1.715 for Salpeter IMF)
            - 'TYPE_SMHM' : Type of fitting function for the stellar-mass-halo-mass relation for quiescent galaxies, see Behroozi et al. 2019 for detail
                            (Currently three options ['true', 'obs', 'true_all'])
            - 'sigma8' : The normalization of the power spectrum, i.e. the variance when the field is filtered with a top hat filter of radius 8 Mpc/h.
                            This parameter is required to convert from astropy.cosmology to colossus.cosmology
            - 'ns' : The tilt of the primordial power spectrum. This parameter is required to convert from astropy.cosmology to colossus.cosmology
            """
            kwargs_population_base = {
                "z_min": z_min,
                "z_max": z_max,
                "log10host_halo_mass_min": loghm_min,
                "log10host_halo_mass_max": loghm_max,
                "sigma_host_halo_concentration": 0.33,
                "sigma_central_galaxy_mass": 0.2,
                "TYPE_GAL_SIZE": "vdW23",
                "sig_tb": 0.46,
                "frac_SM_IMF": 1.715,
                "TYPE_SMHM": "true",
                "sigma8": 0.8102,
                "ns": 0.9660499,
            }

            table = halo_galaxy_population(sky_area, cosmo, **kwargs_population_base)
            table = table_translator_for_slsim(table, cosmo)

            self._pipeline = table

        # load SkyPy configuration
        if skypy_config is None:
            skypy_config = os.path.join(module_path, "data/SkyPy/slhammock_skypy.yml")
        with open(skypy_config, "r") as file:
            content = file.read()

        # replace z: PLACEHOLDER_Z with halo redshift
        redshift_array = self._pipeline["z"].value.tolist()
        old_z = "z: PLACEHOLDER_Z"
        new_z = f"z: {redshift_array}"
        content = content.replace(old_z, new_z)
        # replace stellar_mass: PLACEHOLDER_MASS with stellar mass
        stellar_mass_array = self._pipeline["stellar_mass"].value.tolist()
        old_stellar_mass = "stellar_mass: PLACEHOLDER_MASS"
        new_stellar_mass = f"stellar_mass: {stellar_mass_array}"
        content = content.replace(old_stellar_mass, new_stellar_mass)

        # update yaml file with given cosmo.
        content = util.update_cosmology_in_yaml_file(cosmo=cosmo, yml_file=content)
        # execute given yml file in skypy and compute magnitudes of the halo galaxies.
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yml"
        ) as tmp_file:
            tmp_file.write(content)
        self._skypy_pipeline = Pipeline.read(tmp_file.name)
        self._skypy_pipeline.execute()
        magnitude_table = self._skypy_pipeline["halo"]
        magnitude_table.remove_columns(["z", "stellar_mass", "coeff"])
        # stack magnitude with the halo galaxy table
        self._final_galaxy_table = hstack([self._pipeline, magnitude_table])

    @property
    def halo_galaxies(self):
        """Slhammock pipeline for galaxies.

        :return: list of galaxies from halo model
        :rtype: list of dict
        """
        return self._final_galaxy_table


def table_translator_for_slsim(table, cosmo):
    """Translation astropy.table generated by either ways of loading csv file
    or implementing halo_galaxy_population function to be readable in slsim.

    :param table: original deflector population
    :type table: astropy.table
    :param cosmo: An instance of an astropy cosmology model
                    (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
    :type cosmo: astropy.cosmology instance or None

    :return: table. astropy.table.Table. This class initializes an object with data on deflectors.
                This data table should have the following columns:

        - 'z': Redshift of the deflectors
        - 'halo_mass': M200 of the halo components in units of M_sol
        - 'halo_mass_acc': M200 of subhalo component at the accretion time in units of M_sol.
            For host halos, this value becomes 0. Currently, this table does not include subhalos.
        - 'e_h': ellipticily of dark matter halo, which is defined by epsilon=(1-q^2)/(1+q^2), where q=b/a and a, b are major and minor axis of dark matter halo, respectively.
        - 'p_h': posiiton angle of the halo in units of degree
        - 'concentration': Concentration parameter of the halo
        - 'stellar_mass': Mass of stars in the object
        - 'e_g': ellipticity of the galaxy, which is defined by epsilon=(1-q^2)/(1+q^2), where q=b/a and a, b are major and minor axis of galaxy, respectively
        - 'p_g': posiiton angle of the galaxy in units of degree
        - 'tb': the scale radius appreared in Hernquist profile in units of arcsec.
            This parameter relates to the commonly used galaxy effective (half-mass) radius by t_b = 0.551*theta_eff.
        - 'angular_size': galaxy effective radius in units of arcsec
    """
    sigma8 = 0.8102  # from Planck18 TT,TE,EE+lowE+lensing+BAO best-fit value
    ns = 0.9660499  # from Planck18 TT,TE,EE+lowE+lensing+BAO best-fit value
    cosmo_col = cosmology.fromAstropy(cosmo, sigma8, ns, cosmo_name="my_cosmo")
    cosmo_col.Tcmb0 = 2.725

    if "z" not in table.colnames:
        table.rename_column("zl", "z")
    if "concentration" not in table.colnames:
        table.rename_column("con", "concentration")
    if "stellar_mass" not in table.colnames:
        table.rename_column("m_g", "stellar_mass")
    if "halo_mass" not in table.colnames:
        table.rename_column("m_h", "halo_mass")
    if "halo_mass_acc" not in table.colnames:
        table.rename_column("m_acc", "halo_mass_acc")
    if "ellipticity" not in table.colnames:
        table.rename_column("e_g", "ellipticity")
    if "angular_size" not in table.colnames:
        angular_size_in_arcsec = table["tb"] / 0.551
        table.add_column(angular_size_in_arcsec, name="angular_size")

    M200_array, r200_array, c200_array = zip(
        *[
            mass_defs.changeMassDefinition(Mvir, cvir, z, "vir", "200c")
            for Mvir, cvir, z in zip(
                table["halo_mass"], table["concentration"], table["z"]
            )
        ]
    )
    M200_array = np.array(M200_array)
    c200_array = np.array(c200_array)

    hubble = cosmo_col.H0 / 100.0
    table["halo_mass"] = (
        M200_array / hubble
    )  # convert to Mvir [M_sol/h] to physical M200c [M_sol]
    table["halo_mass_acc"] = (
        table["halo_mass_acc"] / hubble
    )  # convert to Mvir [M_sol/h] to physical M200c[M_sol] Currently not supported
    table["concentration"] = c200_array
    table["stellar_mass"] = (
        table["stellar_mass"] / hubble
    )  # convert to stellar mass [M_sol/h] to physical stellar mass [M_sol]

    table["e_h"] = util.ellip_from_axis_ratio2epsilon(
        table["e_h"]
    )  # convert from 1-q to (1-q^2)/(1+q^2)
    table["ellipticity"] = util.ellip_from_axis_ratio2epsilon(
        table["ellipticity"]
    )  # convert from 1-q to (1-q^2)/(1+q^2)

    return table


def halo_galaxy_population(
    sky_area,
    cosmo,
    z_min,
    z_max,
    log10host_halo_mass_min,
    log10host_halo_mass_max,
    sigma_host_halo_concentration,
    sigma_central_galaxy_mass,
    sig_tb,
    TYPE_GAL_SIZE,
    frac_SM_IMF,
    sigma8,
    ns,
    TYPE_SMHM,
    **kwargs,
):
    """
    :param sky_area: Sky area over which galaxies are sampled.
                            Must be in units of solid angle.
    :type sky_area: `~astropy.units.Quantity`
    :param cosmo: An instance of an astropy cosmology model
                    (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
    :type cosmo: astropy.cosmology instance or None
    :param z_min: Lower limit of the redshift range when generating delector population.
    :type z_min: float
    :param z_max: Upper limit of the redshift range when generating delector population.
    :type z_max: float
    :param log10host_halo_mass_min: Lower limit of host halo mass (in log10 scale).
    :type log10host_halo_mass_min: float
    :param log10host_halo_mass_max: Upper limit of host halo mass (in log10 scale).
    :type log10host_halo_mass_max: float
    :param sigma_host_halo_concentration: Intrinsic scatter of the concentration parameter of dark matter halos.
             Represents the variance value in a log-normal distribution. (https://arxiv.org/abs/astro-ph/0608157)
    :type sigma_host_halo_concentration: float
    :param sigma_central_galaxy_mass: Intrinsic scatter of the stellar-mass-halo-mass relation.
            Represents the variance value in a log-normal distribution.
            (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
            :type sigma_central_galaxy_mass: float
    :param sig_tb: Scatter of the galaxy-mass-size relation only used when you set 'oguri20' in "TYPE_GAL_SIZE".
                    Represents the variance value in a log-normal distribution.
    :type sig_tb: float
    :param TYPE_GAL_SIZE: Type of galaxy size model to use.
                    (Now three available options ['vdw23', 'oguri20', 'karmakar23'])
    :type TYPE_GAL_SIZE: str
    :param frac_SM_IMF: Fraction of stellar M/L ratio against the Chabrier initial mass function(IMF).
                    (e.g., 1.0 for Chabrier IMF, 1.715 for Salpeter IMF)
    :type frac_SM_IMF: float
    :param TYPE_SMHM: Type of fitting function for the stellar-mass-halo-mass relation for quiescent galaxies, see Behroozi et al. 2019 for detail
                    (Currently three options ['true', 'obs', 'true_all'])
    :type TYPE_SMHM: str
    :param sigma8: The normalization of the power spectrum, i.e. the variance when the field is filtered with a top hat filter of radius 8 Mpc/h.
                            This parameter is required to convert from astropy.cosmology to colossus.cosmology
    :type sigma8: float
    :param ns: The tilt of the primordial power spectrum. This parameter is required to convert from astropy.cosmology to colossus.cosmology
    :type ns: float
    :param kwargs: keyword arguments
    :type kwargs: dict

    :return: table_pop containing as follows:
        - 'z" : redshift
        - 'halo_mass' : halo mass in units of M_sol/h
        - 'halo_mass_acc' : halo mass at accretion time in units of M_sol/h for subhalos.
        - 'e_h' : ellipticity of halos defined as 1-q, where q is axis ratio
        - 'p_h' : position angle of halos in units of degree.
        - 'concentration' : concentration parameter of halos
        - 'stellar_mass' : stellar mass in units of M_sol/h
        - 'ellipticity' : ellipticity of galaxies defined as 1-q, where q is axis ratio
        - 'p_g' : position angle of galaxies in units of degree.
        - 'tb' : scale radius of galaxies appeared in Hernquist profile in units of arcsec

    """

    dz = 0.001
    dlogMh = 0.001
    dlnMh = np.log(10**dlogMh)
    # cosmological parameters
    area = sky_area.value
    halo_gal_pop_array = np.array([])
    halo_gal_pop_array = np.empty((0, 10), float)
    zz = np.arange(z_min, z_max + dz, dz)
    Mh_min = 10**log10host_halo_mass_min
    Mh_max = 10**log10host_halo_mass_max
    MMh = 10 ** np.arange(np.log10(Mh_min), np.log10(Mh_max), dlogMh)
    paramc, params = galaxy_population.gals_init(TYPE_SMHM)
    sig_c = sigma_host_halo_concentration
    sig_mcen = sigma_central_galaxy_mass
    cosmo_col = cosmology.fromAstropy(cosmo, sigma8, ns, cosmo_name="my_cosmo")
    cosmo_col.Tcmb0 = 2.725

    for z in zz:
        zz2 = np.full(len(MMh), z)
        NNh = area * halo_population.dNhalodzdlnM_lens(MMh, zz2, cosmo_col) * dlnMh * dz
        Nh = np.random.poisson(NNh)
        indices = np.nonzero(Nh)[0]
        if len(indices) == 0:
            continue

        zl_tab = np.repeat(zz2[indices], Nh[indices])
        Mhosthl_tab = np.repeat(MMh[indices], Nh[indices])
        conhl_tab = halo_population.concent_m_w_scatter(Mhosthl_tab, z, sig_c)
        # in physical [Mpc/h]
        eliphl_tab, polarhl_tab = halo_population.gene_e_ang_halo(Mhosthl_tab)

        # mshsat_tot = 0
        Mhosthl_tab_re = Mhosthl_tab
        hubble = cosmo_col.H0 / 100.0

        Mcenl_ave = galaxy_population.stellarmass_halomass(
            Mhosthl_tab_re / (hubble), zl_tab, paramc, frac_SM_IMF
        ) * (hubble)
        Mcenl_scat = np.random.lognormal(0.0, sig_mcen, size=Mhosthl_tab_re.shape)
        Mcenl_tab = Mcenl_ave * Mcenl_scat

        elipcenl, polarcenl = galaxy_population.set_gals_param(polarhl_tab)
        tb_cen = galaxy_population.galaxy_size(
            Mhosthl_tab_re,
            Mcenl_tab / frac_SM_IMF,
            zl_tab,
            cosmo_col,
            model=TYPE_GAL_SIZE,
            scatter=True,
            sig_tb=sig_tb,
        )
        halogal_par_mat = np.hstack(
            (
                zl_tab.reshape(-1, 1),
                Mhosthl_tab_re.reshape(-1, 1),
                np.zeros_like(Mhosthl_tab).reshape(-1, 1),
                eliphl_tab.reshape(-1, 1),
                polarhl_tab.reshape(-1, 1),
                conhl_tab.reshape(-1, 1),
                Mcenl_tab.reshape(-1, 1),
                elipcenl.reshape(-1, 1),
                polarcenl.reshape(-1, 1),
                tb_cen.reshape(-1, 1),
            )
        )

        halo_gal_pop_array = np.append(halo_gal_pop_array, halogal_par_mat, axis=0)

    columns_pop = [
        "z",
        "halo_mass",
        "halo_mass_acc",
        "e_h",
        "p_h",
        "concentration",
        "stellar_mass",
        "ellipticity",
        "p_g",
        "tb",
    ]
    table_pop = Table(halo_gal_pop_array, names=columns_pop)

    return table_pop
