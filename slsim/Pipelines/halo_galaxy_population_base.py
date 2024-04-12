import sys
import astropy.constants as const
import lens_gals
# Definition of parameters


class HaloGalaxyPopulationBase(ABC):
    """Abstract Base Class to create samples of original population for the halo-galaxy compound model.

    All object that inherit from Lensed Sample must contain the methods it contains.
    """
    def __init__(
        self,
        sky_area=None,
        cosmo=None,
        z_min = None,
        z_max = None,
        log10host_halo_mass_min = None,
        log10host_halo_mass_max = None,
        log10subhalo_mass_min = 10.
        sigma_host_halo_concentration = 0.33  # Intrinsic scatter (https://arxiv.org/abs/astro-ph/0608157)
        sigma_subhalo_concentration = 0.33
        sigma_central_galaxy_mass = 0.2  # Intrinsic scatter (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
        sigma_satellite_galaxy_mass = 0.2
        sigma_tb = 0.46
    ):
        self.z_min = 0.01
        self.z_max = 5.0,
        self.log10host_halo_mass_min = 11.
        self.log10host_halo_mass_max = 16.
        self.log10subhalo_mass_min = 10.
        self.sigma_host_halo_concentration = 0.33  # Intrinsic scatter (https://arxiv.org/abs/astro-ph/0608157)
        self.sigma_subhalo_concentration = 0.33
        self.sigma_central_galaxy_mass = 0.2  # Intrinsic scatter (https://iopscience.iop.org/article/10.3847/1538-4357/ac4cb4/pdf)
        self.sigma_satellite_galaxy_mass = 0.2
        self.sigma_tb = 0.46 



        # method to calculate galaxy half-light effective size (or rb in Hernquist profile)
        # options: 'vdW23'(JWST-base), 'oguri20'(simple), 'karmakar23'(IllustrisTNG-base), (default 'vdW23')
        self.TYPE_GAL_SIZE = 'vdW23'

        # fraction of Stellar mass-to-light ratio in respect to Chabrier IMF
        # Chabrier: frac_SM_IMF=1.0, Salpeter: =1.715 (default 1.715)
        self.frac_SM_IMF = 1.715

        # type of fitting formula of the stellar-mass-halo-mass relation
        # options: 'true','true_all','obs', see Berhoozi et al. 2019 Table J1
        # default 'true'
        self.TYPE_SMHM = 'true'
        self.paramc, self.params = lens_gals.gals_init(self.TYPE_SMHM)


    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        # if name in self.__dict__:
        #     raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value


sys.modules[__name__] = HaloGalaxyPopulationBase()
