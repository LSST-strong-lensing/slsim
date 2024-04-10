import sys
import astropy.constants as const

# Definition of parameters


class _const:
    def __init__(self):
        self.cc = const.c.to("m s-1").value
        self.G = const.G.to("m3 kg^-1 s^-2").value
        self.G_MpcMsun = const.G.to("m2 Mpc Msun^-1 s^-2").value
        self.c2_G = self.cc**2/self.G_MpcMsun
        self.pc = const.pc.to("m").value
        self.Mpc = self.pc*1e6
        self.Msun = const.M_sun.to("kg").value
        self.kg_to_Msun = 1./self.Msun
        self.m_to_Mpc = 1/self.Mpc
        self.kpc_to_Mpc = 1.e-3
        self.cosmo_weos = -1.0
        self.nonflat = 0.0
        self.zsmax = 1.e3
        self.rt_range = 4.0    # Give plenty of surface to measure lensing event
        self.maxlev = 5
        # self.flag_h = 1
        self.flag_h = -1.0
        # self.flag_sh = 2

    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value


sys.modules[__name__] = _const()
