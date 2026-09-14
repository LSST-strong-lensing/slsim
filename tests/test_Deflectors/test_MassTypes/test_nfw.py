from slsim.Deflectors.MassTypes.nfw import NFW
from slsim.Sources.source import Source


class TestNFW(object):

    def test_ellitpicty(self):
        light = Source(z=0.5)
        e1_, e2_ = 0.1, -0.1
        nfw = NFW(light=light, halo_mass=1e14, concentration=10, e1=e1_, e2=e2_)
        e1, e2 = nfw.ellipticity
        assert e1_ == e1
        assert e2_ == e2
