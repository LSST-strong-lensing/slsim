from slsim.Sources.SourceTypes.source_base import SourceBase
class ExtendedSource(object):
    def __init__(self, source_dict, cosmo=None, **kwargs):
        self.cosmo = cosmo
