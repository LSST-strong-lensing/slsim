import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Util import param_util
from slsim.Sources.source_pop_base import SourcePopBase
from astropy.table import Column
from slsim.Util.param_util import average_angular_size, axis_ratio, eccentricity
from lenstronomy.Util import constants
import COSMOSCatalog


"""
Turn Real galaxy images from the COSMOS field survey into source light model objects for SLsim.
"""

class COSMOSImageSources(COSMOSCatalog, SourcePopBase):