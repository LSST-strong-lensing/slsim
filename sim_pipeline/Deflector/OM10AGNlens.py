from gg_lens import GGLens
from sim_pipeline.Deflector.deflector_base import LensBase
import numpy as np

class OM10AGNLens(LensBase):
    """
    Class to manage individual galaxy-AGN lenses from OM10 Catalog
    OM10 Catalog: https://github.com/drphilmarshall/OM10/tree/master
    """

    def __init__(self, source_dict, deflector_dict, cosmo,
                 source_type='point_source',
                 test_area=4 * np.pi):
        
        super(GGLens, self).__init__(source_dict, deflector_dict, cosmo,
                 source_type=source_type,
                 test_area=4 * np.pi,
                 mixgauss_means=None, mixgauss_stds=None, mixgauss_weights=None)
        return
    
    
    


