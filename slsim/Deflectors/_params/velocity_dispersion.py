from pydantic import BaseModel, PositiveFloat
from astropy.cosmology import Cosmology


class vel_disp_composite_model(BaseModel, arbitrary_types_allowed=True):
    r: PositiveFloat
    m_star: PositiveFloat
    rs_star: PositiveFloat
    m_halo: PositiveFloat
    c_halo: float
    cosmo: Cosmology
