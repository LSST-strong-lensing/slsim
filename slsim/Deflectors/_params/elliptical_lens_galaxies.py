from pydantic import BaseModel, PositiveFloat


class vel_disp_from_m_star(BaseModel):
    m_star: PositiveFloat
