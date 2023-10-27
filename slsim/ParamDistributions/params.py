from pydantic import BaseModel
from pydantic import Field
import numpy as np

class GaussianMixtureModel(BaseModel):
    means: list[float] = [0.00330796, -0.07635054, 0.11829008]
    stds: list[float] = [np.sqrt(0.00283885), np.sqrt(0.01066668), np.sqrt(0.0097978)]
    weights: list[float] = [0.62703102, 0.23732313, 0.13564585]
    