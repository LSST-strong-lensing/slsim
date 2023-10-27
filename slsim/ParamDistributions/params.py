from pydantic import BaseModel, Field, PositiveFloat, model_validator
import numpy as np

class GaussianMixtureModel(BaseModel):
    means: list[float] = [0.00330796, -0.07635054, 0.11829008]
    stds: list[PositiveFloat] = [np.sqrt(0.00283885), np.sqrt(0.01066668), np.sqrt(0.0097978)]
    weights: list[PositiveFloat] = [0.62703102, 0.23732313, 0.13564585]
    
    @model_validator(mode="after")
    def check_lenghts(self):
        if len(self.means) != len(self.stds) or len(self.means) != len(self.weights):
            raise ValueError("The lenghts of means, stds and weights must be equal")
        return self
    