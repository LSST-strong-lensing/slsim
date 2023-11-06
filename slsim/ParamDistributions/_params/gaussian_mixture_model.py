from pydantic import BaseModel, PositiveFloat, PositiveInt, model_validator, field_validator
import numpy as np


class GaussianMixtureModel(BaseModel):
    means: list[float] = [0.00330796, -0.07635054, 0.11829008]
    stds: list[PositiveFloat] = [
        np.sqrt(0.00283885),
        np.sqrt(0.01066668),
        np.sqrt(0.0097978),
    ]
    weights: list[PositiveFloat] = [0.62703102, 0.23732313, 0.13564585]

    @field_validator("weights")
    @classmethod
    def check_weights(cls, weight_values):
        if sum(weight_values) != 1:
            raise ValueError("The sum of the weights must be 1")
        return weight_values

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.means) != len(self.stds) or len(self.means) != len(self.weights):
            raise ValueError("The lengths of means, stds and weights must be equal")
        return self

class GaussianMixtureModel_rvs(BaseModel):
    size: PositiveInt