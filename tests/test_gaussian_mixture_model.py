import numpy as np
from slsim.ParamDistributions.gaussian_mixture_model import GaussianMixtureModel


def test_gaussian_mixture_model():
    model = GaussianMixtureModel()
    means = [0.00330796, -0.07635054, 0.11829008]
    stds = [np.sqrt(0.00283885), np.sqrt(0.01066668), np.sqrt(0.0097978)]
    weights = [0.62703102, 0.23732313, 0.13564585]

    # Check if model was initialized correctly
    assert model.means == means, "Model means do not match expected means."
    assert (
        model.stds == stds
    ), "Model standard deviations do not match expected standard deviations."
    assert model.weights == weights, "Model weights do not match expected weights."

    # Check if `rvs` function returns an array of the correct size
    size = 100
    samples = model.rvs(size)
    assert len(samples) == size, f"Expected {size} samples, got {len(samples)}."

    # Check if at least 80% of the samples are between -1 and 1
    within_range = np.sum((-1 <= samples) & (samples <= 1))
    assert (
        within_range / size >= 0.8
    ), "Less than 80% of samples are within the range (-1, 1)."
    f"Only {within_range / size * 100}% are within range."
