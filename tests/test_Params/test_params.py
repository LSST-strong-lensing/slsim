from slsim.ParamDistributions.gaussian_mixture_model import GaussianMixtureModel

a = GaussianMixtureModel(means=[1, 2, 3], stds=[1, 2, 3], weights=[0.4, 0.4, 0.2])
print(a.stds)
