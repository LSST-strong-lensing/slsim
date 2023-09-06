import numpy as np


class GaussianMixtureModel:
    """A Gaussian Mixture Model (GMM) class.

    This class is used to represent a mixture of Gaussian distributions, each of which
    is defined by its mean, standard deviation and weight.
    """

    def __init__(self, means=None, stds=None, weights=None):
        """
        The constructor for GaussianMixtureModel class. The default values are the
        means, standard deviations, and weights of the fits to the data in the table
        2 of https://doi.org/10.1093/mnras/stac2235 and others.

        :param means: the mean values of the Gaussian components.
        :type means: list of float
        :param stds: The standard deviations of the Gaussian components.
        :type stds: list of float
        :param weights: The weights of the Gaussian components in the mixture.
        :type weights: list of float
        """
        if means is None:
            means = [0.00330796, -0.07635054, 0.11829008]
        if stds is None:
            stds = [np.sqrt(0.00283885), np.sqrt(0.01066668), np.sqrt(0.0097978)]
        if weights is None:
            weights = [0.62703102, 0.23732313, 0.13564585]
        assert (
            len(means) == len(stds) == len(weights)
        ), "Lengths of means, standard deviations, and weights must be equal."
        self.means = means
        self.stds = stds
        self.weights = weights

    def rvs(self, size):
        """Generate random variables from the GMM distribution.

        :param size: The number of random variables to generate.
        :type size: int
        :return: An array of random variables sampled from the GMM distribution.
        :rtype: np.array
        """
        components = np.random.choice(len(self.means), size=size, p=self.weights)
        return np.array(
            [
                np.random.normal(self.means[component], self.stds[component])
                for component in components
            ]
        )
