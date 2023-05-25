import numpy as np
class GaussianMixtureModel:
    """
        A Gaussian Mixture Model (GMM) class.
        This class is used to represent a mixture of Gaussian distributions,
        each of which is defined by its mean, standard deviation and weight.
        """
    def __init__(self, means, stds, weights):
        """

        The constructor for GaussianMixtureModel class.

        :param means: the mean values of the Gaussian components.
        :type means: list of float
        :param stds: The standard deviations of the Gaussian components.
        :type stds: list of float
        :param weights: The weights of the Gaussian components in the mixture.
        :type weights: list of float
        """
        assert len(means) == len(stds) == len(
            weights), 'Lengths of means, standard deviations, and weights must be equal.'
        self.means = means
        self.stds = stds
        self.weights = weights

    def rvs(self, size):
        """
        Generate random variables from the GMM distribution.

        :param size: The number of random variables to generate.
        :type size: int

        :return: An array of random variables sampled from the GMM distribution.
        :rtype: np.array
        """
        components = np.random.choice(len(self.means), size=size, p=self.weights)
        return np.array([np.random.normal(self.means[component], self.stds[component]) for component in components])

