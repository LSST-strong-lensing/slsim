class LOSIndividual(object):
    """Class to store the quantities of an individual line of sight."""

    def __init__(self, kappa=None, gamma=None, gamma1=None, gamma2=None):
        """

        :param gamma: [gamma1, gamma2] (takes these values if present)
        :type gamma: list of floats
        :param gamma1: gamma1 component
        :param gamma2: gamma2 component
        :param kappa: convergence (takes this values if present)
        :type kappa: float
        """
        if kappa is None:
            kappa = 0
        if gamma1 is None or gamma2 is None:
            if gamma is None:
                gamma = [0, 0]
        else:
            gamma = [gamma1, gamma2]
        self._kappa = kappa
        self._gamma = gamma

    @property
    def convergence(self):
        """Line of sight convergence.

        :return: kappa
        """
        return self._kappa

    @property
    def shear(self):
        """Line of sight shear.

        :return: gamma1, gamma2
        """
        return self._gamma[0], self._gamma[1]
