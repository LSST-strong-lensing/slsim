__author__ = 'Paras Sharma'

# This file contains the class to simulate
# microlensing light curves
# It requires CUDA GPU to run, mainly because it uses
# (https://github.com/weisluke/microlensing) to generate magnification maps
# the light curves are generated using the magnification maps by
# using Amoeba (https://github.com/Henry-Best-01/Amoeba).


class Microlensing(object):
    """
    Class to simulate
    microlensing light curves
    """
    def __init__(self):
        pass