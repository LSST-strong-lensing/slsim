#!/usr/bin/env python

"""Tests for `sim_pipeline` package."""

import pytest
from sim_pipeline.sim_pipeline import draw_gaussian
import numpy as np
import numpy.testing as npt


from sim_pipeline import sim_pipeline


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest tests function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_draw_gaussian():
    """
    tests the function draw_gaussina

    :return:
    """
    for mean in np.linspace(-10, 10, 11):
        sigma = 1
        np.random.seed(41)
        drawn_numbers = draw_gaussian(mean=mean, sigma=sigma, num=10000)
        mean_drawn = np.mean(drawn_numbers)
        npt.assert_almost_equal(mean_drawn, mean, decimal=3)
        sigma_drawn = np.std(drawn_numbers)
        npt.assert_almost_equal(sigma_drawn, sigma, decimal=2)
