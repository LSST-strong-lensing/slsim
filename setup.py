#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="DESC/SLSC",
    author_email="sibirrer@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="LSST strong lensing simulation pipeline",
    install_requires=requirements,
    extras_require={"Halos": ["hmf"]},
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="slsim",
    name="slsim",
    packages=find_packages(include=["slsim", "slsim.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/LSST-strong-lensing/slsim",
    version="0.1.0",
    zip_safe=False,
)
