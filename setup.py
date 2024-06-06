"""
setup.py
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

necessary_require = [
    "rocketpy",
    "numpy>=1.13",
    "scipy>=1.0",
    "matplotlib>=3.0",
    "netCDF4>=1.6.4",
    "requests",
    "pytz",
    "simplekml",
]

env_analysis_require = [
    "timezonefinder",
    "windrose>=1.6.8",
    "IPython",
    "ipywidgets>=7.6.3",
    "jsonpickle",
]

setuptools.setup(
    name="elara-simulations",
    version="1.0.0",
    install_requires=necessary_require,
    extras_require={
        "env_analysis": env_analysis_require,
        "all": necessary_require + env_analysis_require,
    },
    maintainer="Elara Aerospace team",
    author="Suriyaa Sundararuban",
    author_email="github@suriyaa.tk",
    description="Python-powered trajectory simulation for Elara Aerospace Rocketry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elara-aerospace/rocket-simulations",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
