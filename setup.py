import os
from setuptools import setup, find_packages


__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "diffprof", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())

PACKAGENAME = "diffprof"


setup(
    name=PACKAGENAME,
    version=__version__,
    author=("Dash Stevanovich", "Andrew Hearin", "Erwin Lau", "Daisuke Nagai"),
    author_email="ahearin@anl.gov",
    description="Differentiable model of halo internal structure",
    long_description="Differentiable model of halo internal structure",
    install_requires=["numpy", "jax", "scipy", "astropy"],
    packages=find_packages(),
    url="https://github.com/BaryonPasters/diffprof",
    package_data={"diffprof": ("tests/testing_data/*.dat",)},
)
