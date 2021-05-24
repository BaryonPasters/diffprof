from setuptools import setup, find_packages


PACKAGENAME = "diffprof"
VERSION = "0.0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=("Dash Stevanovich", "Andrew Hearin", "Erwin Lau", "Daisuke Nagai"),
    author_email="ahearin@anl.gov",
    description="Differentiable model of halo internal structure",
    long_description="Differentiable model of halo internal structure",
    install_requires=["numpy", "jax", "scipy"],
    packages=find_packages(),
    url="https://github.com/BaryonPasters/diffprof",
    package_data={"diffprof": ("tests/testing_data/*.dat",)},
)
