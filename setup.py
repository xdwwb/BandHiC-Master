from setuptools import setup, find_packages

# Optional: read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="bandhic",
    version="0.2.0",
    author="Weibing Wang",
    description="BandHiC: a memory-efficient Python package for managing and analyzing Hi-C data down to sub-kilobase resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.3",
        "scipy>=1.16",
        "cooler>0.10",
        "pandas>=2.3",
        "joblib>=1.2",
        "numba>=0.59",
    ],
)
