# setup for datasynth

from setuptools import setup, find_packages

setup(
    name="datasynth",
    version="1.0.0",
    packages=find_packages(include=["datasynth", "datasynth.*"]),
    install_requires=[
        "numpy==2.2.2",
        "pandas==2.2.3",
    ],
    setup_requires=["flake8", "pytest-runner"],
    tests_require=["pytest"],
)
