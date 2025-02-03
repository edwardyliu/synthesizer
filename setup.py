# setup for synthesizer

from setuptools import setup, find_packages

setup(
    name="synthesizer",
    version="1.0.0",
    packages=find_packages(include=["synthesizer", "synthesizer.*"]),
    install_requires=[
        "numpy==2.2.2",
        "pandas==2.2.3",
    ],
    setup_requires=["flake8", "pytest-runner"],
    tests_require=["pytest"],
)
