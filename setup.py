import setuptools


def long_description():
    with open("README.md", "rb") as fh:
        long_description = fh.read().decode()
    return long_description


setuptools.setup(
    name="numpy-partition",
    version="1.18.9",
    author="partitionby",
    author_email="partitionby@protonmail.ch",
    description="SQL PARTITION BY and window functions for NumPy",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/partitionby/numpy-partition",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=(
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ),
)
