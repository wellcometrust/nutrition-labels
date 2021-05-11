import os

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="Nutrition Labels",
    version="0.0.1",
    author="Liz Gallagher",
    author_email="e.gallagher@wellcome.ac.uk",
    description="Train a model to predict whether a grant contains tech or not",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="",
    packages=setuptools.find_packages(
        include=["nutrition_labels", "representation_labels"],
        exclude=["notebooks", "tests", "build", "data", "models"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    tests_require=[
        "pytest"
    ]
)