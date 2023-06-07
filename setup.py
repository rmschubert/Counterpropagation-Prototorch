from setuptools import find_packages, setup

PROJECT_URL = "https://github.com/rmschubert/Counterpropagation-Prototorch"
DOWNLOAD_URL = "https://github.com/rmschubert/Counterpropagation-Prototorch.git"

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "prototorch>=0.7.4",
    "scipy",
    "torch-kmeans"
]


setup(
    name="counterprop_prototorch",
    version="1.0.2",
    description="Counter Propagation based on "
    "Prototorch. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ronny Schubert",
    author_email="trebuhcsynnor@gmail.com",
    url=PROJECT_URL,
    download_url=DOWNLOAD_URL,
    license="MIT",
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require={ 
        "dev": ["twine>=4.0.2"],
    },
    packages=find_packages(),
    zip_safe=False,
)