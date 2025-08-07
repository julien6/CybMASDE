from setuptools import setup, find_packages

# Fonction pour lire les dépendances depuis requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

setup(
    name="mma_wrapper",
    version="0.1.0",
    author="Julien Soule",
    author_email="julien.soule@lcis.grenoble-inp.fr",
    description="A package containing the MOISE-MARL API implementation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/julien6/moise-marl",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
