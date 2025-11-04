from setuptools import setup, find_packages

# Basic metadata
setup(
    name="cybmasde",
    version="0.1.0",
    author="Julien Soulé",
    author_email="julien.soule@univ-grenoble-alpes.fr",
    description="CybMASDE — A Multi-Agent System Design Environment combining MOISE+ and MARL",
    long_description=open(
        "README.md", encoding="utf-8").read() if __file__ else "",
    long_description_content_type="text/markdown",
    url="https://github.com/julien6/CybMASDE",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "jsonschema",
        "pyyaml",
        "argparse",
        "scipy",
        "tqdm",
        "gymnasium",  # if environments rely on it
        "marllib",    # if used in training phase
        "rllib",      # optional: Ray RLlib if needed
    ],
    entry_points={
        "console_scripts": [
            "cybmasde = cli.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
