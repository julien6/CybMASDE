from setuptools import setup, find_packages

setup(
    name='custom_envs',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=["pillow", "gymnasium", "numpy", "pettingzoo", "pygame", "matplotlib"],
    include_package_data=True,
    package_data={"custom_envs": ["movingcompany/env/asset/*.png"]},
    author='Julien Soule',
    author_email='julien.soule@lcis.grenoble-inp.fr',
    description='Custom cooperative environments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/julien6/omarl_experiments',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)