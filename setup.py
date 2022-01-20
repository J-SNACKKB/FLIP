from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('baselines/__init__.py', 'r') as f:
    init = f.readlines()

for line in init:
    if '__author__' in line:
        __author__ = line.split("'")[-2]
    if '__email__' in line:
        __email__ = line.split("'")[-2]
    if '__version__' in line:
        __version__ = line.split("'")[-2]

setup(
    name='baselines',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='Package for running FLIP baselines.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'biopython',
        'tqdm',
        'scipy', 
        ### FILL THIS OUT -- doesn't have to be extensive w/ environment file ###
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'train_all = baselines.train_all:main',
        ]
    },
)