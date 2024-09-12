from setuptools import setup, find_packages

setup(
    name='annotation_utilities',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'shapely',
        'numba',
        'numpy',
        'matplotlib'
    ],
)
