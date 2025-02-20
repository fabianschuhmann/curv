import os, sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import multiprocessing


setup(
    name='curv',
    version='1.0',
    packages=find_packages(include=['curv']),
    package_data={
        'curv': ['make_ndx', 'calculate','plot'],
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy','scipy','networkx','matplotlib','tqdm','glob','os','MDAnalysis'],
    entry_points={
        'console_scripts': [
            'curv=.run:main',
        ],
    },
)
