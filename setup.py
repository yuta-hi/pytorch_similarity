#!/usr/bin/env python

from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='torch_similarity',
    version='1.0.0',
    description='Similarity Measure for PyTorch',
    long_description=open('README.md').read(),
    author='yuta-hi',
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    url='https://github.com/yuta-hi/pytorch_similarity',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
