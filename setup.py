import os
import configparser 
from setuptools import setup, find_packages


       
setup(
    name='plot_utils',
    version='0.1.0',
    maintainer='Mark Lescroart',
    packages=find_packages(),
    description="Useful plotting functions; mostly wraps matplotlib",
    long_description=open('README.md').read(),
    url='https://github.com/piecesofmindlab/plot_utils',
    download_url='https://github.com/piecesofmindlab/plot_utils',
)


