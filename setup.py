from setuptools import setup, find_packages

import sys
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.version_info.major != 3:
    print('This package is only compatible with Python 3, but you are running '
          'Python {}. The installation will likelyfail.'.format(sys.version_info.major))
          


setup(name='icassp20_T6',
          version='0.1.1',
          description='python code for tutorial 6 at ICASSP 2020',
	  long_description=read('README.md'),
          url='https://github.com/Mufabo/ICASSP2020_tutorial_6',
          author='M. Fatih Bostanci',
          author_email='fatih.bostanci@hotmail.de',
          license='GNU v3',
          packages= find_packages(),
          install_requires=[
            'numpy',
            'matplotlib',
            'pyclustering',
            'scipy',
            'setuptools',
            'statsmodels',
            'warnings'
            ],
          zip_safe=False)

