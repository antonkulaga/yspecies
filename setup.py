try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.0.4',
      py_modules=['yspecies', 'enums', 'dataset', 'misc', 'utils'],
      packages=find_packages(),
      description='yspecies')