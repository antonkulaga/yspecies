try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.0.3',
      py_modules=['yspecies'],
      packages=find_packages(),
      description='yspecies')