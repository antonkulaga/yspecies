try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.1.8',
      py_modules=['yspecies', 'dataset', 'workflow', 'utils', 'partition', "selection"],
      packages=find_packages(),
      description='yspecies')