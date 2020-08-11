try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.2.1',
      py_modules=['yspecies', 'dataset', 'workflow', 'utils', 'partition', "selection", "results", 'tuning', 'model'],
      packages=find_packages(),
      description='yspecies')