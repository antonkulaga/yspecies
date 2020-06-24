try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.1.1',
      py_modules=['yspecies', 'enums', 'dataset', 'workflow', 'misc'],
      packages=find_packages(),
      description='yspecies')