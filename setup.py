try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='yspecies',
      version='0.3.0',
      py_modules=['yspecies', 'config', 'dataset', 'explanations', 'helpers', 'models', 'partition', 'preprocess', 'selection', 'tuning', 'utils', 'workflow'],
      packages=find_packages(),
      description='yspecies')
