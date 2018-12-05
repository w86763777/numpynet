from setuptools import setup

setup(name='numpynet',
      version='0.0.1',
      description='High level neural network api implemented in numpy',
      author='Alan Wu',
      author_email='w86763777@gmail.com',
      packages=[
          'numpynet'],
      install_requires=[
          'tqdm',
          'munch'],
      setup_requires=[
          'tqdm',
          'munch'])
