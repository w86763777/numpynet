from setuptools import setup


setup(name='numpynet',
      version='0.0.1',
      description='High level neural network api implemented in numpy',
      author='Alan Wu',
      author_email='w86763777@gmail.com',
      packages=[
          'numpynet', 'numpynet.dataset'],
      package_dir={
          'numpynet': 'numpynet',
          'numpynet.dataset': 'numpynet/dataset'},
      package_data={
          'numpynet.dataset': ['data/*.csv']},
      install_requires=[
          'numpy',
          'tqdm',
          'munch'])
