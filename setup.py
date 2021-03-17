from setuptools import setup
setup(name='ElevationSampler',
version='1.0',
description='Sample elevation from DEM',
url='#',
author='Johannes Polster',
author_email='johannes.polster@posteo.de',
license='MIT',
packages=['ElevationSampler'],
install_requires=[
          'rasterio',
          'geopandas',
          'pandas',
          'numpy',
          'pyproj',
          'shapely',
          'scipy'
      ],
zip_safe=False)

