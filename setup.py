from setuptools import setup

setup(name='ElevationSampler',
      version='2.0',
      description='Sample elevation from DEM',
      url='https://github.com/tschopo/elevation_profile',
      author='Johannes Polster',
      author_email='johannes.polster@posteo.de',
      license='GPL-3.0',
      packages=['ElevationSampler'],
      install_requires=[
          'rasterio',
          'geopandas',
          'pandas',
          'numpy',
          'pyproj',
          'shapely',
          'scipy',
          'matplotlib'
      ],
      zip_safe=True)
