# ElevationSampler

## Description

Create elevation profiles of a LineString, by sampling from a DEM-raster. 

Features:
- Bicubic interpolation when sampling from the Raster (take into account 4x4 cells around Sample point)
- Smoothing of the elevation profile
- Pass segments that should be linearly interpolated (e.g. for Bridges / Tunnels)
- Adjust height for Forest / Urban areas
- Automatic reference system handling

## Install

```sh
git clone https://github.com/tschopo/elevation_profile.git
python3 -m pip install -e elevation_profile/
```

## Update

```sh
cd elevation_profile
git pull
```

## Examples

```python
from ElevationSampler import DEM
import pandas as pd
from shapely.geometry import  LineString

# load the DEM
elevation_model = DEM("DEM.tif")

# define a line to sample along / or wrap in a geopandas GeoSeries for crs handling
line = LineString([(2, 0), (2, 4), (3, 4)])

# sample every 10m along the line
sample_distance = 10
elevation_profile = elevation_model.elevation_profile(line, distance=sample_distance, interpolated=True)

# on the Line between 10 and 20 meters theres is a bridge
data = {'start_dist' : [10], 
        'end_dist':[20]} 
brunnels = pd.DataFrame(data) 

# TODO calc sample_distance automatically, if not equidistant raise error
elevation_profile = elevation_profile.interpolate_brunnels(brunnels, sample_distance)

# adjust the elevation profile where there is high variance (forests / urban areas)
elevation_profile = elevation_profile.to_terrain_model()

# smooth the elevation profile
elevation_profile = elevation_profile.smooth()

# resample the elevation profile to get value at every 22 meters instead 10
elevation_profile = elevation_profile.resample(22)

# get the inclination in degrees of the profile
inclination = elevation_profile.inclination(degrees=True)

# get the cumulative ascent of the profile
print(elevation_profile.cumulative_ascent())

distances = elevation_profile.distances
elevations = elevation_profile.elevations
 
```

## Dependencies
Should be automatically installed when installing with pip

```
'rasterio',
'geopandas',
'pandas',
'numpy',
'pyproj',
'shapely',
'scipy'
```
