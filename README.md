# ElevationSampler

## Description

Create elevation profiles of a LineString, by sampling from a DEM-raster. 

Features:
- Bicubic interpolation when sampling from the Raster (take into account 4x4 cells around Sample point)
- Smoothing of the elevation profile
- Pass segments that schould be linearly interpolated (e.g. for Bridges / Tunnels)
- Adjust height for Forest / Urban areas
- Automatic reference system handling

## Install
run 
```
python3 -m pip install -e elevation_profile/
```

## Examples

```
# load the DEM
elevation_sampler = ElevationSampler("DEM.tif")

# define a line to sample along / or wrap in a geopandas GeoSeries for crs handling
line = LineString([(2, 0), (2, 4), (3, 4)])

# sample every 10m along the line
sample_distance = 10
x_coords, y_coords, distances, elevation = elevation_sampler.elevation_profile(line, distance=sample_distance, interpolated = True)

# on the Line between 10 and 20 meters theres is a bridge
data = {'start_dist' : [10], 
        'end_dist':[20]} 
brunnels = pd.DataFrame(data) 

ele_brunnel = elevation_sampler.interpolate_brunnels(elevation, distances, brunnels, sample_distance)

# adjust the elevation profile where there is high variance (forests / urban areas)
ele_adjusted = elevation_sampler.adjust_forest_height(ele_brunnel)

# smooth the elevation profile
ele_smoothed = elevation_sampler.smooth_ele(ele_adjusted)

# resample the elevation profile to get value at every 22 meters instead 10
distances_22, elevation_22 = elevation_sampler.resample_ele(ele_adjusted,distances,22)

# get the inclination in degrees of the profile
incl_22 = elevation_sampler.ele_to_incl(elevation_22, distances_22, degrees=True)
 
```
