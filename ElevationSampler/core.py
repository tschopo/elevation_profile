import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy import interpolate
from shapely.geometry import Point
from scipy.signal import savgol_filter

class ElevationSampler:

    def __init__(self, dem, elevation_band=1):
        """
        Parameters
        ----------
            dem : str or rasterio raster object
                location to geotiff with elevation data
        """
        if isinstance(dem, str):
            dem = rasterio.open(dem)
        self.dem = dem
        self.elev = dem.read(1)
        self.dem_crs = CRS.from_wkt(dem.crs.to_wkt())

        print("Loaded dem as EPSG:"+str(self.dem_crs.to_epsg()))

    def sample_point(self, point, interpolated=True):
        """
        Parameters
        ----------
            point : Point
                must be same crs as dem
            interpolated : bool
                default True. If True then the elevation is bicubic interpolated.
        
        Returns
        -------
            elevation : float
        """
        
        p_x = point.x
        p_y = point.y

        return self.sample_coords(p_x, p_y, interpolated=interpolated)
    
    def sample_coords(self, p_x, p_y, interpolated=True):
        """
        Parameters
        ----------
            p_x : float
                x coordinate / longitude
            p_y : float
                y coordinate / latitude
        
        Returns
        -------
            elevation : float
                elevation at p_x, p_y
        """
        
        # get the index of the raster pixel containing the point
        row,col = self.dem.index(p_x, p_y)

        if not interpolated:
            return self.elev[row,col]

        # get raster pixel center
        r_x, r_y = self.dem.xy(row,col)

        # get the correct surrounding raster pixels, depending on point location in raster cell
        if p_x <= r_x and p_y <= r_y:
            row_from = -1
            row_to = 2
            col_from = -2
            col_to = 1
        if p_x >= r_x and p_y <= r_y:
            row_from = -1
            row_to = 2
            col_from = -1
            col_to = 2
        if p_x >= r_x and p_y >= r_y:
            row_from = -2
            row_to = 1
            col_from = -1
            col_to = 2
        if p_x <= r_x and p_y >= r_y:
            row_from = -2
            row_to = 1
            col_from = -2
            col_to = 1

        row_from += row
        row_to += row
        col_from += col
        col_to += col

        # the 16 supporting points of the interpolattion
        z = self.elev[row_from:row_to+1,col_from:col_to+1]
        z = z.flatten()

        # get the coordinates for each supporintg point
        x_coors = []
        y_coors = []
        for row in range(row_from, row_to + 1):
            for col in range(col_from, col_to + 1):
                x,y = self.dem.xy(row,col) 
                x_coors.append(x)
                y_coors.append(y)

        # 5. interpolate cubic with scipy
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
        f = interpolate.interp2d(x_coors, y_coors, z,kind='cubic')

        e = f(p_x,p_y)[0]

        return e

    def	elevation_profile(self, line, distance=10, interpolated=True):
        """
        Parameters
        ----------
            line : LineString or GeoSeries
                either shapely linestring, must be same crs as dem, or geopandas series with 1 linestring entry the crs is converted automatically
            distance : int
                default 10. The distance between the sample points on the line. Last distance may be shorter.
            interpolated : bool
                if True, then the elevation is bicubic interpolated
        
        Returns
        -------
            x_coords, y_coords, distance from start, elevations or interpolated_elevations
                x and y coords in CRS of dem
        """
        
        old_crs = None
        if isinstance(line, gpd.GeoSeries):
            if line.crs.to_epsg() != self.dem_crs.to_epsg():
                old_crs = line.crs
                line = line.to_crs(self.dem_crs)

            line = line.iloc[0]
            
        # 3. process the line to obtain evenly spaced sample points along the line
        # https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
        distances = np.arange(0, line.length, distance)
        sample_points = [line.interpolate(d) for d in distances] + [line.boundary[1]]
        
        sample_point_x_coords = []
        sample_point_y_coords = []
        sample_point_elevation = []

        for sample_point in sample_points:
            p_x = sample_point.x
            p_y = sample_point.y

            sample_point_x_coords.append(p_x)
            sample_point_y_coords.append(p_y)
            sample_point_elevation.append(self.sample_coords(p_x,p_y,interpolated=interpolated))
        
        return (sample_point_x_coords, sample_point_y_coords, np.append(distances, line.length), sample_point_elevation)
    
    def interpolate_brunnels_old(self, elevation, distances, brunnels, trip_geom, distance_delta=10, merge_distance=10, filter_brunnel_length=10, buffer_factor=1):
        """
        Parameters
        ----------
            elevation : numpy array
                The elevation values
            distances : numpy array
                The distances of the elevation values, as returned by method elevation_profile
            brunnels : GeoDataFrame 
                Dataframe of Tunnel and Bridges with Linestring of brunnel in "geom" Column. 
            trip_geom : GeoSeries
                Geoseries containing the trip Linestring in the "geom" column
            distance_delta : int
                distance_delta between distances
            merge_distance : float
                Brunnels that are below this distance apart will be merged to one
            filter_brunnel_length : float
                Brunnels that are smaller than this will be ignored
            buffer_factor : float
                sample distance before and after brunnel is calculated as buffer_factor*distance_delta
                
        Returns
        -------
            elevation array where brunnels are linearly interpolted
        """
        
        if brunnels.crs.to_epsg() != self.dem_crs.to_epsg():
            brunnels.to_crs(self.dem_crs, inplace=True)
            
        if trip_geom.crs.to_epsg() != self.dem_crs.to_epsg():
            trip_geom = trip_geom.to_crs(self.dem_crs)
        
        # add start_dist, end_dist to brunnel GeoDataFrame
        brunnels['start_dist'] = np.nan
        brunnels['end_dist'] = np.nan

        for index, brunnel in brunnels.iterrows():

            start_point = Point(brunnel.geom.coords[0])
            end_point = Point(brunnel.geom.coords[-1])

            start_dist = trip_geom.project(start_point).iloc[0]
            end_dist = trip_geom.project(end_point).iloc[0]

            # make brunnels all same direction
            if start_dist > end_dist:
                tmp = start_point
                start_point = end_point
                end_point = tmp

            # calculate distance with flipped point new
            brunnels.loc[index,'start_dist'] = trip_geom.project(start_point).iloc[0]
            brunnels.loc[index,'end_dist'] = trip_geom.project(end_point).iloc[0]
            
        # sort by start_dist
        brunnels.sort_values(by=['start_dist'], inplace=True)
        brunnels.reset_index(drop=True, inplace=True)

        drop_idx = []
        for index, brunnel in brunnels.iterrows():

            # merge adjacent brunnels
            # + buffer
            # merge by deleting current row and adjusting values of previous row
            #print(index)
            if index > 0 and  brunnel['start_dist'] <= brunnels.loc[index-1,'end_dist'] + merge_distance:
                brunnels.loc[index-1,'end_dist'] = brunnel['end_dist']
                brunnels.loc[index-1,'geom'] = LineString(brunnels.loc[index-1,"geom"].coords[:] + brunnel["geom"].coords[:])

                brunnels.loc[index,'start_dist'] = brunnels.loc[index-1,'start_dist']
                brunnels.loc[index,'geom'] = LineString(brunnels.loc[index-1,"geom"].coords[:] + brunnel["geom"].coords[:])
                drop_idx.append(index)
                
        brunnels.drop(drop_idx, inplace=True)
        
        # filter the super small tunnels
        brunnels = brunnels[brunnels.length > filter_brunnel_length]

        start_dists = brunnels['start_dist'].values
        end_dists = brunnels['end_dist'].values

        ele_brunnel = elevation.copy()
        for i,x in enumerate(distances):

            # if x in brunnel
            # get index of brunnel
            idx = np.argwhere((x >= start_dists-buffer_factor*distance_delta)  & (x <= end_dists+buffer_factor*distance_delta))

            assert idx.size <= 1

            if idx.size == 1:

                idx = idx[0][0]

                # get index of elevation data
                start_idx = round((start_dists[idx])/distance_delta) - buffer_factor
                end_idx = round((end_dists[idx])/distance_delta) + buffer_factor
                
                start_ele = None
                end_ele = None

                # if trip doesnt start with brunnel 
                if start_idx > 0:
                    start_ele = ele_brunnel[start_idx]

                # if trip doesnt end with brunnel:
                if end_idx < len(ele_brunnel) - 1:
                    end_ele = ele_brunnel[end_idx]

                # if trip start with brunnel
                if start_ele == None:
                    start_ele = end_ele

                # if trip ends with brunnel
                if end_ele == None:
                    end_ele = start_ele

                # if trip is completely brunnel
                if start_ele == None and end_ele == None:

                    # then take ele at start and ele at end as elevations
                    start_idx = 0
                    end_idx = round(end_dists[-1]/distance_delta)
                    start_ele = ele_brunnel[start_idx]
                    end_ele = ele_brunnel[end_idx]

                assert start_ele != None
                assert end_ele != None

                # linearly interpolate between start and end point
                # take into account buffer
                p1 = (start_dists[idx]-buffer_factor*distance_delta,start_ele)
                p2 = (end_dists[idx]+buffer_factor*distance_delta,end_ele)

                m = (p2[1] - p1[1])/ (p2[0]-p1[0])
                c = p1[1] - m*p1[0]
                ele_brunnel[i] = m*x+c

        return ele_brunnel

    def interpolate_brunnels(self, elevation, distances, brunnels, distance_delta=10):
        """
        Parameters
        ----------
            elevation : numpy array
                The elevation values
            distances : numpy array
                The distances of the elevation values, as returned by method elevation_profile
            brunnels : DataFrame 
                Dataframe of start_dist and end_dist for each section that shoud be linearly interpolated
            distance_delta : int
                distance_delta between distances
        
        Returns
        -------
            elevation array where brunnels are linearly interpolted
        """
        
        start_dists = brunnels['start_dist'].values
        end_dists = brunnels['end_dist'].values

        ele_brunnel = elevation.copy()
        for i,x in enumerate(distances):

            # if x in brunnel
            # get index of brunnel
            idx = np.argwhere((x >= start_dists-distance_delta)  & (x <= end_dists+distance_delta))

            assert idx.size <= 1

            if idx.size == 1:

                idx = idx[0][0]

                # get index of elevation data
                start_idx = round((start_dists[idx])/distance_delta) - 1
                end_idx = round((end_dists[idx])/distance_delta) + 1
                
                start_ele = None
                end_ele = None

                # if trip doesnt start with brunnel 
                if start_idx > 0:
                    start_ele = ele_brunnel[start_idx]

                # if trip doesnt end with brunnel:
                if end_idx < len(ele_brunnel) - 1:
                    end_ele = ele_brunnel[end_idx]

                # if trip start with brunnel
                if start_ele == None:
                    start_ele = end_ele

                # if trip ends with brunnel
                if end_ele == None:
                    end_ele = start_ele

                # if trip is completely brunnel
                if start_ele == None and end_ele == None:

                    # then take ele at start and ele at end as elevations
                    start_idx = 0
                    end_idx = round(end_dists[-1]/distance_delta)
                    start_ele = ele_brunnel[start_idx]
                    end_ele = ele_brunnel[end_idx]

                assert start_ele != None
                assert end_ele != None

                # linearly interpolate between start and end point
                # take into account buffer
                p1 = (start_dists[idx]-distance_delta,start_ele)
                p2 = (end_dists[idx]+distance_delta,end_ele)

                m = (p2[1] - p1[1])/ (p2[0]-p1[0])
                c = p1[1] - m*p1[0]
                ele_brunnel[i] = m*x+c

        return ele_brunnel
    
    def adjust_forest_height(self, elevation, window_size = 12, std_thresh = 3, sub_factor = 3, clip = 20):
        """
        Compute a rolling standard deviation and substract height in areas with high std
        
        Parameters
        ----------
            elevation : numpy array
                the elevation data
            window_size : int
            std_thresh : float
                if std above this value then substract
            sub_factor: float
                the std is multiplied by this factor and subtracted
            clip: float
                the maximm value that is subtracted
        
        Returns
        -------
            elevation : numpy array
                the adjusted elevation
                
        """

        elevation=pd.Series(elevation)
        t = elevation.rolling(window_size).std()
        elevation[t > std_thresh] = elevation[t > std_thresh] - np.clip(t[t > std_thresh] * sub_factor,0,clip)

        return elevation.values

    def smooth_ele(self, elevation, window_size=301, poly_order=3):
        return savgol_filter(elevation, window_size, poly_order)
    
    def resample_ele(self, elevation, distances, distance):
        """
        Resamples the elevation every n distance. (last elevation is also returned)
        
        Parameters
        ----------
        
            elevation : numpy array
            distances : numpy array
            distance : float
            
        Returns
        -------
            (numpay array, numpy array)
                the distances and resampled elevations
        """
        
        distances_interpolated = np.arange(0, distances[-1], 100)
        
        if distances_interpolated[-1] <= distances[-1]:
            distances_interpolated = np.append(distances_interpolated,distances[-1])
        
        elevation_interpolated = np.interp(distances_interpolated, distances, elevation)
        
        return (distances_interpolated, elevation_interpolated)
    
        # return subset of elevation, resampled at distance
        
    def ele_to_incl(self, elevation, distances, degrees=False):
        """
        Parameters
        ----------
            elevation : list like
            distances : list like
        
        Returns
        -------
            Numpy array
                Inclination in promille or degrees. n-1 points returned.
        """
        
        slopes = []

        # https://www.omnicalculator.com/construction/elevation-grade
        for i in range(len(elevation)-1):
            rise = elevation[i+1] - elevation[i]
            run = distances[i+1] - distances[i]

            if degrees:
                slopes.append(np.arctan(rise/run))
            else:
                slopes.append(rise/run  * 1000)            
            
        return np.array(slopes)         
            
        
    # def cum_asc_desc
        # return cumulative ascent and descent