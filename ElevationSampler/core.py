from __future__ import annotations

from typing import Union, Tuple, List, Optional

import geopandas as gpd
import pandas as pd
import rasterio
from geopandas import GeoSeries
from numpy import ndarray
from pandas import DataFrame
from pyproj import CRS
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from shapely.geometry import LineString
from shapely.geometry import Point
import numpy as np
from matplotlib import pyplot as plt


class ElevationProfile:

    def __init__(self, distances: ndarray, elevations: ndarray):
        self.distances = np.array(distances)
        self.elevations = np.array(elevations)
        self.distances_orig = np.array(distances)
        self.elevations_orig = np.array(elevations)

    def elevation(self, distance: float) -> float:
        """
        Returns the elevation at distance.

        Parameters
        ----------
        distance

        Returns
        -------

        """

        elevation = np.interp(distance, self.distances, self.elevations)

        return elevation

    def get_elevations(self) -> ndarray:
        """
        Returns a copy of the elevations array.

        Returns
        -------

        """
        return self.elevations.copy()

    def get_distances(self) -> ndarray:
        """
        Returns a copy of the distances array.

        Returns
        -------

        """

        return self.distances.copy()

    def _check_distance_delta(self, distance_delta: float) -> float:

        if distance_delta is None:
            # check if equidistant
            if np.sum(self.distances[:-2] - self.distances[1:-1]) > 0.1:
                raise Exception("Distances must be equidistant. Resample first!")

            distance_delta = self.distances[1] - self.distances[0]

        return distance_delta

    def interpolate_brunnels(self, brunnels: DataFrame, distance_delta: Optional[float] = None,
                             construct_brunnels: bool = True, max_brunnel_length: float = 300,
                             construct_brunnel_thresh: float = 5, diff_kernel_dist: int = 6) -> ElevationProfile:
        """
        Linearly interpolate between start and endpoint where there are tunnels of bridges.
        Automatically construct bridges over steep valleys and tunnels through steep mountains.

        Parameters
        ----------

            brunnels : DataFrame
                Dataframe of start_dist and end_dist for each section that should be linearly interpolated
            distance_delta : float
                distance_delta between distances, if None then the distance_delta is calculated from distances
            construct_brunnels : bool
                if True, then add brunnels in steep areas
            max_brunnel_length : float
                The maximum length of a constructed brunnel
            construct_brunnel_thresh : float
                The elevation delta from one sample point to the next at which a brunnel is attempted to be constructed
            diff_kernel_dist : int
                The sample point distance at which the difference is computed for constructing brunnels. E.g. if
                diff_kernel_dist = 2 then the difference is not taken from the next, but from the one after the next.

        Returns
        -------
            numpy array
            elevation array where brunnels are linearly interpolated
        """

        elevation = self.elevations.copy()
        distances = self.distances.copy()
        brunnels = brunnels.copy()

        distance_delta = self._check_distance_delta(distance_delta)

        if brunnels.shape[0] == 0 and not construct_brunnels:
            return self

        # construct brunnels in steep regions
        if construct_brunnels:

            # construct brunnels in steep regions

            diff_kernel = [1] + [0 for _ in range(diff_kernel_dist - 1)] + [-1]
            # diff_kernel = np.array([1,0, -1])
            diff = np.convolve(np.array(elevation), diff_kernel, 'same')

            start_dists = []
            end_dists = []
            brunnel_types = []

            i = 0
            while i < len(elevation) - 1:

                # bridge when downhill
                if diff[i] < (construct_brunnel_thresh * (-1)):

                    # get maximum in the next 200m
                    max_i = min(i + 1 + int(max_brunnel_length / distance_delta), len(elevation))
                    # print(i, max_i, distances[i], distances[max_i])

                    # DEBUG
                    if i + 1 >= max_i:
                        print(i + 1, max_i, len(elevation) - 1)

                    assert (i + 1 < max_i)

                    # get the idx if the maximum in the next 200m
                    max_idx = np.argmax(elevation[i + 1:max_i])
                    max_idx = max_idx + i + 1
                    # print("max_idx ", max_idx)
                    # print("------")
                    start_dists.append(distances[i])
                    end_dists.append(distances[max_idx])
                    brunnel_types.append("bridge")

                # tunnel if incline
                elif diff[i] > construct_brunnel_thresh:

                    max_i = min(i + 1 + int(max_brunnel_length / distance_delta), len(elevation))

                    assert (i + 1 < max_i)

                    # get minimum in the next 200m
                    max_idx = np.argmin(elevation[i + 1:max_i])
                    max_idx = max_idx + i + 1
                    start_dists.append(distances[i])
                    end_dists.append(distances[max_idx])
                    brunnel_types.append("tunnel")

                i += 1

            # merge overlapping brunnels

            # overlaps if end_point is larger than start point of next brunnel
            # then also check if next brunnel overlaps (multiple consecutive overlapping brunnels)
            # print(start_dists)
            # print(end_dists)

            start_dists, end_dists = _filter_overlapping(start_dists.copy(), end_dists.copy())

            brunnel_types = ["constructed" for _ in start_dists]

            data = {"brunnel": brunnel_types, "start_dist": start_dists, "end_dist": end_dists,
                    "length": np.array(end_dists) - np.array(start_dists)}
            constructed_brunnels = pd.DataFrame(data)

            # filter small brunnels
            constructed_brunnels = constructed_brunnels[constructed_brunnels.length > distance_delta]

            # check if constructed brunnel overlaps with real brunnel
            # if overlaps --> discard

            drop_idx = []
            for idx, brunnel in constructed_brunnels.iterrows():
                start_in_brunnel = (brunnel.start_dist >= brunnels['start_dist']) & (
                        brunnel.start_dist <= brunnels['end_dist'])
                end_in_brunnel = (brunnel.end_dist >= brunnels['start_dist']) & (
                        brunnel.end_dist <= brunnels['end_dist'])

                # chick if constructed a brunnel around an existing one
                # check if start is smaller than start and end ist larger than end
                around_brunnel = (brunnel.start_dist <= brunnels['start_dist']) & (
                        brunnel.end_dist >= brunnels['end_dist'])

                if np.sum(start_in_brunnel | end_in_brunnel | around_brunnel) > 0:
                    drop_idx.append(idx)

            constructed_brunnels = constructed_brunnels.drop(drop_idx)

            # merge with other brunnels and sort
            brunnels = pd.concat([brunnels, constructed_brunnels], ignore_index=True)

            brunnels = brunnels.sort_values("start_dist")
            brunnels = brunnels.reset_index(drop=True)

        if brunnels.shape[0] == 0:
            return self

        start_dists = brunnels['start_dist'].values
        end_dists = brunnels['end_dist'].values

        ele_brunnel = elevation.copy()
        for i, x in enumerate(distances):

            # if x in brunnel
            # get index of brunnel
            idx = np.argwhere((x >= start_dists) & (x <= end_dists))

            assert idx.size <= 1

            # DEBUG
            if idx.size > 1:
                print(x, idx, len(end_dists), brunnels.shape)
                print(brunnels.iloc[idx[0]])
                print(brunnels.iloc[idx[1]])

            if idx.size == 1:

                idx = idx[0][0]

                # get index of elevation data
                start_idx = np.int64(round((start_dists[idx]) / distance_delta) - 1)
                end_idx = np.int64(round((end_dists[idx]) / distance_delta) + 1)

                start_ele = None
                end_ele = None

                # if trip doesnt start with brunnel
                if start_idx > 0:
                    if type(start_idx) != np.int64 and \
                            type(start_idx) != np.int32 and \
                            type(start_idx) != int:
                        print(start_idx)
                        print(ele_brunnel.shape)
                        print(type(start_idx))
                    start_ele = ele_brunnel[start_idx]

                # if trip doesnt end with brunnel:
                if end_idx < len(ele_brunnel) - 1:
                    end_ele = ele_brunnel[end_idx]

                # if trip start with brunnel
                if start_ele is None:
                    start_ele = end_ele

                # if trip ends with brunnel
                if end_ele is None:
                    end_ele = start_ele

                # if trip is completely brunnel
                if start_ele is None and end_ele is None:
                    # then take ele at start and ele at end as elevations
                    start_idx = 0
                    end_idx = round(end_dists[-1] / distance_delta)
                    start_ele = ele_brunnel[start_idx]
                    end_ele = ele_brunnel[end_idx]

                assert start_ele is not None
                assert end_ele is not None

                # linearly interpolate between start and end point
                # take into account buffer
                p1 = (start_dists[idx] - distance_delta, start_ele)
                p2 = (end_dists[idx] + distance_delta, end_ele)

                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]
                ele_brunnel[i] = m * x + c

            self.elevations = ele_brunnel
        return self

    def to_terrain_model(self, method: str = "variance",
                         window_size: int = 12, std_thresh: float = 3, sub_factor: float = 2,
                         clip: float = 20, minimum_loops: int = 1) -> ElevationProfile:
        """
        Adjusts for objects on the surface.

        Parameters
        ----------

            method : "variance" or "minimum"
                if "variance": subtracts the variance in areas with high standard deviation: should use in conjunction
                with smooth
                if "minimum": fits a function through the local minima. must supply distances array
            window_size : int
                window size used for the rolling window function for std
            std_thresh : float
                if std above this value then subtract
            sub_factor: float
                the std is multiplied by this factor and subtracted
            clip: float
                the maximum value that is subtracted
            minimum_loops
                how often the minima a resampled.

        Returns
        -------
            elevation : numpy array
                the adjusted elevation

        """
        elevation = self.elevations.copy()
        distances = self.distances.copy()

        if method == "variance":
            elevation = pd.Series(elevation)
            t = elevation.rolling(window_size).std()
            elevation[t > std_thresh] = elevation[t > std_thresh] - np.clip(t[t > std_thresh] * sub_factor, 0, clip)
            elevation = elevation.values
        elif method == "minimum":
            for _ in range(minimum_loops):
                minima = argrelextrema(elevation, np.less)

                # if there is only 1 or less minima, this method doesn't make sense
                if len(minima) < 2:
                    return self

                elevation = ElevationProfile(distances[minima], elevation[minima]).resample(distances).elevations
                # _, elevation = ElevationSampler.resample_ele(distances)

        else:
            raise Exception("must supply either method='variance' or method='minimum'")

        self.elevations = elevation

        return self

    def smooth(self, window_size: int = 301, poly_order: int = 3, mode: str = "nearest") -> ElevationProfile:
        self.elevations = savgol_filter(self.elevations, window_size, poly_order, mode=mode)
        return self

    def resample(self, distance: Union[float, ndarray]) -> ElevationProfile:
        """
        Resamples the elevation every n distance. (last elevation is also returned)

        Parameters
        ----------

            distance : float or numpy array
                the distance values from which the resampled elevation values are drawn from
                if float equidistant resampling.
        Returns
        -------
        """

        elevation = self.elevations.copy()
        distances = self.distances.copy()

        if np.isscalar(distance):
            distances_interpolated = np.arange(0, distances[-1], distance)
            distances_interpolated = np.append(distances_interpolated, distances[-1])
        elif type(distance) is np.ndarray:
            distances_interpolated = distance
        else:
            raise Exception("distance must be float or Numpy Array!")

        elevation_interpolated = np.interp(distances_interpolated, distances, elevation)

        self.elevations = elevation_interpolated
        self.distances = distances_interpolated

        return self

    def inclination(self, degrees: bool = False) -> ndarray:
        """
        Parameters
        ----------
            degrees : bool
                if True returns inclination in degrees, if False in per mille.

        Returns
        -------
            Numpy array
                Inclination in promille or degrees. n-1 points returned (inclination between the distances).
        """
        elevation = self.elevations
        distances = self.distances

        slopes = []

        # https://www.omnicalculator.com/construction/elevation-grade
        for i in range(len(elevation) - 1):
            rise = elevation[i + 1] - elevation[i]
            run = distances[i + 1] - distances[i]

            if degrees:
                slopes.append(np.arctan(rise / run))
            else:
                slopes.append(rise / run * 1000)

        return np.array(slopes)

    def cumulative_ascent(self, ascending=True) -> float:

        # subtract with next elevation
        ele_diff = self.elevations[1:] - self.elevations[:-1]
        asc = ele_diff.copy()

        if ascending:
            asc[asc < 0] = 0
        else:
            asc[asc > 0] = 0

        return np.sum(asc)

    def cumulative_descent(self) -> float:
        return self.cumulative_ascent(ascending=False)

    def cum_asc_desc(self) -> Tuple[float, float]:
        """

        Parameters
        ----------

        Returns
        -------
            float, float
            cumulative ascent and descent
        """

        return self.cumulative_ascent(), self.cumulative_descent()

    def reset(self):
        """
        reset the elevations and distances array to original.

        Returns
        -------

        """
        self.elevations = self.elevations_orig.copy()
        self.distances = self.distances_orig.copy()

        return self

    def copy(self):
        copied = ElevationProfile(self.distances_orig, self.elevations_orig)
        copied.distances = self.distances.copy()
        copied.elevations = self.elevations.copy()
        return copied

    def flip(self):
        """
        Flip the elevation profile so that start = end and end = start.

        Returns
        -------

        """

        self.distances_orig = self.distances_orig[-1] - self.distances_orig[::-1]
        self.distances = self.distances[-1] - self.distances[::-1]
        self.elevations_orig = self.elevations_orig[::-1]
        self.elevations = self.elevations_orig[::-1]

        return self

    def plot(self, *plot_args, plot_kwargs=None):
        """
        Plot the elevation profile with pyplot.plot()

        Parameters
        ----------
        plot_args :
            additional arguments for the pyplot.plot function
        plot_kwargs : dict
            dictionary of keyword arguments for the pyplot.plot function

        Returns
        -------

        """

        if plot_kwargs is None:
            plot_kwargs = {}

        plt.figure()
        plt.plot(self.distances, self.elevations, *plot_args, **plot_kwargs)
        plt.show()


class DEM:
    """
    Digital Elevation Model. Contains the elevation raster.

    """

    def __init__(self, file: str, elevation_band: int = 1):
        """
        Parameters
        ----------
            file : str
                location to geotiff with elevation data
        """

        self.dem = rasterio.open(file)

        elev = self.dem.read(elevation_band)

        # pad the elevation array, so that not out of bounds
        self.elev = np.pad(elev, 2, mode='edge')

        self.crs = CRS.from_wkt(self.dem.crs.to_wkt())

        print("Loaded dem as EPSG:" + str(self.crs.to_epsg()))

    def sample_point(self, point: Point, interpolated: bool = True) -> float:
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

    def sample_coords(self, p_x: float, p_y: float, interpolated: bool = True) -> float:
        """
        Parameters
        ----------

            p_x : float
                x coordinate / longitude
            p_y : float
                y coordinate / latitude
            interpolated : bool
                Weather or not should be sampled from interpolated dem values.

        Returns
        -------
            elevation : float
                elevation at p_x, p_y
        """

        # get the index of the raster pixel containing the point
        row, col = self.dem.index(p_x, p_y)

        # return nan value if not in raster
        if not ((self.dem.bounds.bottom <= p_y <= self.dem.bounds.top) and
                (self.dem.bounds.left <= p_x <= self.dem.bounds.right)):
            return np.nan

        if not interpolated:
            return self.elev[row+2, col+2]  # +2 because of padding

        # get raster pixel center
        r_x, r_y = self.dem.xy(row, col)

        # because elev is padded
        row = row + 2
        col = col + 2

        row_from, row_to, col_from, col_to = None, None, None, None

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

        assert row_from is not None
        assert row_to is not None
        assert col_from is not None
        assert col_to is not None

        row_from += row
        row_to += row
        col_from += col
        col_to += col

        # the 16 supporting points of the interpolation
        z = self.elev[row_from:row_to + 1, col_from:col_to + 1]
        z = z.flatten()

        # get the coordinates for each supporting point
        x_coors = []
        y_coors = []
        for row in range(row_from, row_to + 1):
            for col in range(col_from, col_to + 1):
                x, y = self.dem.xy(row, col)
                x_coors.append(x)
                y_coors.append(y)

        # 5. interpolate cubic with scipy
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
        f = interpolate.interp2d(x_coors, y_coors, z, kind='cubic')

        e = f(p_x, p_y)[0]

        return e

    def elevation_profile(self, line: Union[LineString, GeoSeries], distance: float = 10, interpolated: bool = True) \
            -> ElevationProfile:
        """
        Given a Line Geometry and a distance, uniformly samples from the DEM along the Line. Optionally performs bicubic
        interpolation when sampling. Returns an ElevationProfile Object.

        Parameters
        ----------
            line : LineString or GeoSeries
                either shapely linestring, must be same crs as dem, or geopandas series with 1 linestring entry the crs
                is converted automatically
            distance : float
                default 10. The distance between the sample points on the line. Last distance may be shorter.
            interpolated : bool
                if True, then the elevation is bicubic interpolated.
        
        Returns
        -------
            ElevationProfile
        """

        if isinstance(line, gpd.GeoSeries):
            if line.crs.to_epsg() != self.crs.to_epsg():
                line = line.to_crs(self.crs)

            line = line.iloc[0]

        # 3. process the line to obtain evenly spaced sample points along the line
        # https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
        distances = np.arange(0, line.length, distance)

        # cannot use line.boundary[1] for last point because if line is a circle boundary is empty
        # so we use last coords to construct last point
        last_coord = line.coords[-1]
        sample_points = [line.interpolate(d) for d in distances] + [Point(last_coord[0], last_coord[1])]

        sample_point_elevation = []

        for sample_point in sample_points:
            p_x = sample_point.x
            p_y = sample_point.y

            sample_point_elevation.append(self.sample_coords(p_x, p_y, interpolated=interpolated))

        distances = np.append(distances, line.length)
        elevations = np.array(sample_point_elevation)

        return ElevationProfile(distances, elevations)


def _filter_overlapping(start_dists: List[float], end_dists: List[float]) -> Tuple[List[float], List[float]]:
    """
    From a list of intervals, defined by start and end dists, merges overlapping intervals and returns a set of
    intervals that do not overlap.

    Parameters
    ----------
    start_dists
    end_dists

    Returns
    -------

    """

    if len(start_dists) == 0:
        return [], []

    # stack contains previous elements
    start_dist_stack = []
    end_dist_stack = []

    start_dist_stack.append(start_dists[0])
    end_dist_stack.append(end_dists[0])

    for i in range(0, len(start_dists)):

        # check if not overlaps with stack top
        if start_dists[i] > end_dist_stack[-1]:

            # push to stack
            start_dist_stack.append(start_dists[i])
            end_dist_stack.append(end_dists[i])

        # if overlaps with stack top and end dist is greater
        elif start_dists[i] <= end_dist_stack[-1] < end_dists[i]:

            # update the end dist of stack top
            end_dist_stack[-1] = end_dists[i]

    return start_dist_stack, end_dist_stack
