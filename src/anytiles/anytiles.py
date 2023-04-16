from __future__ import annotations


__all__ = ['AnyTiles']

import itertools

import geopandas as gpd

from toolz import pipe, curried

import abc
from typing import Type

import shapely.geometry.base
from pandas import Series

from functools import cached_property, lru_cache
from pathlib import Path

import geopy
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from abc import ABC, ABCMeta, abstractmethod

from ctypes import Union
from os import PathLike as _PathLike, PathLike

import numpy as np

from functools import cached_property
from abc import ABC, abstractmethod
import pandas as pd
import pyproj

from numpy import ndarray
import numba as nb
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
from anytiles.decorators import metadata
from anytiles.util import num2deg, deg2num
from anytiles.geometry import Geometry
from sklearn.utils.extmath import cartesian


class Tiles(DataFrame):
    values: ndarray[float]

    @cached_property
    def x(self) -> ndarray:
        return self.values[:, ::2]

    @cached_property
    def y(self) -> ndarray:
        return self.values[:, 1::2]

    @classmethod
    def from_points(
        cls,
        points: DataFrame | ndarray,
        files: list[PathLike] | Series[PathLike] = None,
        output_folder: PathLike = None,
        **kwargs
    ):
        if isinstance(points, DataFrame):
            index = points.index
            values = points.values
        else:
            index = np.arange(len(points))
            values = points

        if len(values.shape) == 2:
            match values.shape[1]:
                case 4:
                    values = np.concatenate((
                        values,
                        values[:, [0, 1, 3, 2]]
                    ))
                case 8:
                    values = values.reshape((-1, 4, 2))
        if len(values.shape) != 3:
            raise ValueError(f'Invalid shape: {values.shape}')
        if values.shape[1:] != (4, 2):
            raise ValueError(f'Invalid shape: {values.shape}')

        centroid = values.mean(axis=1)
        delta = values - centroid[:, None]
        theta = np.arctan2(delta[:, :, 1], delta[:, :, 0])
        # theta must be [0, 2pi)
        theta[theta < 0] += 2 * np.pi
        indices = np.argsort(theta, axis=1)
        r = np.arange(len(values))[:, None]
        # data = values[r, indices].reshape((-1, 8))
        data = (
            values[r, indices]
            .reshape((-1, 8))
        )

        res = cls(
            data=data,
            index=index,
        )
        res.files = files
        res.output_folder = output_folder
        return res

    _metadata = GeoDataFrame._metadata + 'files output_folder name zoom'.split()

    @metadata
    def geometry(self) -> Geometry:
        indices = np.arange(len(self)).repeat(5)
        data = self.values
        data = np.concatenate((data, data[:, :2]), axis=1)
        coords = data.reshape(-1, 2)
        # coords = np.concatenate((coords, coords[:, 0]))
        rings = shapely.linearrings(coords, indices=indices)
        geometry = shapely.polygons(rings)
        res = Geometry(geometry=geometry, index=self.index, crs=4326)
        res.tiles = self
        return res

    def __init_subclass__(cls, *, metadata=None, **kwargs):
        if metadata is not None:
            cls._metadata = cls._metadata + list(metadata)


class AnyTiles(Tiles):
    values: ndarray[float]

    @classmethod
    def from_geometry(
        cls,
        tiles: GeoSeries | GeoDataFrame,
        files: list[PathLike] | Series[PathLike],
        output_folder: PathLike = None,
        zoom=15,
        **kwargs
    ):
        index = tiles.index
        points: DataFrame = pd.DataFrame.from_records((
            coord
            for polygon in tiles.geometry
            for coord in polygon.exterior.coords
        ), columns='x y'.split(), nrows=len(tiles) * 5)
        trans = pyproj.proj.Transformer.from_crs(tiles.crs, 'epsg:4326', always_xy=True).transform
        points = DataFrame(
            np.vstack(
                trans(points.values[:, 0], points.values[:, 1])
            ).T
        ).drop(range(4, len(points), 5), axis=0)
        points = DataFrame((
            points.values
            .reshape((-1, 8))
        ), index=index)

        res = cls.from_points(
            points=points,
            files=files,
            output_folder=output_folder,
            **kwargs
        )
        res.zoom = zoom
        return res

    # noinspection PyArgumentList
    @metadata
    def slippy(self) -> Tiles:
        x = self.x.ravel()
        y = self.y.ravel()
        assert (
                (x >= -180) &
                (x <= 180) &
                (y >= -90) &
                (y <= 90)
        ).all()

        GW = x.min()
        GE = x.max()
        GS = y.min()
        GN = y.max()
        TW, TN = deg2num(GW, GN, self.zoom)
        TE, TS = deg2num(GE, GS, self.zoom)

        tw = np.arange(TW, TE + 2)
        tn = np.arange(TN, TS + 2)
        gw, gn = num2deg(tw, tn, self.zoom)
        ge = gw[1:]
        gs = gn[1:]
        gw = gw[:-1]
        gn = gn[:-1]
        data = np.hstack((
            cartesian((gw, gn)),
            cartesian((ge, gs)),
        ))[:, [0, 1, 2, 3, 0, 3, 2, 1]]
        index = pd.MultiIndex.from_arrays(
            cartesian((tw[:-1], tn[:-1])).T,
            names='x y'.split()
        )
        points = DataFrame(data, index=index, )
        res = Tiles.from_points(
            points=points,
        )
        return res
    # todo: where do we put the sjoin?


    @classmethod
    def from_dev(cls) -> AnyTiles:
        tiles = gpd.read_file(
            '/home/arstneio/Downloads/epk2_eng/epk2.shp',
            rows=5,
        )
        anytiles = AnyTiles.from_geometry(
            tiles=tiles,
            files=None,
            output_folder=None,
        )
        return anytiles
