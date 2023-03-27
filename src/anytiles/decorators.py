from __future__ import annotations
from functools import cached_property
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame


if False:
    from .anytiles import AnyTiles

class metadata(property):
    def __set_name__(self, owner: AnyTiles, name):
        self.name = name
        if '_metadata' not in owner.__dict__:
            metadata = owner._metadata = []
        else:
            metadata = owner._metadata
        metadata.append(name)

    def __set__(self, instance: AnyTiles, value):
        if isinstance(value, (Series, DataFrame)):
            value = value.loc[instance.index]
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        instance.__dict__[self.name] = value = super().__get__(instance, owner)
        return value


    def __delete__(self, instance):
        instance.__dict__.pop(self.name, None)
