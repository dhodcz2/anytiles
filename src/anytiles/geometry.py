from __future__ import annotations

from typing import Type

from geopandas import GeoDataFrame, GeoSeries
from anytiles.util import import_folium
from pandas import Series, DataFrame


if False:
    import folium
    from anytiles.anytiles import Tiles

def iter_colors():
    yield from 'red green yellow orange blue purple pink'.split()

class Geometry(GeoDataFrame):
    _metadata = GeoDataFrame._metadata + ['tiles']
    tiles: Tiles

    def explore(self, *args, tiles: str = 'cartodbdark_matter', **kwargs) -> folium.Map:
        folium = import_folium()
        from anytiles.anytiles import Tiles
        colors = iter_colors()
        # no fill
        m = GeoDataFrame.explore(
            self.reset_index(),
            *args,
            tiles=tiles,
            **kwargs,
            # color=next(colors),
            name='original',
            style_kwds={
                'color': next(colors),
                'fill': False,
            }
        )
        for key, value in self.tiles.__dict__.items():
            if not isinstance(value, Tiles):
                continue
            # m = value.explore(
            m = GeoDataFrame.explore(
                value.geometry.reset_index(),
                *args,
                tiles=tiles,
                **kwargs,
                color=next(colors),
                m=m,
                name=key,
                # fill_color=None,
                style_kwds={
                    'color': next(colors),
                    'fill': False,
                }
            )

        folium.LayerControl().add_to(m)
        return m
