#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database/URBAN'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/004_S10_GISA_EXPOSURE'
REGIONS = {'East Asia': (100., 148., 15., 45.),
           'North America': (-130., -65., 15., 45.)}
PANELS = {'A': 'East Asia', 'B': 'North America'}
LAND_COLOR = '#D2D2D2'
CITY_COLOR = '#C25D34'
OCEAN_COLOR = '#FFFFFF'
BIN_ORDER = ('-100-0 km', '-300--100 km')
RATIO_COLUMNS = ['region', 'distance_band', 'average_impervious_area_ratio']
DPI = 600
FONT_SIZE = 11

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'axes.linewidth': .7,
})


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_plot_data(data_dir):
    ratios = pd.read_csv(data_dir / 'FigS10_urban_ratios.csv', float_precision='round_trip')
    expected = [(region, band) for region in REGIONS for band in BIN_ORDER]
    if (list(ratios.columns) != RATIO_COLUMNS
            or list(ratios[RATIO_COLUMNS[:2]].itertuples(index=False, name=None)) != expected):
        raise ValueError('Expected the four original region/band ratio rows')
    values = ratios.average_impervious_area_ratio.to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Impervious area ratios must lie within [0, 1]')
    regions = {}
    for panel, region in PANELS.items():
        with xr.open_dataset(data_dir / f'FigS10{panel}_urban.nc', engine='h5netcdf') as source:
            if (set(source.data_vars) != {'display_class', 'signed_distance_km'}
                    or source.display_class.dims != ('lat', 'lon')
                    or source.signed_distance_km.dims != ('contour_lat', 'contour_lon')):
                raise ValueError(f'Panel {panel}: incompatible plotting fields/dimensions')
            if (source.attrs.get('region') != region or source.attrs.get('year') != 2021
                    or source.attrs.get('urban_threshold') != .2
                    or source.attrs.get('contour_stride') != 10):
                raise ValueError(f'Panel {panel}: incompatible region/year/threshold metadata')
            dataset = source.load()
        lat, lon = dataset.lat.values, dataset.lon.values
        west, east, south, north = REGIONS[region]
        for coordinates, lower, upper in ((lat, south, north), (lon, west, east)):
            if (coordinates.ndim != 1 or coordinates.size < 2 or not np.isfinite(coordinates).all()
                    or not np.allclose(np.diff(coordinates), .01, rtol=0, atol=1e-4)
                    or coordinates[0] < lower - 1e-6 or coordinates[-1] > upper + 1e-6):
                raise ValueError(f'Panel {panel}: native 0.01-degree coordinates changed')
        if (not np.array_equal(dataset.contour_lat.values, lat[::10])
                or not np.array_equal(dataset.contour_lon.values, lon[::10])):
            raise ValueError(f'Panel {panel}: contour grid differs from original stride')
        category = dataset.display_class.values
        finite = np.isfinite(category)
        if not finite.any() or np.isinf(category).any() or not np.isin(category[finite], [0., 1.]).all():
            raise ValueError(f'Panel {panel}: display classes must be 0, 1 or missing')
        distance = dataset.signed_distance_km.values
        if np.isinf(distance).any() or not np.isfinite(distance).any() or np.nanmax(distance) > 1e-6:
            raise ValueError(f'Panel {panel}: invalid signed inland distances')
        rows = ratios.loc[ratios.region.eq(region)].to_dict('records')
        regions[region] = (dataset, rows)
    return regions


def coordinate_edges(values):
    values = np.asarray(values, float)
    mid = (values[:-1]+values[1:])/2
    return np.r_[values[0]-(mid[0]-values[0]), mid,
                 values[-1]+(values[-1]-mid[-1])]


def draw_city_map(axis, region, dataset):
    """Keep native urban classes and the original every-tenth-cell contours."""
    axis.set_extent(REGIONS[region], crs=ccrs.PlateCarree())
    axis.set_aspect('equal', adjustable='box')
    axis.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR,
                     edgecolor='none', zorder=0)
    axis.add_feature(cfeature.LAND, facecolor=LAND_COLOR,
                     edgecolor='none', linewidth=0, zorder=0)
    axis.pcolormesh(
        coordinate_edges(dataset.lon.values), coordinate_edges(dataset.lat.values),
        np.ma.masked_invalid(dataset.display_class.values), shading='flat',
        cmap=ListedColormap([LAND_COLOR, CITY_COLOR]),
        vmin=-.5, vmax=1.5, transform=ccrs.PlateCarree(),
        rasterized=True, zorder=2,
    )
    axis.contour(
        dataset.contour_lon.values, dataset.contour_lat.values,
        np.ma.masked_invalid(dataset.signed_distance_km.values),
        levels=[-300, -100], colors='#595959', linewidths=.65,
        linestyles=['solid', 'dashed'],
        transform=ccrs.PlateCarree(), zorder=3,
    )
    west, east, _, _ = REGIONS[region]
    axis.set_xticks(np.arange(np.ceil(west / 10) * 10, east + .01, 10),
                    crs=ccrs.PlateCarree())
    axis.set_yticks([20, 30, 40], crs=ccrs.PlateCarree())
    axis.xaxis.set_major_formatter(LongitudeFormatter())
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(direction='out', length=3, width=.7,
                     pad=2, labelsize=FONT_SIZE)


def ratio_text(rows):
    values = {row['distance_band']: row['average_impervious_area_ratio'] for row in rows}
    near, far = (values[key] for key in BIN_ORDER)
    return ('Average impervious area ratio\n'
            f'−300–−100 km: {far:.2f}\n'
            f'−100–0 km: {near:.2f}')


def add_ratio_annotation(axis, rows, region):
    east_asia = region == 'East Asia'
    axis.text(.985 if east_asia else .015, .025,
              ratio_text(rows), transform=axis.transAxes,
              ha='right' if east_asia else 'left', va='bottom',
              fontsize=FONT_SIZE - 2, linespacing=1.12, zorder=8)


def legend_handles():
    return [Patch(facecolor=CITY_COLOR, label='Impervious area ratio > 0.2'),
            Line2D([0], [0], color='#595959', lw=.7, ls='-', label='−300 km'),
            Line2D([0], [0], color='#595959', lw=.7, ls='--', label='−100 km')]


def create_ab_composite(regions):
    """Use the original composite geometry, with equal physical map heights."""
    fig = plt.figure(figsize=(10.5, 3.45), facecolor='white')
    map_bottom, map_height = .13, .65
    def map_width(region):
        west, east, south, north = REGIONS[region]
        return (map_height * fig.get_figheight() / fig.get_figwidth()
                * (east-west) / (north-south))
    left, gap = .07, .075
    widths = [map_width(region) for region in REGIONS]
    positions = [left, left+widths[0]+gap]
    axes, letters = [], []
    for x, width, region, panel in zip(positions, widths, REGIONS, PANELS):
        axis = fig.add_axes([x, map_bottom, width, map_height], projection=ccrs.PlateCarree())
        dataset, rows = regions[region]
        draw_city_map(axis, region, dataset)
        add_ratio_annotation(axis, rows, region)
        axes.append(axis)
        letters.append(fig.text(x-.025, map_bottom+map_height+.012, panel.lower(),
                                fontsize=15, fontweight='bold'))
    common_legend = fig.legend(handles=legend_handles(), loc='upper center', ncol=3,
                               frameon=False, bbox_to_anchor=(.52, .98), fontsize=FONT_SIZE)
    return fig, axes, letters, common_legend


def save_figures(regions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, letters, common_legend = create_ab_composite(regions)
    try:
        fig.canvas.draw()
        a, b = [axis.get_position() for axis in axes]
        if not np.allclose([a.y0, a.y1], [b.y0, b.y1], atol=1e-7, rtol=0):
            raise RuntimeError('S10 composite map heights are not aligned')
        fig.savefig(output_dir/'FigS10_composite.png', dpi=DPI,
                    facecolor='white', bbox_inches='tight', pad_inches=.035)
        common_legend.set_visible(False)
        for letter in letters:
            letter.set_visible(False)
        for panel, selected in zip(PANELS, axes):
            for axis in axes:
                axis.set_visible(axis is selected)
            # Preserve the composite axes; only rearrange the three legend entries.
            standalone_legend = selected.legend(handles=legend_handles(), loc='lower center',
                bbox_to_anchor=(.5, 1.035), ncol=1, frameon=False, fontsize=FONT_SIZE,
                borderaxespad=0., labelspacing=.25)
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                boxes = [selected.get_tightbbox(renderer), standalone_legend.get_window_extent(renderer)]
                extent = Bbox.union([box for box in boxes if box is not None])
                extent = extent.transformed(fig.dpi_scale_trans.inverted()).padded(.035)
                fig.savefig(output_dir/f'FigS10{panel}.png', dpi=DPI,
                            facecolor='white', bbox_inches=extent, pad_inches=0)
            finally:
                standalone_legend.remove()
        print(f'Saved composite and two panels: {output_dir}')
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=project_path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=project_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--cartopy-data-dir', type=project_path,
                        help='Optional existing/shared Natural Earth cache directory')
    args = parser.parse_args()
    if args.cartopy_data_dir is not None:
        cartopy.config['data_dir'] = args.cartopy_data_dir
    save_figures(load_plot_data(args.data_dir), args.output_dir)


if __name__ == '__main__':
    main()
