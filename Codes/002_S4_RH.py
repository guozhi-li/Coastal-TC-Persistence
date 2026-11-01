#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.io import shapereader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

VARIABLE = 'RH'
FIGURE = 'S4'
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database' / VARIABLE
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results' / '002_S4_RH'
DEFAULT_COASTLINE = PROJECT_DIR / 'Database/GSHHG/GSHHS_f_L1_1400.shp'
SPATIAL_MASK_ALPHA = .93
FILES = {'ERA5': {'color': '#244A73', 'marker': 'o'},
         'MERRA2': {'color': '#D69E79', 'marker': 'o'}}
DPI = 600
FONT_SIZE = FONT = 8.5
NH_MONTHS = (7, 8, 9, 10, 11)

SH_MONTHS = (12, 1, 2, 3, 4)

DISTANCE_EDGES = np.arange(-400.0, 801.0, 100.0)

REGIONS = (
    {"name": "5-20N", "label": "NH (5–20°N)", "south": 5.0,
      "north": 20.0, "months": NH_MONTHS},
    {"name": "20-40N", "label": "NH (20–40°N)", "south": 20.0,
      "north": 40.0, "months": NH_MONTHS},
    {"name": "5-40S", "label": "SH (5–40°S)", "south": -40.0,
      "north": -5.0, "months": SH_MONTHS},
)

EXCLUSION_POLYGON = np.asarray(
    [
        (-26, 40), (60, 40), (40, 10), (23, -32), (23, -40),
        (-105, -40), (-105, 7), (-25, 7), (-26, 40),
    ],
    dtype=float,
)

MAP_COLORS = [
    "#3B79AE", "#5F9FC0", "#8ABBD0", "#B7D3E3", "#E5EFF7",
    "#FFFFFF",
    "#FBEDE7", "#F6C9C0", "#EE9E97", "#DA848D", "#C07278",
]

CMAP = LinearSegmentedColormap.from_list(
    "M2_consistent_blue_red", MAP_COLORS, N=256
)

VARIABLE_SETTINGS = {
    "RH": {
        "ylabel": r"Trend in RH (% decade$^{-1}$)",
        "map_title": r"Trend in RH (% decade$^{-1}$)",
        "vmax": 1.6,
        "ticks": (-1.6, -0.8, 0.0, 0.8, 1.6),
        "unit": "% decade^-1",
    },
    "VWS": {
        "ylabel": r"Trend in VWS (m s$^{-1}$ decade$^{-1}$)",
        "map_title": r"Trend in VWS (m s$^{-1}$ decade$^{-1}$)",
        "vmax": 0.8,
        "ticks": (-0.8, -0.4, 0.0, 0.4, 0.8),
        "unit": "m s^-1 decade^-1",
    },
}

CONFIG = VARIABLE_SETTINGS[VARIABLE]



plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.linewidth": 0.7,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)

def require(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')

def exclusion_mask(distance):
    lon, lat = np.meshgrid(distance.lon.values, distance.lat.values)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    inside = MplPath(EXCLUSION_POLYGON, closed=True).contains_points(points, radius=1e-10).reshape(lon.shape)
    return xr.DataArray(inside | (np.abs(lat) <= 5), coords={'lat': distance.lat, 'lon': distance.lon}, dims=('lat', 'lon'))

def cyclic(array):
    array = array.assign_coords(lon=np.mod(array.lon.values, 360.0)).sortby("lon")
    values, lon = add_cyclic_point(np.asarray(array.values, float), coord=np.asarray(array.lon.values, float))
    return (values, lon, np.asarray(array.lat.values, float))

def clean_tick(value, _):
    return '0' if np.isclose(value, 0) else f'{value:g}'.replace('-', '−')

def style_map(axis):
    tr = ccrs.PlateCarree()
    axis.set_xlim(-180, 180)
    axis.set_ylim(-40, 40)
    axis.set_xticks([-180, -120, -60, 0, 60, 120])
    axis.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W'])
    axis.set_yticks(np.arange(-40, 41, 20), crs=tr)
    axis.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    axis.tick_params(direction='out', length=3, width=0.7, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color('#222')

def draw_map(fig, axis, cax, spatial, distance, coastline):
    data, lon, lat = cyclic(spatial.trend)
    vmax = CONFIG["vmax"]
    levels = np.linspace(-vmax, vmax, 25)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    filled = axis.contourf(
        lon,
        lat,
        data,
        levels=levels,
        cmap=CMAP,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # Stipple all significant native-grid cells at approximately 2-degree spacing.
    # Periodic matching avoids duplicate or misplaced points at the date line.
    sig_native = np.asarray((spatial.MK_P < 0.05).transpose('lat', 'lon').values, bool)
    lon_native = np.asarray(spatial.MK_P.lon.values, float)
    lat_native = np.asarray(spatial.MK_P.lat.values, float)
    lon_sample = np.arange(0.0, 360.0, 2.0)
    lat_sample = np.arange(-40.0, 40.0 + 0.001, 2.0)
    lon_index = np.argmin(
        np.abs((lon_native[None, :] - lon_sample[:, None] + 180.0) % 360.0 - 180.0),
        axis=1,
    )
    _, first = np.unique(lon_index, return_index=True)
    lon_index = lon_index[np.sort(first)]
    lat_index = np.unique([int(np.argmin(np.abs(lat_native-value))) for value in lat_sample])
    sampled_sig = sig_native[np.ix_(lat_index, lon_index)]
    sampled_lon, sampled_lat = np.meshgrid(lon_native[lon_index], lat_native[lat_index])
    axis.scatter(
        sampled_lon[sampled_sig],
        sampled_lat[sampled_sig],
        s=0.80,
        marker="o",
        c="black",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=5,
        rasterized=True,
    )
    # Align the display mask to the exact ERA5 trend grid, using the same
    # distance limits and geographic exclusions as the B-D profile calculation.
    dist = distance.sel(lat=spatial.trend.lat, lon=spatial.trend.lon).transpose('lat', 'lon')
    analysis_domain = (
        np.isfinite(dist) & (dist >= DISTANCE_EDGES[0])
        & (dist <= DISTANCE_EDGES[-1]) & ~exclusion_mask(dist)
    )
    # Outline land from the prefiltered GSHHG polygons; do not fill land.
    require(coastline)
    reader = shapereader.Reader(str(coastline))
    try:
        axis.add_geometries(
            list(reader.geometries()), crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='#555555', linewidth=0.4, zorder=4,
        )
    finally:
        reader.close()
    # Layer order: land outline / offshore contour, stippling, then overlay.
    # Apply the overlay to both inland and offshore areas outside -400..800 km.
    mask_values, mask_lon, mask_lat = cyclic((~analysis_domain).astype(float))
    mask_values = np.ma.masked_where(mask_values < 0.5, mask_values)
    axis.pcolormesh(
        mask_lon, mask_lat, mask_values, shading='nearest',
        cmap=mpl.colors.ListedColormap(['white']), vmin=0.0, vmax=1.0,
        alpha=SPATIAL_MASK_ALPHA, transform=ccrs.PlateCarree(),
        zorder=7, edgecolors='none', antialiased=False, rasterized=True,
    )
    # Outline the selected analysis region independently of the display overlay.
    boundary_values, boundary_lon, boundary_lat = cyclic(analysis_domain.astype(float))
    axis.contour(
        boundary_lon, boundary_lat, boundary_values,
        levels=[0.5], colors='#444444', linewidths=0.35,
        linestyles=[(0, (3, 2))], transform=ccrs.PlateCarree(), zorder=6,
    )
    style_map(axis)
    box = cax.get_position()
    title = fig.text(box.x0 - 0.014, box.y0 + box.height / 2, CONFIG['map_title'], ha='right', va='center', fontsize=FONT)
    bar = fig.colorbar(filled, cax=cax, orientation='horizontal', ticks=CONFIG['ticks'], extend='both')
    bar.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_tick))
    bar.ax.tick_params(length=2.5, width=0.7, pad=2)
    bar.outline.set_linewidth(0.7)
    return title

def shared_profile_limits(table):
    low = (table.OLS_Trend_per_Decade - table.OLS_CI95_per_Decade).min()
    high = (table.OLS_Trend_per_Decade + table.OLS_CI95_per_Decade).max()
    span = max(high - low, abs(high) * 0.2, abs(low) * 0.2, 0.1 if VARIABLE == 'RH' else 0.03)
    return (float(low - 0.08 * span), float(high + 0.08 * span))

def draw_profile(axis, table, region, ylabel=False, legend=False):
    """Draw one regional profile using the exact M2 B-D visual grammar."""
    for source in ("ERA5", "MERRA2"):
        sample = table[
            (table.Region == region["name"]) & (table.Source == source)
        ].sort_values("Distance_Center_km")
        x = sample.Distance_Center_km.to_numpy(float)
        y = sample.OLS_Trend_per_Decade.to_numpy(float)
        ci = sample.OLS_CI95_per_Decade.to_numpy(float)
        significant = (sample.MK_P.to_numpy(float) < 0.05)
        color = FILES[source]["color"]
        marker = FILES[source]["marker"]

        axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.13, linewidth=0)
        axis.plot(x, y, color=color, linewidth=1.25, zorder=3)
        axis.scatter(
            x[significant], y[significant], s=11, marker=marker,
            facecolor=color, edgecolor=color, linewidth=0.4, zorder=4,
        )
        axis.scatter(
            x[~significant], y[~significant], s=17, marker=marker,
            facecolor="white", edgecolor=color, linewidth=0.8, zorder=4,
        )

    # M2-style zero references, drawn by every B-D profile axis.
    axis.axhline(0, color='#777777', linewidth=0.55, zorder=1)
    axis.axvline(0, color='#777777', linewidth=0.55, zorder=1)
    axis.set_xlim(-400, 800)
    axis.set_xticks((-400, -200, 0, 200, 400, 600, 800))
    axis.set_ylim(*shared_profile_limits(table))
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_tick))
    axis.set_xlabel("Distance to coast (km)", fontsize=FONT_SIZE)
    axis.text(
        0.03, 0.96, region["label"], transform=axis.transAxes,
        ha="left", va="top", fontsize=FONT_SIZE,
    )
    if ylabel:
        axis.set_ylabel(CONFIG["ylabel"], fontsize=FONT_SIZE)
    else:
        axis.tick_params(axis="y", labelleft=False)
    axis.tick_params(
        direction="out", length=3, width=0.7, pad=2, labelsize=FONT_SIZE
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")

    if legend:
        handles = [
            Line2D(
                [0], [0], color=FILES[source]["color"],
                marker=FILES[source]["marker"], markersize=2.7,
                linewidth=1.25, label=source,
            )
            for source in ("ERA5", "MERRA2")
        ]
        axis.legend(
            handles=handles, loc="upper right", frameon=False,
            fontsize=FONT_SIZE, handlelength=1.50, handletextpad=0.35,
            labelspacing=0.22, borderaxespad=0.22,
        )


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def read_plot_data(data_dir):
    nc_path = data_dir / f'Fig{FIGURE}A_{VARIABLE}_spatial.nc'
    csv_path = data_dir / f'Fig{FIGURE}BCD_{VARIABLE}_profiles.csv'
    require(nc_path)
    require(csv_path)
    with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
        if set(ds.data_vars) != {'trend', 'MK_P', 'distance_to_coast'}:
            raise ValueError('Spatial file must contain only trend, MK_P and distance_to_coast')
        if any(ds[key].dims != ('lat', 'lon') for key in ds.data_vars):
            raise ValueError('All spatial fields must use (lat, lon) dimensions')
        spatial = ds.load()
    p = spatial.MK_P.values
    valid = p[np.isfinite(p)]
    if not len(valid) or np.any((valid < 0) | (valid > 1)):
        raise ValueError('MK_P values must lie between 0 and 1')
    for coordinate in ('lat', 'lon'):
        values = spatial[coordinate].values
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError(f'{coordinate} must be finite and strictly increasing')
    table = pd.read_csv(csv_path, float_precision='round_trip')
    columns = ['Region', 'Source', 'Distance_Center_km', 'OLS_Trend_per_Decade', 'OLS_CI95_per_Decade', 'MK_P']
    expected = [(region['name'], source, center) for source in ('ERA5', 'MERRA2')
                for region in REGIONS for center in (DISTANCE_EDGES[:-1]+DISTANCE_EDGES[1:])/2]
    if list(table.columns) != columns or list(table[columns[:3]].itertuples(index=False, name=None)) != expected:
        raise ValueError('Expected 72 ordered profile rows and six plotting columns')
    if not np.isfinite(table[columns[2:]].to_numpy(float)).all():
        raise ValueError('Profile statistics must be finite')
    if (table.OLS_CI95_per_Decade < 0).any() or not table.MK_P.between(0, 1).all():
        raise ValueError('Invalid profile confidence interval or MK_P')
    return spatial, table


def validate_coastline(path):
    missing = [str(path.with_suffix(suffix)) for suffix in ('.shp', '.shx', '.dbf')
               if not path.with_suffix(suffix).is_file()]
    if missing:
        raise FileNotFoundError('Incomplete GSHHG shapefile. Copy the matching .shp, .shx and .dbf files together. Missing: ' + ', '.join(missing))


def panel_bbox(fig, axes, artists):
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in axes]
    boxes.extend(artist.get_window_extent(renderer) for artist in artists)
    return Bbox.union([box for box in boxes if box is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(.025)


def save_panel(fig, output_file, axes, artists):
    axis_states = [(axis, axis.get_visible()) for axis in fig.axes]
    text_states = [(text, text.get_visible()) for text in fig.texts]
    try:
        for axis, visible in axis_states:
            axis.set_visible(visible and axis in axes)
        for text, visible in text_states:
            text.set_visible(visible and text in artists)
        fig.canvas.draw()
        fig.savefig(output_file, dpi=DPI, facecolor='white',
                    bbox_inches=panel_bbox(fig, axes, artists), pad_inches=0)
    finally:
        for axis, visible in axis_states:
            axis.set_visible(visible)
        for text, visible in text_states:
            text.set_visible(visible)


def draw_figures(output_dir, coastline, spatial, profiles):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.4, 4.8), facecolor='white')
    try:
        left, right, top = .08, .94, .91
        map_height = (right-left)*8.4/(4.5*4.8)
        map_bottom = top-map_height
        axis_a = fig.add_axes([left, map_bottom, right-left, map_height],
                             projection=ccrs.PlateCarree(central_longitude=180))
        cax = fig.add_axes([.49, .948, .45, .017])
        title = draw_map(fig, axis_a, cax, spatial, spatial.distance_to_coast, coastline)
        fig.text(.055, .935, 'A', fontsize=10.5, fontweight='bold', va='bottom')
        panels = [('a_' + VARIABLE + '_spatial', [axis_a, cax], [title])]
        profile_top, profile_bottom = map_bottom-.100, .120
        for (letter, region), (x0, width) in zip(zip('BCD', REGIONS), ((.08, .24), (.39, .24), (.70, .24))):
            axis = fig.add_axes([x0, profile_bottom, width, profile_top-profile_bottom])
            draw_profile(axis, profiles, region, ylabel=letter == 'B', legend=letter == 'D')
            fig.text(x0-.023, profile_top+.010, letter, fontsize=10.5, fontweight='bold', va='bottom')
            panels.append((f'{letter.lower()}_ERA5_MERRA2_{region["name"]}', [axis], []))
        # Draw once, then crop those same axes and labels at a common physical scale.
        fig.canvas.draw()
        fig.savefig(output_dir/f'Fig{FIGURE}_composite.png', dpi=DPI, facecolor='white', bbox_inches='tight', pad_inches=.025)
        for suffix, axes, artists in panels:
            save_panel(fig, output_dir/f'Fig{FIGURE}{suffix}.png', axes, artists)
        print(f'Saved composite and four panels: {output_dir}')
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=resolve_project_path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=resolve_project_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--coastline', type=resolve_project_path, default=DEFAULT_COASTLINE)
    args = parser.parse_args()
    validate_coastline(args.coastline)
    spatial, profiles = read_plot_data(args.data_dir)
    draw_figures(args.output_dir, args.coastline, spatial, profiles)


if __name__ == '__main__':
    main()
