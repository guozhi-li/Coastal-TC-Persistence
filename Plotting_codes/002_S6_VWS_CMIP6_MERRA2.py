#!/usr/bin/env python3
import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from cartopy.io import shapereader

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database/VWS_ATTRIBUTION'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/002_S6_VWS_CMIP6_MERRA2'
GSHHG_SHP = PROJECT_DIR / 'Database/GSHHG/GSHHS_f_L1_1400.shp'
DPI = 600


EXPERIMENTS = {
    "hist-aer": {"label": "AER", "color": "#DDAA9E", "marker": "o"},
    "hist-GHG": {"label": "GHG", "color": "#456D9F", "marker": "s"},
    "hist-nat": {"label": "NAT", "color": "#88ADC1", "marker": "^"},
    "historical": {"label": "HIST", "color": "#9B5F5B", "marker": None},
}

NH_MONTHS = (7, 8, 9, 10, 11)

S6_REGION = {
    "name": "WNP_20-40N",
    "label": "WNP (20–40°N)",
    "lat_min": 20.0,
    "lat_max": 40.0,
    "months": NH_MONTHS,
    "season": "July–November",
}

EAST_ASIA_EXTENT = (100.0, 145.0, 10.0, 45.0)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.unicode_minus": True,
})


PROFILE_COLUMNS = ('Series', 'Distance_Center_km', 'VWS_Trend_m_s_per_decade', 'MK_P')
SERIES_ORDER = ('AER', 'GHG', 'NAT', 'HIST', 'ERA5')
DISTANCE_CENTERS = tuple(range(-350, 400, 100))
MAP_VARIABLES = ('aod_trend', 'MK_P', 'distance_to_coast')


def validate_profiles(table):
    """Validate 32 ensemble medians and eight ERA5 trends with genuine MK P."""
    if table.columns.duplicated().any() or set(table.columns) != set(PROFILE_COLUMNS):
        raise ValueError(f'S6 profiles require exactly these columns: {PROFILE_COLUMNS}')
    table = table.loc[:, list(PROFILE_COLUMNS)].copy()
    keys = ['Series', 'Distance_Center_km']
    for name in PROFILE_COLUMNS[1:]:
        table[name] = pd.to_numeric(table[name], errors='raise')
    expected = [(series, distance) for series in SERIES_ORDER for distance in DISTANCE_CENTERS]
    if len(table) != 40 or table.duplicated(keys).any():
        raise ValueError('S6 requires 40 unique profile points: five series and eight distance bins')
    if set(table[keys].itertuples(index=False, name=None)) != set(expected):
        raise ValueError('S6 profile series/distance combinations are incomplete or unexpected')
    if not np.isfinite(table['VWS_Trend_m_s_per_decade']).all():
        raise ValueError('S6 profile trends must be finite')
    observed = table.Series.eq('ERA5')
    p = table.loc[observed, 'MK_P']
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError('ERA5 profiles require genuine finite MK P values within [0, 1]')
    if not table.loc[~observed, 'MK_P'].isna().all():
        raise ValueError('Model medians have no MK test and must leave MK_P blank')
    order = pd.MultiIndex.from_tuples(expected, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(PROFILE_COLUMNS)]


def validate_aod_map(dataset):
    """Protect native East Asia coordinates and retain real P values, not flags."""
    if set(dataset.data_vars) != set(MAP_VARIABLES):
        raise ValueError(f'S6 AOD map requires exactly these variables: {MAP_VARIABLES}')
    for name in MAP_VARIABLES:
        if dataset[name].dims != ('lat', 'lon'):
            raise ValueError(f'S6 {name} must have dimensions (lat, lon)')
    lat, lon = dataset.lat.values, dataset.lon.values
    expected_lat = np.arange(10.0, 45.0 + .001, .5)
    expected_lon = np.arange(100.0, 145.0 + .001, .625)
    if (lat.shape != expected_lat.shape or lon.shape != expected_lon.shape
            or not np.allclose(lat, expected_lat, rtol=0, atol=1e-8)
            or not np.allclose(lon, expected_lon, rtol=0, atol=1e-8)):
        raise ValueError('S6 must retain the 71 x 73 native MERRA-2 East Asia grid')
    for key, expected in {'period': '1980-2024', 'season': 'July-November',
                          'trend_method': 'OLS', 'significance_method': 'original Mann-Kendall'}.items():
        if dataset.attrs.get(key) != expected:
            raise ValueError(f'S6 AOD map has incompatible {key} metadata')
    trend = np.asarray(dataset.aod_trend.values, dtype=float)
    p = np.asarray(dataset.MK_P.values, dtype=float)
    distance = np.asarray(dataset.distance_to_coast.values, dtype=float)
    finite = np.isfinite(trend)
    if np.isinf(trend).any() or not finite.any() or not np.isfinite(distance).all():
        raise ValueError('S6 AOD trends/distances contain invalid values')
    if np.isinf(p).any() or not np.isfinite(p[finite]).all() or not ((p[finite] >= 0) & (p[finite] <= 1)).all():
        raise ValueError('Every finite AOD trend requires a genuine MK P value within [0, 1]')
    if not np.isnan(p[~finite]).all():
        raise ValueError('AOD cells without a valid trend must leave MK_P missing')
    return dataset




def check_coastline(path):
    path = Path(path)
    actual = path if path.is_file() else path.with_suffix(path.suffix.upper())
    missing = []
    for suffix in ('.shp', '.shx', '.dbf'):
        component = actual.with_suffix(suffix)
        if not component.is_file() and not component.with_suffix(suffix.upper()).is_file():
            missing.append(str(component))
    if missing:
        raise FileNotFoundError('Missing GSHHG shapefile components: ' + ', '.join(missing))
    return actual


@lru_cache(maxsize=1)
def load_land_geometries():
    path = check_coastline(GSHHG_SHP)
    reader = shapereader.Reader(str(path))
    try:
        return tuple(geometry for geometry in reader.geometries()
                     if geometry is not None and not geometry.is_empty)
    finally:
        reader.close()


def load_plot_data(data_dir):
    data_dir = Path(data_dir)
    path = data_dir / 'FigS6A_VWS_profiles.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing S6 plotting data: {path}; run preprocessing first')
    profiles = validate_profiles(pd.read_csv(path, float_precision='round_trip'))
    ensemble = profiles.loc[profiles.Series.ne('ERA5')].rename(columns={
        'VWS_Trend_m_s_per_decade': 'Median_VWS_Trend'}).copy()
    experiments = {settings['label']: name for name, settings in EXPERIMENTS.items()}
    ensemble['Experiment'] = ensemble.Series.map(experiments)
    ensemble['Region'] = S6_REGION['name']
    era5 = profiles.loc[profiles.Series.eq('ERA5')].copy()
    era5['Region'] = S6_REGION['name']
    path = data_dir / 'FigS6B_MERRA2_AOD.nc'
    if not path.is_file():
        raise FileNotFoundError(f'Missing S6 AOD plotting data: {path}; run preprocessing first')
    with xr.open_dataset(path, engine='h5netcdf') as source:
        aod = validate_aod_map(source.load())
    return ensemble, aod, era5


def _unicode_minus(value, decimals=None):
    if np.isclose(value, 0.0):
        return "0"
    text = f"{value:.{decimals}f}" if decimals is not None else f"{value:g}"
    return text.replace("-", "−")

def _numeric_formatter(decimals=None):
    return mpl.ticker.FuncFormatter(lambda value, position: _unicode_minus(value, decimals))

def style_plain_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")

def profile_legend_handles(axis):
    """Use a filled black ERA5 symbol only in the legend."""
    handles, labels = axis.get_legend_handles_labels()
    handles = [
        Line2D([], [], color='black', linewidth=1.35, marker='o',
               markersize=3.4, markerfacecolor='black', markeredgecolor='black',
               markeredgewidth=0.7, label='ERA5') if label == 'ERA5' else handle
        for handle, label in zip(handles, labels)
    ]
    return handles, labels

def draw_cmip6_axis(axis, ensemble, region, show_ylabel=False, show_legend=False, era5_table=None):
    table = ensemble[ensemble["Region"].eq(region["name"])]
    for experiment, settings in EXPERIMENTS.items():
        selected = table[table["Experiment"].eq(experiment)].sort_values(
            "Distance_Center_km"
        )
        x = selected["Distance_Center_km"].to_numpy(float)
        y = selected["Median_VWS_Trend"].to_numpy(float)
        axis.plot(
            x, y, color=settings["color"], marker=settings["marker"],
            markersize=3.4, markeredgewidth=0.6, linewidth=1.15,
            label=settings["label"], zorder=3,
        )
    if era5_table is not None:
        reference = era5_table[era5_table['Region'].eq(region['name'])].sort_values('Distance_Center_km')
        x = reference['Distance_Center_km'].to_numpy(float)
        y = reference['VWS_Trend_m_s_per_decade'].to_numpy(float)
        significant = reference['MK_P'].to_numpy(float) < 0.05
        axis.plot(
            x, y, color='black', linewidth=1.35, marker='o', markersize=3.4,
            markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.7,
            label='ERA5', zorder=4,
        )
        # Black filled circles: p<0.05; open black circles: p>=0.05.
        axis.scatter(
            x[significant], y[significant], s=3.4**2,
            marker='o', facecolor='black', edgecolor='black', linewidth=0.7,
            zorder=5,
        )
    axis.axvline(0.0, color="black", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    axis.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    axis.set_xlim(-400, 400)
    axis.set_xticks([-400, -200, 0, 200, 400])
    axis.set_xlabel("Distance to coast (km)")
    if show_ylabel:
        axis.set_ylabel(r"Trend in VWS (m s$^{-1}$ decade$^{-1}$)")
    if show_legend:
        handles, labels = profile_legend_handles(axis)
        axis.legend(handles, labels, frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.02),
                    ncol=2, handlelength=1.7, borderaxespad=0.4)
    style_plain_axis(axis)

def map_color_limit(trend):
    values = np.abs(np.asarray(trend.values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("MERRA-2 AOD trend map contains no finite values")
    upper = float(np.nanpercentile(values, 98.0))
    return max(upper, 0.001)

def draw_aod_map(axis, aod_result, land_geometries, add_colorbar_to=None):
    trend = aod_result["aod_trend"]
    distance = aod_result["distance_to_coast"]
    significant = (aod_result["MK_P"] < .05) & np.isfinite(trend)
    transform = ccrs.PlateCarree()
    vmax = map_color_limit(trend)
    levels = np.linspace(-vmax, vmax, 25)
    # Same blue-white-red palette style used in Fig. M2A.
    colors = [
        "#3B79AE", "#5F9FC0", "#8ABBD0", "#B7D3E3", "#E5EFF7",
        "#FFFFFF",
        "#FBEDE7", "#F6C9C0", "#EE9E97", "#DA848D", "#C07278",
    ]
    color_positions = np.linspace(0.0, 1.0, len(colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "M2A_like_red_blue",
        list(zip(color_positions, colors)),
        N=256,
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    plot = axis.contourf(
        trend.lon, trend.lat, trend,
        levels=levels, cmap=cmap, norm=norm,
        extend="both", transform=transform, zorder=2,
    )
    # Significance is calculated on every original grid cell (p < 0.05).
    # For display only, thin the significant cells to approximately 2° × 2°
    # by sampling the nearest native-grid point to each 2° longitude/latitude.
    sig_native = np.asarray(significant.values, dtype=bool)
    lon_native = np.asarray(significant.lon.values, dtype=float)
    lat_native = np.asarray(significant.lat.values, dtype=float)
    lon_sample = np.arange(
        np.ceil(float(lon_native.min()) / 2.0) * 2.0,
        np.floor(float(lon_native.max()) / 2.0) * 2.0 + 0.001,
        2.0,
    )
    lat_sample = np.arange(
        np.ceil(float(lat_native.min()) / 2.0) * 2.0,
        np.floor(float(lat_native.max()) / 2.0) * 2.0 + 0.001,
        2.0,
    )
    lon_index = np.unique([
        int(np.argmin(np.abs(lon_native - value))) for value in lon_sample
    ])
    lat_index = np.unique([
        int(np.argmin(np.abs(lat_native - value))) for value in lat_sample
    ])
    sampled_sig = sig_native[np.ix_(lat_index, lon_index)]

    sampled_lon, sampled_lat = np.meshgrid(
        lon_native[lon_index], lat_native[lat_index]
    )
    axis.scatter(
        sampled_lon[sampled_sig], sampled_lat[sampled_sig],
        s=2.0, marker="o", c="black", linewidths=0,
        transform=transform, zorder=5, rasterized=True,
    )
    axis.contour(
        distance.lon, distance.lat, distance,
        levels=[-400.0], colors="black", linestyles="--",
        linewidths=0.65, transform=transform,
    )
    axis.contour(
        distance.lon, distance.lat, distance,
        levels=[400.0], colors="black", linestyles="-",
        linewidths=0.65, transform=transform,
    )
    axis.add_geometries(
        land_geometries, crs=transform, facecolor="none",
        edgecolor="#303030", linewidth=0.35, zorder=4,
    )
    axis.set_extent(EAST_ASIA_EXTENT, crs=transform)
    axis.set_xticks([100, 110, 120, 130, 140], crs=transform)
    axis.set_yticks([10, 20, 30, 40], crs=transform)
    axis.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, pos: "0°" if abs(x) < 1e-8 else f"{abs(int(x))}°{'E' if x > 0 else 'W'}"
    ))
    axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, pos: "0°" if abs(y) < 1e-8 else f"{abs(int(y))}°{'N' if y > 0 else 'S'}"
    ))
    axis.tick_params(labelsize=7, direction="out")
    axis.text(
        0.01, 0.02,
        "July–November\nDashed: 400 km inland   Solid: 400 km offshore",
        transform=axis.transAxes, ha="left", va="bottom", fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        zorder=7,
    )
    if add_colorbar_to is not None:
        colorbar = add_colorbar_to.colorbar(
            plot, ax=axis, orientation="vertical", fraction=0.025, pad=0.025,
            format=_numeric_formatter(2),
        )
        colorbar.update_ticks()
        colorbar.set_label(r"Trend in AOD (decade$^{-1}$)")
        colorbar.ax.tick_params(labelsize=7)
    return plot

def restyle_cmip6_axis(axis):
    """Use a complete axis frame and keep legend clear of all curves."""
    axis.set_xlim(-400.0, 400.0)
    axis.set_xticks([-400, -200, 0, 200, 400])
    axis.set_ylim(-0.40, 0.18)
    axis.set_yticks([-0.40, -0.20, 0.00])
    era5_lines = [line for line in axis.lines if line.get_label() == 'ERA5']
    if era5_lines:
        observed = np.asarray(era5_lines[0].get_ydata(), dtype=float)
        observed = observed[np.isfinite(observed)]
        if observed.size and (observed.min() < -0.40 or observed.max() > 0.18):
            margin = max(0.025, 0.08 * np.ptp(observed))
            axis.set_ylim(min(-0.40, observed.min()-margin), max(0.18, observed.max()+margin))
            axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))

    axis.xaxis.set_major_formatter(_numeric_formatter())
    axis.yaxis.set_major_formatter(_numeric_formatter(1))
    axis.set_xlabel("Distance to coast (km)")
    axis.set_ylabel(
        r"Trend in VWS (m s$^{-1}$ decade$^{-1}$)",
        labelpad=7,
    )
    axis.tick_params(
        direction="out",
        top=False,
        right=False,
        width=0.8,
        length=3.5,
        pad=2.5,
    )
    for side in ("left", "right", "bottom", "top"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(0.8)
        axis.spines[side].set_color("#333333")
    for item in axis.texts:
        if item.get_text() == S6_REGION["season"]:
            item.set_position((0.03, 0.05))
            item.set_ha("left")
            item.set_va("bottom")
    legend = axis.get_legend()
    if legend is not None:
        handles, labels = profile_legend_handles(axis)
        legend.remove()
        axis.legend(
            handles, labels,
            loc="lower right",
            bbox_to_anchor=(0.99, 0.02),
            frameon=False,
            ncol=2,
            handlelength=1.7,
            columnspacing=0.8,
            handletextpad=0.45,
            borderaxespad=0.4,
        )
        # Extend the display axis below the curves enough for the lower-right legend.
        # Use rendered legend height so single-panel and composite outputs both fit.
        axis.figure.canvas.draw()
        renderer = axis.figure.canvas.get_renderer()
        legend = axis.get_legend()
        axis_box = axis.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        reserve = min(0.65, (legend_box.y1 - axis_box.y0 + 8.0) / axis_box.height)
        data_lines = [line for line in axis.lines
                      if line.get_label() in {'AER', 'GHG', 'NAT', 'HIST', 'ERA5'}]
        all_values = np.concatenate([np.asarray(line.get_ydata(), float) for line in data_lines])
        finite = all_values[np.isfinite(all_values)]
        if finite.size:
            lower, upper = axis.get_ylim()
            # Map the lowest plotted value to a position above the legend plus padding.
            minimum = float(finite.min())
            lower_needed = (minimum - reserve * upper) / (1.0 - reserve)
            axis.set_ylim(min(lower, lower_needed), upper)
            axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

def restyle_aod_axis(axis):
    """Remove the map annotation and anchor its geographic frame at the bottom."""
    axis.set_anchor("S")
    for item in list(axis.texts):
        item.remove()

def simplify_colorbar(colorbar, contour):
    """Use five two-decimal labels and a narrow colorbar."""
    limit = float(
        np.nanmax(np.abs(np.asarray(contour.levels, dtype=float)))
    )
    colorbar.set_ticks(np.linspace(-limit, limit, 5))
    colorbar.ax.yaxis.set_major_formatter(_numeric_formatter(2))
    colorbar.update_ticks()
    colorbar.set_label(
        r"Trend in AOD (decade$^{-1}$)",
        fontsize=8.5,
        labelpad=3,
    )
    colorbar.ax.tick_params(
        labelsize=7.0, width=0.7, length=2.4, pad=1.5
    )
    colorbar.outline.set_linewidth(0.65)

def uppercase_panel_label(fig, axis, letter, dx=0.030, dy=0.010):
    """Draw an uppercase panel label outside the upper-left corner."""
    box = axis.get_position()
    fig.text(
        max(0.006, box.x0 - dx),
        min(0.995, box.y1 + dy),
        letter.lower(),
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

def fit_map_extent_to_slot(fig, axis):
    """Adjust longitude width so the map uses the full allotted panel height."""
    slot = axis.get_position(original=True)
    figure_width, figure_height = fig.get_size_inches()
    slot_aspect = (
        slot.width * figure_width
    ) / (
        slot.height * figure_height
    )
    latitude_min, latitude_max = 10.0, 45.0
    longitude_center = 122.5
    longitude_width = slot_aspect * (
        latitude_max - latitude_min
    )
    longitude_min = max(
        100.0, longitude_center - 0.5 * longitude_width
    )
    longitude_max = min(
        145.0, longitude_center + 0.5 * longitude_width
    )
    axis.set_extent(
        [
            longitude_min, longitude_max,
            latitude_min, latitude_max,
        ],
        crs=ccrs.PlateCarree(),
    )



def make_composite(ensemble, aod_result, era5_table):
    """Use the original S6 composite as the single source of panel geometry."""
    land_geometries = load_land_geometries()
    fig = plt.figure(figsize=(7.40, 3.00), facecolor="white")
    line_axis = fig.add_axes([0.105, 0.205, 0.365, 0.685])
    map_axis = fig.add_axes(
        [0.555, 0.205, 0.335, 0.685],
        projection=ccrs.PlateCarree(),
    )
    draw_cmip6_axis(
        line_axis, ensemble, S6_REGION,
        show_ylabel=True, show_legend=True, era5_table=era5_table,
    )
    restyle_cmip6_axis(line_axis)
    contour = draw_aod_map(
        map_axis, aod_result, land_geometries,
        add_colorbar_to=None,
    )
    restyle_aod_axis(map_axis)
    fit_map_extent_to_slot(fig, map_axis)
    map_axis.set_anchor("S")
    fig.canvas.draw()
    # Re-anchor after Cartopy applies geographic aspect. This forces the map's
    # bottom X axis to coincide with panel A's bottom X axis.
    line_box = line_axis.get_position()
    map_box = map_axis.get_position()
    map_axis.set_position([
        map_box.x0,
        line_box.y0,
        map_box.width,
        map_box.height,
    ])
    map_axis.set_anchor("S")
    fig.canvas.draw()
    map_box = map_axis.get_position()
    color_axis = fig.add_axes([
        min(0.985, map_box.x1 + 0.008),
        map_box.y0,
        0.007,
        map_box.height,
    ])
    colorbar = fig.colorbar(
        contour,
        cax=color_axis,
        orientation="vertical",
        extend="both",
        format=_numeric_formatter(2),
    )
    simplify_colorbar(colorbar, contour)
    # Only the composite contains panel letters.
    uppercase_panel_label(
        fig, line_axis, "A", dx=0.034, dy=0.010
    )
    uppercase_panel_label(
        fig, map_axis, "B", dx=0.034, dy=0.010
    )
    return fig, ((line_axis,), (map_axis, color_axis))




def save_figures(ensemble, aod_result, era5_table, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, groups = make_composite(ensemble, aod_result, era5_table)
    try:
        fig.savefig(output_dir / 'FigS6_composite.png', dpi=DPI, facecolor='white',
                    bbox_inches='tight', pad_inches=.06)
        # Hide composite-only letters, then retain the exact same axes and font scale.
        labels = list(fig.texts)
        for text in labels:
            text.set_visible(False)
        all_axes = list(fig.axes)
        for filename, group in zip(('FigS6a_CMIP6_VWS.png', 'FigS6b_MERRA2_AOD.png'), groups):
            for axis in all_axes:
                axis.set_visible(axis in group)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [axis.get_tightbbox(renderer) for axis in group]
            extent = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
            fig.savefig(output_dir / filename, dpi=DPI, facecolor='white',
                        bbox_inches=extent, pad_inches=0)
            print(f'Saved: {output_dir / filename}')
        print(f'Saved: {output_dir / "FigS6_composite.png"}')
    finally:
        plt.close(fig)


def main(argv=None):
    global GSHHG_SHP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--coastline', type=Path, default=GSHHG_SHP)
    args = parser.parse_args(argv)
    GSHHG_SHP = check_coastline(args.coastline.expanduser().resolve())
    load_land_geometries.cache_clear()
    save_figures(*load_plot_data(args.data_dir.expanduser().resolve()),
                 output_dir=args.output_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
