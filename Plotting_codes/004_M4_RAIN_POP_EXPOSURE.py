#!/usr/bin/env python3
import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR/'Database/EXPOSURE'
DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Results/004_M4_TC_RAIN_POP_EXPOSURE'
DEFAULT_RAW_DATA_DIR = PROJECT_DIR/'Raw_data'
EXPOSURE_CSV_FILENAMES = ('M4_event_points.csv', 'M4_exposure_occurrences.csv',
                          'M4_grid_coordinates.csv')
GSHHG_LAND_FILE = PROJECT_DIR/'Database/GSHHG/GSHHS_f_L1_1400.shp'

GISA_YEAR_MIN, GISA_YEAR_MAX = 1985, 2021

GISA_YEAR_TICKS = [1985, 1997, 2009, 2021]

REPEAT_THRESHOLDS = (1, 2, 3)

PANEL_B_YMAX = 15000.0

PANEL_A_YMAX = 130000.0

EXPOSURE_DISTANCE_CASES = (
    ('m300_m100', '−300–−100 km', '#C25D34'),
    ('pm100', '±100 km', '#496181'),
    ('p100_p300', '100–300 km', '#6A946F'),
)

DPI = 600

FONT_SIZE = 9.5

FIG1_REFERENCE_WIDTH_IN = 6.65

FIG1_REFERENCE_FONT_PT = 7.5

FIG1_REFERENCE_PANEL_PT = 10.0

COLORBAR_THICKNESS_PT = 6.0

INSET_FONT_SIZE = FONT_SIZE

REGION_COLORS = {
    "NH (>20°N)": "#222222",
    "East Asia": "#C25D34",
    "North America": "#C25D34",
}

MAP_EXTENTS = {
    "East Asia": (100.0, 145.0, 15.0, 45.0),
    "North America": (-130.0, -65.0, 15.0, 45.0),
}

EXPOSURE_COLORS = [
    '#3B79AE', '#5F9FC0', '#8ABBD0', '#B7D3E3', '#E5EFF7',
    '#FFFFFF', '#FBEDE7', '#F6C9C0', '#EE9E97', '#DA848D', '#C07278',
]

EXPOSURE_CMAP = LinearSegmentedColormap.from_list(
    'M4_exposure_blue_red', EXPOSURE_COLORS, N=256
)

EXPOSURE_NORM = Normalize(vmin=-2, vmax=2)

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


SERIES_COLUMNS = (
    'year', 'exposure_raw', 'raw_ols_fitted', 'raw_ols_ci95_lower',
    'raw_ols_ci95_upper', 'exposure_smooth5', 'smooth5_ols_fitted',
    'smooth5_ols_ci95_lower', 'smooth5_ols_ci95_upper',
)
STAT_COLUMNS = (
    'panel', 'series', 'distance_case', 'repeat_threshold', 'slope_decade',
    'slope_ci95_lower', 'slope_ci95_upper', 'trend_p',
)
YEARS = np.arange(1985, 2022)
REGIONS = {'C': 'East Asia', 'D': 'North America'}
DISTANCE_KEYS = ('m300_m100', 'pm100', 'p100_p300')


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    return pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')


def validate_series(table):
    missing = set(SERIES_COLUMNS)-set(table.columns)
    if missing:
        raise ValueError(f'M4A series missing columns: {sorted(missing)}')
    table = table[list(SERIES_COLUMNS)].copy()
    for column in SERIES_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    table = table.sort_values('year').reset_index(drop=True)
    if not np.array_equal(table.year.values, YEARS):
        raise ValueError('M4A requires each year from 1985 to 2021 exactly once')
    for column in SERIES_COLUMNS[1:5]:
        if not np.isfinite(table[column]).all():
            raise ValueError(f'M4A has missing/nonfinite raw values: {column}')
    for column in SERIES_COLUMNS[5:]:
        if not np.isfinite(table[column].iloc[2:-2]).all():
            raise ValueError(f'M4A has missing/nonfinite complete five-year means: {column}')
        if not table[column].iloc[[0, 1, -2, -1]].isna().all():
            raise ValueError('Five-year endpoints must remain NaN; do not pad or smooth again')
    if table.exposure_raw.lt(0).any() or table.exposure_smooth5.dropna().lt(0).any():
        raise ValueError('Exposure counts must be nonnegative')
    for prefix in ('raw', 'smooth5'):
        lower, fit, upper = (table[f'{prefix}_ols_ci95_lower'],
                             table[f'{prefix}_ols_fitted'], table[f'{prefix}_ols_ci95_upper'])
        if ((lower > fit) | (fit > upper)).any():
            raise ValueError(f'Invalid mean-fit confidence band: {prefix}')
    return table


def validate_statistics(table):
    missing = set(STAT_COLUMNS)-set(table.columns)
    if missing:
        raise ValueError(f'M4 statistics missing columns: {sorted(missing)}')
    table = table[list(STAT_COLUMNS)].copy()
    for name in ('panel', 'series', 'distance_case'):
        table[name] = table[name].fillna('').astype(str).str.strip()
    for name in STAT_COLUMNS[3:]:
        table[name] = pd.to_numeric(table[name], errors='raise')
    if len(table) != 13 or table.groupby('panel').size().to_dict() != {'A': 2, 'B': 9, 'C': 1, 'D': 1}:
        raise ValueError('Expected A:2, B:9, C:1 and D:1 trend records')
    a, b = table.loc[table.panel.eq('A')], table.loc[table.panel.eq('B')]
    if set(a.series) != {'raw', 'smooth5'}:
        raise ValueError('M4A requires separate raw and smooth5 trend statistics')
    expected = {(distance, threshold) for distance in DISTANCE_KEYS for threshold in (1, 2, 3)}
    if set(zip(b.distance_case, b.repeat_threshold)) != expected:
        raise ValueError('M4B must contain three distance bands and thresholds 1-3')
    if table[['slope_decade', 'trend_p']].isna().any().any():
        raise ValueError('Every trend requires a finite slope and P value')
    if not np.isfinite(table[['slope_decade', 'trend_p']].to_numpy(float)).all():
        raise ValueError('Nonfinite slope or P value')
    if not table.trend_p.between(0, 1).all():
        raise ValueError('P values must be within [0, 1]')
    ci = table.loc[~table.panel.eq('A'), ['slope_ci95_lower', 'slope_decade', 'slope_ci95_upper']]
    if not np.isfinite(ci.to_numpy(float)).all() or (np.diff(ci.to_numpy(float), axis=1) < 0).any():
        raise ValueError('Invalid trend confidence intervals')
    return table


def plot_data_from_results(table, statistics, maps):
    """Validate in-memory tables/maps and retain the original plotting inputs."""
    table = validate_series(table)
    statistics = validate_statistics(statistics)
    coastal_series = {'years': table.year.to_numpy(int), 'raw': table.exposure_raw.to_numpy(float)}
    for series, prefix, key in [('raw', 'raw', 'raw_fit'), ('smooth5', 'smooth5', 'smooth_fit')]:
        row = statistics.loc[statistics.panel.eq('A') & statistics.series.eq(series)].iloc[0]
        coastal_series[key] = {
            'slope_decade': float(row.slope_decade), 'trend_p': float(row.trend_p),
            'fitted': table[f'{prefix}_ols_fitted'].to_numpy(float),
            'fitted_ci95_lower': table[f'{prefix}_ols_ci95_lower'].to_numpy(float),
            'fitted_ci95_upper': table[f'{prefix}_ols_ci95_upper'].to_numpy(float),
        }
    coastal_series['smooth_fit']['smoothed'] = table.exposure_smooth5.to_numpy(float)
    sensitivity = statistics.loc[statistics.panel.eq('B')].copy()
    spatial, regional_fits = {}, {}
    for panel, region in REGIONS.items():
        name = f'FigM4{panel}_trend.nc'
        dataset = maps[panel]
        if dataset.attrs.get('region') != region or dataset.attrs.get('panel') != panel:
            raise ValueError(f'{name}: wrong region/panel metadata')
        if dataset.attrs.get('significance_method') != 'mk':
            raise ValueError(f'{name}: map P values must use Mann-Kendall significance')
        if not {'trend_of_exposure', 'trend_p'}.issubset(dataset.data_vars):
            raise ValueError(f'{name}: missing trend or original MK P field')
        trend = dataset.trend_of_exposure.transpose('lat', 'lon').values.copy()
        p_values = dataset.trend_p.transpose('lat', 'lon').values.copy()
        lat, lon = dataset.lat.values.copy(), dataset.lon.values.copy()
        coordinate_edges(lat)
        coordinate_edges(lon)
        if not np.issubdtype(p_values.dtype, np.floating):
            raise ValueError(f'{name}: trend_p must contain floating-point P values')
        if np.isinf(trend).any() or np.isinf(p_values).any():
            raise ValueError(f'{name}: infinite trend or P values')
        finite_p = p_values[np.isfinite(p_values)]
        if ((finite_p < 0) | (finite_p > 1)).any():
            raise ValueError(f'{name}: P values must be within [0, 1]')
        spatial[region] = np.where(p_values < .05, trend, np.nan), lat, lon
        row = statistics.loc[statistics.panel.eq(panel)].iloc[0]
        regional_fits[region] = {name: float(row[name]) for name in (
            'slope_decade', 'slope_ci95_lower', 'slope_ci95_upper', 'trend_p')}
    return spatial, regional_fits, sensitivity, coastal_series


def load_plot_data(data_dir):
    table = read_csv(data_dir/'FigM4A_timeseries.csv')
    statistics = read_csv(data_dir/'FigM4_plot_statistics.csv')
    maps = {}
    for panel in REGIONS:
        path = data_dir/f'FigM4{panel}_trend.nc'
        require_file(path, f'panel {panel} map')
        with xr.open_dataset(path, engine='h5netcdf') as dataset:
            maps[panel] = dataset.load()
    return plot_data_from_results(table, statistics, maps)


def load_calculation_module():
    path = PROJECT_DIR / 'Calculate_codes/004_M4_RAIN_POP_EXPOSURE.py'
    spec = importlib.util.spec_from_file_location('_m4_calculation_for_plot', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_exposure_csv_plot_data(raw_data_dir, csv_dir):
    calculation = load_calculation_module()
    calculation.configure_inputs(raw_data_dir)
    return plot_data_from_results(*calculation.calculate_from_exposure_csv(csv_dir))


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')


def coordinate_edges(values, circular=False):
    values = np.asarray(values, dtype=float)
    working = np.rad2deg(np.unwrap(np.deg2rad(values))) if circular else values.copy()
    if len(working) == 1:
        return np.array([working[0] - 0.05, working[0] + 0.05])
    edges = np.empty(len(working) + 1)
    edges[1:-1] = (working[:-1] + working[1:]) / 2
    edges[0] = working[0] - (working[1] - working[0]) / 2
    edges[-1] = working[-1] + (working[-1] - working[-2]) / 2
    return edges


def draw_annual_exposure(axis, coastal_series):
    """Original and five-year-mean series, OLS fits, and 95% mean-fit bands."""
    years, raw = coastal_series['years'], coastal_series['raw']
    raw_fit, smooth_fit = coastal_series['raw_fit'], coastal_series['smooth_fit']
    raw_color, smooth_color = '#959595', '#C25D34'

    def p_label(value):
        if value < .01:
            return r'$P$ < 0.01'
        if value < .05:
            return r'$P$ < 0.05'
        return rf'$P$ = {value:.3f}'

    def trend_label(name, fit):
        return (f"{name}: {fit['slope_decade']:.0f} "
                + r'decade$^{-1}$, ' + p_label(fit['trend_p']))

    # Fit uncertainty for each series separately; incomplete smooth windows stay NaN.
    for fit, color in [(raw_fit, raw_color), (smooth_fit, smooth_color)]:
        axis.fill_between(years, fit['fitted_ci95_lower'], fit['fitted_ci95_upper'],
                          color=color, alpha=.16, linewidth=0, zorder=1,
                          label='_nolegend_')
    raw_line, = axis.plot(years, raw, color=raw_color, lw=1., zorder=2,
                          label='Raw series')
    smooth_line, = axis.plot(years, smooth_fit['smoothed'], color=smooth_color,
                             lw=1.5, zorder=3, label='5-year mean series')
    raw_trend, = axis.plot(years, raw_fit['fitted'], color=raw_color, lw=1.2,
                           ls='--', zorder=4, label=trend_label('Raw trend', raw_fit))
    smooth_trend, = axis.plot(years, smooth_fit['fitted'], color=smooth_color,
                              lw=1.3, ls='--', zorder=4,
                              label=trend_label('5-year mean trend', smooth_fit))
    axis.set_xlim(GISA_YEAR_MIN-1, GISA_YEAR_MAX+1)
    axis.set_xticks(GISA_YEAR_TICKS)
    axis.set_xlabel('Year')
    axis.set_ylabel('Exposure to CLTC precipitation')
    # Preserve the source figure's 0-130000 limits and ticks at 0, 40, 80, 120.
    axis.set_ylim(0., PANEL_A_YMAX)
    axis.set_yticks(np.arange(0., PANEL_A_YMAX+1., 40000.))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value/1000:g}'))
    axis.text(0, 1.015, '×10³', transform=axis.transAxes,
              fontsize=FONT_SIZE, ha='left', va='bottom')
    axis.legend(handles=[raw_trend, smooth_trend, raw_line, smooth_line],
                loc='upper left', bbox_to_anchor=(-0.01, 1.0),
                frameon=False, fontsize=FONT_SIZE, ncol=1,
                handlelength=1.7, handletextpad=.5, labelspacing=.25, borderaxespad=.3)
    axis.tick_params(direction='out', length=3, width=.7, labelsize=FONT_SIZE)


def slope_errorbars(value, lower, upper):
    """Use the saved lower and upper trend confidence bounds."""
    if not np.isfinite([value, lower, upper]).all() or lower>value or upper<value:
        raise ValueError('Invalid slope estimate or confidence bounds')
    return np.array([[value-lower], [upper-value]])


def draw_repeated_exposure(axis, trends):
    """Minimum exposure frequency on x; one color per distance band."""
    table = trends.copy()
    frequencies = np.asarray(REPEAT_THRESHOLDS, dtype=int)
    bar_width = .34 * .56  # Preserve the narrow bars with room for three bands.
    cases = EXPOSURE_DISTANCE_CASES
    offsets = (np.arange(len(cases)) - (len(cases) - 1) / 2) * bar_width * 1.12
    for (key, label, color), offset in zip(cases, offsets):
        sub = (table[table.distance_case.eq(key)]
               .set_index('repeat_threshold').reindex(frequencies))
        if sub[['slope_decade', 'slope_ci95_lower', 'slope_ci95_upper', 'trend_p']].isna().any().any():
            raise ValueError(f'Missing panel B trend/CI/P data: {key}, repeat {frequencies[0]}–{frequencies[-1]}')
        for index, frequency in enumerate(frequencies):
            record = sub.loc[frequency]
            value = float(record.slope_decade)
            lower = float(record.slope_ci95_lower)
            upper = float(record.slope_ci95_upper)
            error = slope_errorbars(value, lower, upper)
            p_value = float(record.trend_p)
            xpos = frequency + offset
            axis.bar(xpos, value, width=bar_width,
                     yerr=error, color=color,
                     label=label if index == 0 else '_nolegend_',
                     error_kw={'ecolor': '#222', 'elinewidth': .8,
                                'capsize': 2, 'capthick': .8})
            mark = '**' if p_value < .05 else '*' if p_value < .1 else ''
            if mark:
                end = upper if value >= 0 else lower
                axis.annotate(mark, (xpos, end),
                              xytext=(0, 4 if value >= 0 else -4),
                              textcoords='offset points', ha='center',
                              va='bottom' if value >= 0 else 'top',
                              fontsize=FONT_SIZE, fontweight='bold', color='black')

    axis.axhline(0, color='#777', lw=.7)
    axis.set_xlim(frequencies.min()-.55, frequencies.max()+.55)
    axis.set_xticks(frequencies)
    axis.set_xlabel('Threshold of consecutive exposure frequency')
    axis.set_ylabel(r'Trend in exposure (decade$^{-1}$)')
    # Keep the requested upper limit at 15 on the displayed x10^3 scale.
    plotted = table[table.distance_case.isin([case[0] for case in cases])]
    lower = float(plotted.slope_ci95_lower.min())
    upper = float(plotted.slope_ci95_upper.max())
    span = max(upper-lower, 1.0)
    ymin = min(-1000.0, lower-.08*span)
    axis.set_ylim(ymin, PANEL_B_YMAX)
    first_tick = int(np.ceil(ymin/5000.0))*5000
    axis.set_yticks(np.arange(first_tick, PANEL_B_YMAX+1, 5000.0))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: f'{value / 1000:g}')
    )
    axis.text(0, 1.015, r'×10³', transform=axis.transAxes,
              fontsize=FONT_SIZE, ha='left', va='bottom')
    axis.legend(frameon=False, loc='upper right', ncol=1,
                fontsize=FONT_SIZE)
    axis.tick_params(direction='out', length=3, width=.7,
                     labelsize=FONT_SIZE)
    axis.spines[['top', 'right']].set_visible(True)


def draw_regional_trend_inset(map_axis, regional_fits, region):
    """Place one narrow trend bar in the open lower corner of each map."""
    fit = regional_fits[region]
    # [left, bottom, width, height] relative to the map axes.
    # Narrower insets leave more of each regional map visible.
    # Both insets have the same physical width; D's map spans 65°
    # of longitude while C's spans 45°.
    c_width = .18 * (45.0 / 65.0)
    position = ([.79, .045, .18, .43] if region == 'East Asia'
                else [.13, .045, c_width, .43])
    box = map_axis.get_position()
    inset = map_axis.figure.add_axes([
        box.x0 + position[0] * box.width,
        box.y0 + position[1] * box.height,
        position[2] * box.width,
        position[3] * box.height,
    ])
    inset.set_zorder(10)
    inset.patch.set_facecolor('none')  # Fully transparent: underlying map remains visible.
    inset.patch.set_alpha(0.0)
    slope = fit['slope_decade']
    lower, upper = fit['slope_ci95_lower'], fit['slope_ci95_upper']
    error = slope_errorbars(slope, lower, upper)
    inset.bar(0, slope, width=.34, color=REGION_COLORS[region],
              yerr=error,
              error_kw={'ecolor': '#202020', 'elinewidth': .75,
                         'capsize': 2, 'capthick': .75})
    inset.axhline(0, color='#777777', lw=.65)
    inset.set_xlim(-.75, .75)
    span = max(upper-lower, 1.0)
    inset.set_ylim(min(-2500, lower-.12*span), max(11000, upper+.18*span))
    inset.set_yticks([0, 5000, 10000], ['0', '5', '10'])
    inset.set_xticks([])
    inset.set_ylabel('Trend in exposure\n(decade$^{-1}$)',
                    fontsize=INSET_FONT_SIZE, labelpad=1.5)
    inset.tick_params(direction='out', length=2, width=.6,
                      labelsize=INSET_FONT_SIZE, pad=1,
                      top=False, right=False)
    inset.text(.02, 1.04, r'$\times 10^{3}$', transform=inset.transAxes,
               ha='left', va='bottom', fontsize=INSET_FONT_SIZE)
    mark = '**' if fit['trend_p'] < .05 else '*' if fit['trend_p'] < .1 else ''
    if mark:
        top = upper if slope >= 0 else lower
        inset.annotate(mark, (0, top), xytext=(0, 3 if slope >= 0 else -3),
                       textcoords='offset points', ha='center',
                       va='bottom' if slope >= 0 else 'top',
                       fontsize=INSET_FONT_SIZE, fontweight='bold', color='black')
    for spine in inset.spines.values():
        spine.set_linewidth(.6)
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    # Keep the enlarged vertical label inside the map's lower edge.
    inset.yaxis.label.set_y(.62)
    return inset


@lru_cache(maxsize=1)
def gshhg_land_geometries():
    """Read the prefiltered GSHHG land polygons once for all map panels."""
    require_file(GSHHG_LAND_FILE, 'GSHHG land polygons')
    reader = shapereader.Reader(str(GSHHG_LAND_FILE))
    try:
        return tuple(reader.geometries())
    finally:
        reader.close()


def draw_map(axis, region, lat, lon, trend):
    axis.set_extent(MAP_EXTENTS[region], crs=ccrs.PlateCarree())
    axis.set_aspect('equal', adjustable='box')
    axis.set_facecolor('white')
    # Fill GSHHG land polygons without drawing their coastline boundaries.
    axis.add_geometries(
        gshhg_land_geometries(), crs=ccrs.PlateCarree(),
        facecolor='#D2D2D2', edgecolor='none', linewidth=0, zorder=0,
    )
    artist = axis.pcolormesh(
        coordinate_edges(lon), coordinate_edges(lat),
        np.ma.masked_invalid(trend), shading='flat',
        cmap=EXPOSURE_CMAP, norm=EXPOSURE_NORM,
        transform=ccrs.PlateCarree(), rasterized=True, zorder=2,
    )
    west, east, _, _ = MAP_EXTENTS[region]
    axis.set_xticks(np.arange(np.ceil(west / 10) * 10, east + .01, 10),
                    crs=ccrs.PlateCarree())
    axis.set_yticks([20, 30, 40], crs=ccrs.PlateCarree())
    axis.xaxis.set_major_formatter(LongitudeFormatter())
    axis.yaxis.set_major_formatter(LatitudeFormatter())
    axis.tick_params(direction='out', length=3, width=.7, pad=2,
                     labelsize=FONT_SIZE, bottom=True, left=True)
    if not np.isfinite(trend).any():
        axis.text(.5, .5, 'No cells with $P<0.05$',
                  transform=axis.transAxes, ha='center', va='center',
                  fontsize=FONT_SIZE)
    return artist


def match_figure1_text_size(fig, pad_inches):
    """Calibrate displayed text to Fig. 1; panel letters retain Fig. 1's larger size."""
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        export_width = fig.get_tightbbox(renderer).width+2*pad_inches
        scale = export_width/FIG1_REFERENCE_WIDTH_IN
        texts = [text for text in fig.findobj(Text)
                 if text.get_visible() and text.get_text()]
        if not texts:
            break
        targets = [(FIG1_REFERENCE_PANEL_PT if text.get_gid() == 'panel-label'
                    else FIG1_REFERENCE_FONT_PT)*scale for text in texts]
        if max(abs(text.get_fontsize()-target) for text, target in zip(texts, targets)) < .005:
            break
        for text, target in zip(texts, targets):
            text.set_fontsize(target)
    fig.canvas.draw()
    return FIG1_REFERENCE_FONT_PT*scale


def add_upper_right_colorbar(fig, artist, map_axis):
    """A thin horizontal bar; its right triangular tip meets the map's right spine."""
    box = map_axis.get_position()
    width = min(2.2/fig.get_figwidth(), .52*box.width)
    height = COLORBAR_THICKNESS_PT/72.0/fig.get_figheight()
    left_tip = box.x1-width
    bottom = box.y1+(FONT_SIZE+8.0)/72.0/fig.get_figheight()
    cax = fig.add_axes([left_tip, bottom, width, height])
    bar = fig.colorbar(artist, cax=cax, orientation='horizontal',
                       ticks=[-2, -1, 0, 1, 2], extend='both', extendfrac=.04)
    bar.ax.tick_params(labelsize=FONT_SIZE, length=2.5, width=.7, pad=2)
    bar.outline.set_linewidth(.7)
    # The allocated cax box includes the triangular ends; their vertical center is here.
    title = fig.text(left_tip-.10/fig.get_figwidth(), bottom+height/2,
                     r'Trend in exposure (decade$^{-1}$)',
                     ha='right', va='center', fontsize=FONT_SIZE)
    return cax, title


def add_standalone_colorbar(fig, map_axis, font_size):
    """Give standalone C a scale without widening or resizing its map.

    The map is narrower than D. Place the title above the scale so that
    both remain within the map width at the composite's calibrated size.
    """
    box = map_axis.get_position()
    height = COLORBAR_THICKNESS_PT/72.0/fig.get_figheight()
    bottom = box.y1+(FONT_SIZE+8.0)/72.0/fig.get_figheight()
    cax = fig.add_axes([box.x0, bottom, box.width, height])
    mappable = mpl.cm.ScalarMappable(norm=EXPOSURE_NORM, cmap=EXPOSURE_CMAP)
    bar = fig.colorbar(mappable, cax=cax, orientation='horizontal',
                       ticks=[-2, -1, 0, 1, 2], extend='both', extendfrac=.04)
    bar.ax.tick_params(labelsize=font_size, length=2.5, width=.7, pad=2)
    bar.outline.set_linewidth(.7)
    title = fig.text((box.x0+box.x1)/2, bottom+height+4.0/72.0/fig.get_figheight(),
                     r'Trend in exposure (decade$^{-1}$)',
                     ha='center', va='bottom', fontsize=font_size)
    return cax, title


def make_m4_composite(regional_fits, spatial, sensitivity, coastal_series):
    """Two columns: A/C and B/D share both vertical boundaries."""
    fig = plt.figure(figsize=(10.5, 7.0), facecolor='white')
    map_height, map_bottom = .30, .10
    left, column_gap = .085, .065

    def map_width(region):
        west, east, south, north = MAP_EXTENTS[region]
        return (map_height*fig.get_figheight()/fig.get_figwidth()
                * (east-west)/(north-south))

    left_width = map_width('East Asia')
    right_left = left+left_width+column_gap
    axis_c = fig.add_axes([left, map_bottom, left_width, map_height],
                          projection=ccrs.PlateCarree())
    axis_d = fig.add_axes([right_left, map_bottom, map_width('North America'), map_height],
                          projection=ccrs.PlateCarree())
    insets = []
    for axis, region in [(axis_c, 'East Asia'), (axis_d, 'North America')]:
        trend, lat, lon = spatial[region]
        artist = draw_map(axis, region, lat, lon, trend)
        insets.append(draw_regional_trend_inset(axis, regional_fits, region))
    # Retain latitude tick marks on D and hide only their values.
    axis_d.tick_params(axis='y', left=True, right=False, labelleft=False, labelright=False)
    fig.canvas.draw()
    c_box, d_box = axis_c.get_position(), axis_d.get_position()
    cax, colorbar_title = add_upper_right_colorbar(fig, artist, axis_d)
    axis_a = fig.add_axes([c_box.x0, .535, c_box.width, .32])
    axis_b = fig.add_axes([d_box.x0, .535, d_box.width, .32])
    draw_annual_exposure(axis_a, coastal_series)
    draw_repeated_exposure(axis_b, sensitivity)
    axes = [axis_a, axis_b, axis_c, axis_d]
    for axis, letter in zip(axes, 'ABCD'):
        box = axis.get_position()
        fig.text(box.x0-.038, box.y1+.025, letter.lower(),
                 fontsize=FONT_SIZE, fontweight='bold', ha='left', va='bottom',
                 gid='panel-label')
    match_figure1_text_size(fig, pad_inches=.03)
    groups = {
        'A': ([axis_a], []),
        'B': ([axis_b], []),
        'C': ([axis_c, insets[0]], []),
        'D': ([axis_d, insets[1], cax], [colorbar_title]),
    }
    return fig, groups


def save_panel(fig, group, path):
    """Export selected artists without recomputing layout or font sizes."""
    selected_axes, selected_texts = group
    visibility = [(artist, artist.get_visible()) for artist in [*fig.axes, *fig.texts]]
    try:
        for artist, _ in visibility:
            artist.set_visible(artist in selected_axes or artist in selected_texts)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [axis.get_tightbbox(renderer) for axis in selected_axes]
        boxes.extend(text.get_window_extent(renderer) for text in selected_texts)
        bounds = Bbox.union([box for box in boxes if box is not None])
        bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.035)
        fig.savefig(path, dpi=DPI, facecolor='white', bbox_inches=bounds, pad_inches=0)
    finally:
        for artist, visible in visibility:
            artist.set_visible(visible)


def save_figures(spatial, regional_fits, sensitivity, coastal_series, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, groups = make_m4_composite(regional_fits, spatial, sensitivity, coastal_series)
    try:
        # Save the accepted composite before adding a standalone-only C colorbar.
        fig.savefig(output_dir/'FigM4_composite.png', dpi=DPI,
                    facecolor='white', bbox_inches='tight', pad_inches=.03)
        font_size = groups['D'][1][0].get_fontsize()
        cax, title = add_standalone_colorbar(fig, groups['C'][0][0], font_size)
        groups['C'][0].append(cax)
        groups['C'][1].append(title)
        for panel, group in groups.items():
            save_panel(fig, group, output_dir/f'FigM4{panel}.png')
    finally:
        plt.close(fig)
    print('Saved formal FigM4A.png, FigM4B.png, FigM4C.png, FigM4D.png and FigM4_composite.png', flush=True)


def main(argv=None):
    global GSHHG_LAND_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--coastline', type=Path, default=GSHHG_LAND_FILE)
    parser.add_argument('--raw-data-dir', type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument('--exposure-csv-dir', type=Path,
                        help='Use the complete time-by-time urban-grid exposure CSVs explicitly.')
    parser.add_argument('--use-database', action='store_true',
                        help='Force the published Database inputs even when exposure CSVs exist.')
    args = parser.parse_args(argv)
    if args.exposure_csv_dir is not None and args.use_database:
        parser.error('--exposure-csv-dir and --use-database cannot be combined.')
    GSHHG_LAND_FILE = args.coastline.expanduser().resolve()
    for suffix in ('.shp', '.shx', '.dbf'):
        path = GSHHG_LAND_FILE.with_suffix(suffix)
        if not path.is_file() and not path.with_suffix(suffix.upper()).is_file():
            raise FileNotFoundError(f'Missing shapefile component: {path}')
    gshhg_land_geometries.cache_clear()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    csv_dir = (args.exposure_csv_dir.expanduser().resolve() if args.exposure_csv_dir is not None
               else raw_data_dir / 'EXPOSURE_CSV')
    use_csv = (not args.use_database and (args.exposure_csv_dir is not None
               or all((csv_dir / name).is_file() for name in EXPOSURE_CSV_FILENAMES)))
    if use_csv:
        print(f'Recomputing M4 statistics from time-by-time urban-grid exposures: {csv_dir}')
        plot_data = load_exposure_csv_plot_data(raw_data_dir, csv_dir)
    else:
        print(f'Plotting data: {data_dir}')
        plot_data = load_plot_data(data_dir)
    save_figures(*plot_data, output_dir=output_dir)
    print(f'Saved four panel PNGs and one composite PNG to: {output_dir}')


if __name__ == '__main__':
    main()
