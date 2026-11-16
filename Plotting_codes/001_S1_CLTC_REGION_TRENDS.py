#!/usr/bin/env python3
import argparse
import importlib.util
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = PROJECT_DIR/'Database/CLTC/FigS1_plot_data.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Results/001_S1_CLTC_REGION_TRENDS'
DEFAULT_RAW_DATA_DIR = PROJECT_DIR/'Raw_data'
EXPOSURE_CSV_FILENAME = 'S1D_point_population_exposure.csv'



DPI = 600

FIGSIZE_SMALL = (2.50, 2.12)
FIGSIZE_COMPOSITE = (5.00, 4.68)
PANEL_TOP = 0.91
EXPOSURE_X_LIMITS = (-0.65, 2.65)

BAR_WIDTH = 0.62

BAR_SPACING = 1.00

BAR_COLORS = ["#9DB7D5", "#5F83AD", "#244A73"]




FONT_SIZE = 7.0

AXIS_LABEL_SIZE = 7.5

TICK_LABEL_SIZE = 7.0

ANNOTATION_SIZE = 7.0


PANEL_LABEL_SIZE = 10.0

EXPOSURE_DISTANCE_BINS = (
    ("-300 to -100", "−300–−100"),
    ("-100 to 100", "±100"),
    ("100 to 300", "100–300"),
)

EXPOSURE_COLORS = {
    "-300 to -100": "#9DB7D5",
    "-100 to 100": "#5F83AD",
    "100 to 300": "#244A73",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.linewidth": 0.7,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

GROUP_LABELS = ('>20°N', '0–20°N', 'SH')

DISTANCE_LABELS = ('−300–−100', '±100', '100–300')

BASIN_LABELS = ('WNP', 'EP', 'WNA')

PLOT_COLUMNS = (
    'Panel', 'X_Label', 'Metric', 'Trend_per_Decade', 'CI95_per_Decade',
    'MK_P', 'Unit', 'Mean_point_exposure_persons', 'CI95_low', 'CI95_high',
)

def validate_plot_data(table):
    """Validate the ten plotting columns and return A/B/C/D category/metric order.

    MK P values are finite for A-C and blank for D. Numerical values
    retain their original units: trend CIs are half-widths, while panel D CIs
    are absolute lower and upper bounds. No trend or interval is recalculated.
    """
    if table.columns.duplicated().any():
        raise ValueError('S1 plotting data contain duplicate column names')
    missing = set(PLOT_COLUMNS) - set(table.columns)
    extra = set(table.columns) - set(PLOT_COLUMNS)
    if missing or extra:
        raise ValueError(
            f'S1 requires exactly ten plotting columns; '
            f'missing={sorted(missing)}, unexpected={sorted(extra)}'
        )
    table = table.loc[:, list(PLOT_COLUMNS)].copy()
    # Accept the former 12-row label spelling without changing its statistics.
    is_latitude_panel = table['Panel'].eq('A')
    table.loc[is_latitude_panel, 'X_Label'] = table.loc[
        is_latitude_panel, 'X_Label'
    ].replace({'NH (> 20°)': '>20°N', 'NH (≤ 20°)': '0–20°N'})
    counts = table['Panel'].value_counts(dropna=False).to_dict()
    if len(table) != 12 or counts != {'A': 3, 'B': 3, 'C': 3, 'D': 3}:
        raise ValueError('S1 requires exactly 12 rows: A=3, B=3, C=3, D=3; regenerate the former 18-row table with the S1 calculation script')

    keys = ['Panel', 'X_Label', 'Metric']
    expected = (
        [('A', label, 'CLTC count') for label in GROUP_LABELS]
        + [('B', label, 'CLTC count') for label in BASIN_LABELS]
        + [('C', label, 'CLTC count') for label in DISTANCE_LABELS]
        + [('D', label, 'Mean point exposure') for label in DISTANCE_LABELS]
    )
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError('S1 panel/category/metric keys must be present and unique')
    observed = set(table[keys].itertuples(index=False, name=None))
    if observed != set(expected):
        raise ValueError('S1 has unexpected or missing panel/category/metric combinations')

    expected_units = {
        'CLTC count': 'decade^-1',
        'Mean point exposure': 'persons per 3-h TC point',
    }
    if not table['Unit'].eq(table['Metric'].map(expected_units)).all():
        raise ValueError('S1 units do not match the original plotted metrics')

    is_exposure = table['Panel'].eq('D')
    table['MK_P'] = pd.to_numeric(table['MK_P'], errors='raise')
    p_values = table.loc[~is_exposure, 'MK_P']
    if not np.isfinite(p_values.to_numpy(dtype=float)).all() or not p_values.between(0, 1).all():
        raise ValueError('Panels A-C require genuine finite MK P values within [0, 1]')
    if not table.loc[is_exposure, 'MK_P'].isna().all():
        raise ValueError('Panel D uses bootstrap intervals and must leave MK_P blank')

    trend_columns = ['Trend_per_Decade', 'CI95_per_Decade']
    exposure_columns = ['Mean_point_exposure_persons', 'CI95_low', 'CI95_high']
    for column in trend_columns + exposure_columns:
        table[column] = pd.to_numeric(table[column], errors='raise')
    trends = table.loc[~is_exposure, trend_columns]
    exposure = table.loc[is_exposure, exposure_columns]
    if not np.isfinite(trends.to_numpy(dtype=float)).all():
        raise ValueError('Panels A-C require finite trends and CI half-widths')
    if trends['CI95_per_Decade'].lt(0).any():
        raise ValueError('Trend CI half-widths must be nonnegative')
    if not np.isfinite(exposure.to_numpy(dtype=float)).all():
        raise ValueError('Panel D requires finite means and both confidence bounds')
    if (exposure['CI95_low'].lt(0).any()
            or exposure['CI95_low'].gt(exposure['Mean_point_exposure_persons']).any()
            or exposure['Mean_point_exposure_persons'].gt(exposure['CI95_high']).any()):
        raise ValueError('Panel D requires 0 <= lower CI <= mean <= upper CI')
    if (not table.loc[is_exposure, trend_columns].isna().all().all()
            or not table.loc[~is_exposure, exposure_columns].isna().all().all()):
        raise ValueError('Numerical fields not used by a panel must remain blank')

    order = pd.MultiIndex.from_tuples(expected, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(PLOT_COLUMNS)]



def plot_data_from_table(table):
    """Read the three count-trend panels and equal-TC exposure statistics."""
    table = validate_plot_data(table)
    count_results = {}
    for panel in ('A', 'B', 'C'):
        count_results[panel] = [
            {'Trend_events_decade': row.Trend_per_Decade,
             'CI95_events_decade': row.CI95_per_Decade,
             'MK_P': row.MK_P}
            for row in table.loc[table.Panel.eq(panel)].itertuples(index=False)
        ]
    exposure = table.loc[table.Panel.eq('D')].rename(columns={
        'X_Label': 'label', 'Mean_point_exposure_persons': 'mean',
        'CI95_low': 'ci_low', 'CI95_high': 'ci_high'}).copy()
    exposure['distance_key'] = exposure.label.map({label: key for key, label in EXPOSURE_DISTANCE_BINS})
    return count_results, exposure


def load_plot_data(data_file):
    if not data_file.is_file():
        raise FileNotFoundError(f'Missing plotting data: {data_file}. Run the S1 calculation script first.')
    table = pd.read_csv(data_file, encoding='utf-8-sig', float_precision='round_trip')
    return plot_data_from_table(table)


def load_calculation_module():
    path = PROJECT_DIR / 'Calculate_codes/001_S1_CLTC_REGION_TRENDS.py'
    spec = importlib.util.spec_from_file_location('_s1_calculation_for_plot', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_exposure_csv_plot_data(raw_data_dir, csv_dir):
    path = csv_dir / EXPOSURE_CSV_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing complete time-by-time population exposure CSV: {path}')
    calculation = load_calculation_module()
    table = calculation.calculate_from_exposure_csv(raw_data_dir, csv_dir)
    return plot_data_from_table(table)


def significance_mark(p_value):
    # Preserve the significance symbols used by the existing plotcodes figures.
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""

def draw_count_trends(axis, results, labels, y_limits, y_ticks, xlabel=None):
    slopes = np.asarray([item['Trend_events_decade'] for item in results])
    errors = np.asarray([item['CI95_events_decade'] for item in results])
    positions = np.arange(len(labels), dtype=float)*BAR_SPACING
    axis.bar(positions, slopes, yerr=errors, width=BAR_WIDTH, color=BAR_COLORS,
             edgecolor='none', error_kw={'ecolor': '#222222', 'elinewidth': .65,
                                        'capthick': .65, 'capsize': 2.0})
    span = max(float(np.max(slopes+errors))-min(0., float(np.min(slopes-errors))), .5)
    for position, slope, error, result in zip(positions, slopes, errors, results):
        mark = significance_mark(result['MK_P'])
        if mark:
            axis.text(position, slope+error+.025*span, mark,
                      ha='center', va='bottom', fontsize=ANNOTATION_SIZE,
                      fontweight='bold', clip_on=False)
    axis.set_ylim(*y_limits)
    axis.set_yticks(y_ticks)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.axhline(0, color='#222222', linewidth=.55, zorder=0)
    axis.set_xticks(positions, labels)
    axis.set_xlim(*EXPOSURE_X_LIMITS)
    axis.set_ylabel(r'Trend in CLTC count (decade$^{-1}$)')
    if xlabel:
        axis.set_xlabel(xlabel)
    axis.tick_params(direction='out', length=3, width=.7, pad=2)
    axis.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
    axis.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color('#222222')
        axis.spines[side].set_linewidth(.7)



def shared_event_axis(results_a, results_c):
    """Return one common y range and tick array for panels A and C."""
    combined = list(results_a) + list(results_c)
    slopes = np.asarray(
        [item["Trend_events_decade"] for item in combined], dtype=float
    )
    errors = np.asarray(
        [item["CI95_events_decade"] for item in combined], dtype=float
    )
    data_low = float(np.nanmin(np.minimum(slopes - errors, 0.0)))
    data_high = float(np.nanmax(np.maximum(slopes + errors, 0.0)))
    span = max(data_high - data_low, 0.5)
    requested_high = data_high + 0.12 * span
    lower_limit = -1.0
    upper_limit = float(max(0, math.ceil(requested_high)))
    ticks = np.arange(lower_limit, upper_limit + 0.5, 1.0)
    return (lower_limit, upper_limit), ticks




def draw_panel_d(axis, table):
    means = table["mean"].to_numpy(float) / 1_000_000.0
    lower = (table["mean"] - table["ci_low"]).to_numpy(float) / 1_000_000.0
    upper = (table["ci_high"] - table["mean"]).to_numpy(float) / 1_000_000.0
    x = np.arange(len(table))
    axis.bar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        width=BAR_WIDTH,
        color=[EXPOSURE_COLORS[key] for key in table["distance_key"]],
        edgecolor="none",
        error_kw={
            "ecolor": "#222222",
            "elinewidth": 0.7,
            "capsize": 2,
            "capthick": 0.7,
        },
    )
    axis.set_xticks(x, table["label"])
    axis.set_xlabel("Distance to coast (km)")
    axis.set_ylabel("Mean exposure per TC record")
    axis.text(
        0.0, 1.015, r"$\times 10^6$",
        transform=axis.transAxes, ha="left", va="bottom", fontsize=TICK_LABEL_SIZE,
        clip_on=False,
    )
    axis.set_yticks(
        [0.0, 0.4, 0.8, 1.2, 1.6],
        ["0", "0.4", "0.8", "1.2", "1.6"],
    )
    axis.set_xlim(*EXPOSURE_X_LIMITS)
    axis.margins(y=0.12)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7, pad=2)



def draw_panel(axis, letter, count_results, exposure):
    if letter == 'D':
        draw_panel_d(axis, exposure)
    elif letter == 'B':
        draw_count_trends(axis, count_results['B'], BASIN_LABELS, (-1, 2), [-1, 0, 1, 2])
    else:
        limits, ticks = shared_event_axis(count_results['A'], count_results['C'])
        labels = GROUP_LABELS if letter == 'A' else DISTANCE_LABELS
        draw_count_trends(axis, count_results[letter], labels, limits, ticks,
                          xlabel='Distance to coast (km)' if letter == 'C' else None)


def make_composite(count_results, exposure):
    """Match the notebook's four equal-size, aligned panel rectangles."""
    figure = plt.figure(figsize=FIGSIZE_COMPOSITE, facecolor='white')
    panel_width, panel_height = FIGSIZE_SMALL
    total_width, total_height = FIGSIZE_COMPOSITE
    row_height = total_height / 2.0
    for index, letter in enumerate('ABCD'):
        left = (index % 2) * panel_width
        bottom = (1 - index // 2) * row_height
        axis = figure.add_axes([
            (left + 0.27 * panel_width) / total_width,
            (bottom + 0.22 * panel_height) / total_height,
            (0.924 - 0.27) * panel_width / total_width,
            (PANEL_TOP - 0.22) * panel_height / total_height,
        ])
        draw_panel(axis, letter, count_results, exposure)
        figure.text((left + 0.27 * panel_width - 0.35) / total_width,
                    (bottom + PANEL_TOP * panel_height + 0.07) / total_height,
                    letter.lower(), ha='left', va='bottom', fontsize=PANEL_LABEL_SIZE,
                    fontweight='bold', color='#111111')
    return figure



def save_panel(letter, count_results, exposure, path):
    """Save every standalone panel at the same physical size and font scale."""
    figure, axis = plt.subplots(figsize=FIGSIZE_SMALL, facecolor='white')
    figure.subplots_adjust(left=0.27, right=0.924, bottom=0.22, top=PANEL_TOP)
    try:
        draw_panel(axis, letter, count_results, exposure)
        figure.savefig(path, dpi=DPI, facecolor='white')
    finally:
        plt.close(figure)



def save_figures(count_results, exposure, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        'A': 'FigS1a_latitude_region_trends.png',
        'B': 'FigS1b_basin_trends_NH20.png',
        'C': 'FigS1c_distance_band_trends_NH20.png',
        'D': 'FigS1d_mean_point_exposure_NH20.png',
    }
    figure = make_composite(count_results, exposure)
    try:
        figure.savefig(output_dir/'FigS1_composite.png', dpi=DPI, facecolor='white')
    finally:
        plt.close(figure)
    for letter, filename in filenames.items():
        save_panel(letter, count_results, exposure, output_dir/filename)
    print(f'Saved four panels and one composite PNG: {output_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--raw-data-dir', type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument('--exposure-csv-dir', type=Path,
                        help='Use the complete time-by-time population exposure CSV explicitly.')
    parser.add_argument('--use-database', action='store_true',
                        help='Force the published Database table even when exposure CSVs exist.')
    args = parser.parse_args(argv)
    if args.exposure_csv_dir is not None and args.use_database:
        parser.error('--exposure-csv-dir and --use-database cannot be combined.')
    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    csv_dir = (args.exposure_csv_dir.expanduser().resolve() if args.exposure_csv_dir is not None
               else raw_data_dir / 'EXPOSURE_CSV')
    use_csv = (not args.use_database and (args.exposure_csv_dir is not None
               or (csv_dir / EXPOSURE_CSV_FILENAME).is_file()))
    if use_csv:
        print(f'Recomputing S1 statistics from time-by-time population exposures: {csv_dir}')
        plot_data = load_exposure_csv_plot_data(raw_data_dir, csv_dir)
    else:
        print(f'Plotting data: {data_file}')
        plot_data = load_plot_data(data_file)
    save_figures(*plot_data, output_dir=output_dir)


if __name__ == '__main__':
    main()
