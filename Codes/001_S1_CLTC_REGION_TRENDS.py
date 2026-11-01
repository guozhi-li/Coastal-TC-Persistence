#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = PROJECT_DIR/'Database/CLTC/FigS1_plot_data.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Results/001_S1_CLTC_REGION_TRENDS'



DPI = 600

FIGSIZE_COMPOSITE = (6.80, 5.10)

BAR_WIDTH = 0.62

BAR_SPACING = 1.00

BAR_COLORS = ["#9DB7D5", "#5F83AD", "#244A73"]

DURATION_COLORS = ("#789BC1", "#244A73")

FRACTION_BAR_COLOR = "#D69E79"

FRACTION_ACCENT_COLOR = "#C25D34"

FONT_SIZE = 7.0

AXIS_LABEL_SIZE = 7.5

TICK_LABEL_SIZE = 7.0

ANNOTATION_SIZE = 7.0

LEGEND_SIZE = 7.0

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

GROUP_LABELS = ('SH', 'NH (> 20°)', 'NH (≤ 20°)')

DISTANCE_LABELS = ('−300–−100', '±100', '100–300')

DURATION_METRICS = ('CLTC duration', 'Coastal-TC duration', 'Duration fraction')

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
    counts = table['Panel'].value_counts(dropna=False).to_dict()
    if len(table) != 18 or counts != {'A': 3, 'B': 9, 'C': 3, 'D': 3}:
        raise ValueError('S1 requires exactly 18 rows: A=3, B=9, C=3, D=3')

    keys = ['Panel', 'X_Label', 'Metric']
    expected = (
        [('A', label, 'CLTC count') for label in GROUP_LABELS]
        + [('B', label, metric) for label in GROUP_LABELS for metric in DURATION_METRICS]
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
        'CLTC duration': 'h decade^-1',
        'Coastal-TC duration': 'h decade^-1',
        'Duration fraction': 'percentage points decade^-1',
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


def load_plot_data(data_file):
    if not data_file.is_file():
        raise FileNotFoundError(f'Missing plotting data: {data_file}. Run the S1 preprocessing script first.')
    table = validate_plot_data(pd.read_csv(data_file, encoding='utf-8-sig', float_precision='round_trip'))
    count_results = {}
    for panel in ('A', 'C'):
        count_results[panel] = [
            {'Trend_events_decade': row.Trend_per_Decade,
             'CI95_events_decade': row.CI95_per_Decade,
             'MK_P': row.MK_P}
            for row in table.loc[table.Panel.eq(panel)].itertuples(index=False)
        ]
    duration_results = []
    for metric in DURATION_METRICS:
        rows = table.loc[table.Panel.eq('B') & table.Metric.eq(metric)]
        duration_results.append([
            {'PLOTTED_TREND_PER_DECADE': row.Trend_per_Decade,
             'PLOTTED_CI95_PER_DECADE': row.CI95_per_Decade,
             'MK_P': row.MK_P}
            for row in rows.itertuples(index=False)
        ])
    exposure = table.loc[table.Panel.eq('D')].rename(columns={
        'X_Label': 'label', 'Mean_point_exposure_persons': 'mean',
        'CI95_low': 'ci_low', 'CI95_high': 'ci_high'}).copy()
    exposure['distance_key'] = exposure.label.map({label: key for key, label in EXPOSURE_DISTANCE_BINS})
    return count_results, duration_results, exposure


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
    axis.axhline(0, color='#222222', linewidth=.55, zorder=0)
    axis.set_xticks(positions, labels)
    axis.set_xlim(positions[0]-.65, positions[-1]+.65)
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

def plot_duration_fraction_panel(
    ax, region_labels, numerator_results, denominator_results, fraction_results
):
    """Three grouped bars per region with aligned duration and fraction axes."""
    positions = np.arange(len(region_labels), dtype=float)
    width = 0.22
    duration_groups = (
        (numerator_results, positions - width, DURATION_COLORS[0], "CLTC duration"),
        (denominator_results, positions, DURATION_COLORS[1], "Coastal-TC duration"),
    )
    duration_values = []
    duration_errors = []
    handles = []
    for results, x_positions, color, label in duration_groups:
        values = np.asarray(
            [item["PLOTTED_TREND_PER_DECADE"] for item in results], dtype=float
        )
        errors = np.asarray(
            [item["PLOTTED_CI95_PER_DECADE"] for item in results], dtype=float
        )
        duration_values.extend(values)
        duration_errors.extend(errors)
        handles.append(ax.bar(
            x_positions, values, yerr=errors, width=width,
            color=color, edgecolor="none", label=label, zorder=2,
            error_kw={
                "ecolor": "#222222", "elinewidth": 0.7,
                "capthick": 0.7, "capsize": 2.0,
            },
        ))
    duration_values = np.asarray(duration_values, dtype=float)
    duration_errors = np.asarray(duration_errors, dtype=float)
    duration_low = float(np.nanmin(np.minimum(
        duration_values - duration_errors, 0.0
    )))
    duration_high = float(np.nanmax(np.maximum(
        duration_values + duration_errors, 0.0
    )))
    fraction_ax = ax.twinx()
    fraction_values = np.asarray(
        [item["PLOTTED_TREND_PER_DECADE"] for item in fraction_results], dtype=float
    )
    fraction_errors = np.asarray(
        [item["PLOTTED_CI95_PER_DECADE"] for item in fraction_results], dtype=float
    )
    fraction_bars = fraction_ax.bar(
        positions + width, fraction_values, yerr=fraction_errors, width=width,
        color=FRACTION_BAR_COLOR, edgecolor="none", linewidth=0,
        label="Duration fraction", zorder=3,
        error_kw={
            "ecolor": FRACTION_ACCENT_COLOR, "elinewidth": 0.75,
            "capthick": 0.75, "capsize": 2.0,
        },
    )
    fraction_low = float(np.nanmin(np.minimum(fraction_values - fraction_errors, 0.0)))
    fraction_high = float(np.nanmax(np.maximum(fraction_values + fraction_errors, 0.0)))
    # Use five ticks on each axis at identical vertical positions. Zero is
    # the second tick on both axes; the two numeric scales retain their units.
    def nice_step(required):
        required = max(float(required), 1.0e-12)
        exponent = math.floor(math.log10(required))
        magnitude = 10.0 ** exponent
        fraction = required / magnitude
        for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
            if fraction <= candidate:
                return candidate * magnitude
        return 10.0 * magnitude
    duration_step = nice_step(max(
        max(0.0, -duration_low) * 1.05,
        max(0.0, duration_high) * 1.15 / 3.0,
    ))
    fraction_step = nice_step(max(
        max(0.0, -fraction_low) * 1.05,
        max(0.0, fraction_high) * 1.15 / 3.0,
    ))
    duration_ticks = duration_step * np.arange(-1.0, 4.0)
    fraction_ticks = fraction_step * np.arange(-1.0, 4.0)
    duration_limits = (float(duration_ticks[0]), float(duration_ticks[-1]))
    fraction_limits = (float(fraction_ticks[0]), float(fraction_ticks[-1]))
    duration_scale = 4.0 * duration_step
    fraction_scale = 4.0 * fraction_step
    ax.set_ylim(*duration_limits)
    fraction_ax.set_ylim(*fraction_limits)
    ax.set_yticks(duration_ticks)
    fraction_ax.set_yticks(fraction_ticks)
    ax.axhline(0.0, color="#222222", linewidth=0.7, zorder=1)
    ax.set_ylabel(r"Trend in duration (h decade$^{-1}$)")
    fraction_ax.set_ylabel(
        r"Trend in duration fraction (% decade$^{-1}$)",
        color=FRACTION_ACCENT_COLOR, labelpad=1.5,
    )
    ax.set_xticks(positions, region_labels)
    ax.set_xlim(-0.822, len(region_labels) - 0.178)
    for results, x_positions, _, _ in duration_groups:
        for x_position, result in zip(x_positions, results):
            mark = significance_mark(result["MK_P"])
            if mark:
                ax.text(
                    x_position,
                    result["PLOTTED_TREND_PER_DECADE"]
                    + result["PLOTTED_CI95_PER_DECADE"]
                    + 0.035 * duration_scale,
                    mark, ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                    fontweight="bold", clip_on=False,
                )
    for x_position, result in zip(positions + width, fraction_results):
        mark = significance_mark(result["MK_P"])
        if mark:
            fraction_ax.text(
                x_position,
                result["PLOTTED_TREND_PER_DECADE"]
                + result["PLOTTED_CI95_PER_DECADE"]
                + 0.035 * fraction_scale,
                mark, color=FRACTION_ACCENT_COLOR, ha="center", va="bottom",
                fontsize=ANNOTATION_SIZE, fontweight="bold", clip_on=False,
            )
    ax.spines[["top", "right"]].set_visible(False)
    fraction_ax.spines[["top", "left", "bottom"]].set_visible(False)
    fraction_ax.spines["right"].set_color(FRACTION_ACCENT_COLOR)
    fraction_ax.spines["right"].set_linewidth(0.7)
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    fraction_ax.tick_params(
        axis="y", direction="out", length=3, width=0.7, pad=2,
        colors=FRACTION_ACCENT_COLOR,
    )
    ax.legend(
        [handles[0], handles[1], fraction_bars],
        ["CLTC duration", "Coastal-TC duration", "Duration fraction"],
        frameon=False, loc="upper right", bbox_to_anchor=(0.985, 0.985),
        ncol=1, fontsize=LEGEND_SIZE, labelspacing=0.30,
        handletextpad=0.40, borderaxespad=0.0,
    )
    return fraction_ax


def draw_panel_d(axis, table):
    means = table["mean"].to_numpy(float) / 1_000_000.0
    lower = (table["mean"] - table["ci_low"]).to_numpy(float) / 1_000_000.0
    upper = (table["ci_high"] - table["mean"]).to_numpy(float) / 1_000_000.0
    x = np.arange(len(table))
    axis.bar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        width=0.62,
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
    axis.set_xlim(-0.902, 2.902)
    axis.margins(y=0.12)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7, pad=2)


def make_composite(count_results, duration_results, exposure):
    # Keep each original panel's physical dimensions. The old PNG composition
    # shrank panels to ~91%; applying that shrink to axes alone would crowd
    # unchanged native text and the dual-axis panel's legend.
    old_tile_scale = min((.995-.015)/(2+.035)*FIGSIZE_COMPOSITE[0]/3.4,
                         (.95-.015)/(2+.055)*FIGSIZE_COMPOSITE[1]/2.55)
    figure = plt.figure(figsize=tuple(size/old_tile_scale for size in FIGSIZE_COMPOSITE),
                         facecolor='white')
    grid = figure.add_gridspec(2, 2, left=.015, right=.995, bottom=.015,
                              top=.95, wspace=.035, hspace=.055)
    axes, tiles = {}, {}
    for index, letter in enumerate('ABCD'):
        slot = grid[index//2, index%2].get_position(figure)
        # Match the former 3.40 x 2.55 panel slots without embedding raster images.
        scale = min(slot.width*figure.get_figwidth()/3.4,
                    slot.height*figure.get_figheight()/2.55)
        width, height = 3.4*scale/figure.get_figwidth(), 2.55*scale/figure.get_figheight()
        left, bottom = slot.x0+(slot.width-width)/2, slot.y0+(slot.height-height)/2
        tiles[letter] = (left, bottom, width, height)
        if letter in ('A', 'C'):
            l, r, b, t = .27, .924, .22, .975
        else:
            l, r, b, t = .126, .880, .22, .940
        axes[letter] = figure.add_axes([left+l*width, bottom+b*height,
                                        (r-l)*width, (t-b)*height])
    limits, ticks = shared_event_axis(count_results['A'], count_results['C'])
    draw_count_trends(axes['A'], count_results['A'], GROUP_LABELS, limits, ticks)
    draw_count_trends(axes['C'], count_results['C'], DISTANCE_LABELS, limits, ticks,
                      xlabel='Distance to coast (km)')
    fraction_axis = plot_duration_fraction_panel(axes['B'], GROUP_LABELS, *duration_results)
    draw_panel_d(axes['D'], exposure)
    for letter, (left, bottom, width, height) in tiles.items():
        offset = .15 if letter in ('A', 'C') else .006
        figure.text(left+offset*width, bottom+1.03*height, letter,
                    ha='left', va='top', fontsize=PANEL_LABEL_SIZE/old_tile_scale,
                    fontweight='bold', color='#111111', gid='panel-label')
    groups = {letter: [axis] for letter, axis in axes.items()}
    groups['B'].append(fraction_axis)
    return figure, groups


def save_panel(figure, selected_axes, path):
    """Keep the selected panel's physical scale, including B's secondary axis."""
    visibility = [(item, item.get_visible()) for item in [*figure.axes, *figure.texts]]
    try:
        for item, _ in visibility:
            item.set_visible(item in selected_axes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = Bbox.union([axis.get_tightbbox(renderer) for axis in selected_axes])
        bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(.035)
        figure.savefig(path, dpi=DPI, facecolor='white', bbox_inches=bounds, pad_inches=0)
    finally:
        for item, visible in visibility:
            item.set_visible(visible)


def save_figures(count_results, duration_results, exposure, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, groups = make_composite(count_results, duration_results, exposure)
    filenames = {
        'A': 'FigS1a_latitude_region_trends.png',
        'B': 'FigS1b_duration_and_fraction.png',
        'C': 'FigS1c_distance_band_trends_NH20.png',
        'D': 'FigS1d_mean_point_exposure_NH20.png',
    }
    try:
        figure.savefig(output_dir/'FigS1_composite.png', dpi=DPI, facecolor='white')
        for letter, group in groups.items():
            save_panel(figure, group, output_dir/filenames[letter])
    finally:
        plt.close(figure)
    print(f'Saved four panels and one composite PNG: {output_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    print(f'Plotting data: {data_file}')
    save_figures(*load_plot_data(data_file), output_dir=output_dir)


if __name__ == '__main__':
    main()
