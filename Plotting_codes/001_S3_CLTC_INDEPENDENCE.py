#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = PROJECT_DIR / 'Database/CLTC/FigS3_plot_data.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/001_S3_CLTC_INDEPENDENCE'
DPI = 600

COMPOSITE_FIGSIZE = (7.00, 2.55)

PANEL_LABEL_LEFT_PT = 34.0

PANEL_LABEL_UP_PT = 15.0

BAR_WIDTH = 0.44

COLORS_A = ("#9DB7D5", "#244A73")

TRANSLATION_COLORS = ("#9DB7D5", "#244A73")

STEER_AXIS_COLOR = "#C25D34"

TEXT_COLOR = "#222222"

YLIM_A = (-0.02, 0.02)

YTICKS_A = [-0.02, -0.01, 0.00, 0.01, 0.02]

YLIM_B_LEFT = (-0.60, 0.20)

YTICKS_B_LEFT = [-0.60, -0.40, -0.20, 0.00, 0.20]

YLIM_B_RIGHT = (-0.60, 0.20)

YTICKS_B_RIGHT = [-0.60, -0.40, -0.20, 0.00, 0.20]

LABELS_A = ["Coastal TC", "CLTC"]

LABELS_B = ["Coastal TC", "CLTC", "CLTC steering flow"]

YLABEL_A = r"Trend in coast-tracking index (decade$^{-1}$)"

YLABEL_B_LEFT = r"Trend in translation speed (m s$^{-1}$ decade$^{-1}$)"

YLABEL_B_RIGHT = r"Trend in steering flow speed (m s$^{-1}$ decade$^{-1}$)"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.unicode_minus": True,
})

COMPACT_FORMATTER = FuncFormatter(lambda value, pos: f"{value:g}".replace("-", "−"))

PLOT_COLUMNS = (
    'Panel', 'X_Label', 'Metric', 'Trend_per_Decade', 'CI95_per_Decade',
    'MK_P', 'Unit',
)
EXPECTED_ROWS = (
    ('A', 'Coastal TC', 'Coast-tracking index', 'decade^-1'),
    ('A', 'CLTC', 'Coast-tracking index', 'decade^-1'),
    ('B', 'Coastal TC', 'Translation speed', 'm s^-1 decade^-1'),
    ('B', 'CLTC', 'Translation speed', 'm s^-1 decade^-1'),
    ('B', 'Steering flow', 'Steering flow speed', 'm s^-1 decade^-1'),
)


def load_plot_data(data_file):
    """Validate the five plotted estimates, including full numerical MK P."""
    if not data_file.is_file():
        raise FileNotFoundError(
            f'Missing plotting data: {data_file}. Run '
            'Calculate_codes/001_S3_CLTC_INDEPENDENCE.py, or download the plotting Database.'
        )
    table = pd.read_csv(data_file, encoding='utf-8-sig', float_precision='round_trip')
    if set(table.columns) != set(PLOT_COLUMNS) or table.columns.duplicated().any():
        raise ValueError(f'S3 plotting data require exactly these columns: {PLOT_COLUMNS}')
    keys = ['Panel', 'X_Label', 'Metric', 'Unit']
    actual_rows = set(table[keys].itertuples(index=False, name=None))
    expected_rows = set(EXPECTED_ROWS)
    if (len(table) != len(EXPECTED_ROWS) or table.duplicated(keys).any()
            or actual_rows != expected_rows):
        missing = sorted(expected_rows - actual_rows, key=str)
        unexpected = sorted(actual_rows - expected_rows, key=str)
        duplicates = table.loc[table.duplicated(keys, keep=False), keys].to_dict('records')
        raise ValueError(
            'S3 requires exactly two coast-tracking and three speed records. '
            f'Rows: {len(table)}; missing: {missing}; unexpected: {unexpected}; '
            f'duplicates: {duplicates}'
        )
    for column in ('Trend_per_Decade', 'CI95_per_Decade', 'MK_P'):
        table[column] = pd.to_numeric(table[column], errors='raise')
        if not np.isfinite(table[column]).all():
            raise ValueError(f'{column} must contain finite numerical values')
    if table.CI95_per_Decade.lt(0).any():
        raise ValueError('Confidence half-widths must be nonnegative')
    if not table.MK_P.between(0, 1).all():
        raise ValueError('MK_P must be within [0, 1]')
    ordered = pd.MultiIndex.from_tuples(EXPECTED_ROWS, names=keys)
    table = table.set_index(keys).reindex(ordered).reset_index()
    results = [
        {'Trend': float(row.Trend_per_Decade), 'CI95': float(row.CI95_per_Decade),
         'MK_P': float(row.MK_P)}
        for row in table.itertuples(index=False)
    ]
    return results[:2], results[2:4], results[4]


def significance_mark(p):
    return "**" if p < 0.05 else "*" if p < 0.10 else ""

def add_significance(ax, x, result, ylim, color=TEXT_COLOR):
    mark = significance_mark(result["MK_P"])
    if not mark:
        return
    span = ylim[1] - ylim[0]
    y = result["Trend"] + result["CI95"] + 0.035 * span
    ax.text(
        x, y, mark,
        ha="center", va="bottom",
        fontsize=7.5, fontweight="bold",
        color=color, clip_on=False,
    )

def style_single_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)

def plot_panel_a(ax, results_a):
    x = np.arange(2, dtype=float)
    slopes = np.array([r["Trend"] for r in results_a])
    errors = np.array([r["CI95"] for r in results_a])
    ax.bar(
        x, slopes, yerr=errors, width=BAR_WIDTH,
        color=COLORS_A, edgecolor="none", zorder=2,
        error_kw={"ecolor": "black", "elinewidth": 0.7, "capthick": 0.7, "capsize": 2.2},
    )
    for xi, result in zip(x, results_a):
        add_significance(ax, xi, result, YLIM_A, TEXT_COLOR)
    ax.set(xlim=(-0.5, 1.5), ylim=YLIM_A, ylabel=YLABEL_A)
    ax.set_xticks(x, LABELS_A)
    ax.set_yticks(YTICKS_A)
    ax.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.7, zorder=1)
    style_single_axis(ax)

def plot_panel_b(ax_left, translation_results, steering_result):
    """Dual-Y-axis Fig. S3B."""
    x_all = np.array([0.0, 1.0, 2.0])
    # Left axis: Coastal TC and CLTC translation-speed trends
    x_left = x_all[:2]
    slopes_left = np.array([r["Trend"] for r in translation_results])
    errors_left = np.array([r["CI95"] for r in translation_results])
    ax_left.bar(
        x_left, slopes_left, yerr=errors_left, width=BAR_WIDTH,
        color=TRANSLATION_COLORS, edgecolor="none", zorder=3,
        error_kw={"ecolor": "black", "elinewidth": 0.7, "capthick": 0.7, "capsize": 2.2},
    )
    for xi, result in zip(x_left, translation_results):
        add_significance(ax_left, xi, result, YLIM_B_LEFT, TEXT_COLOR)
    ax_left.set_xlim(-0.5, 2.5)
    ax_left.set_ylim(YLIM_B_LEFT)
    ax_left.set_yticks(YTICKS_B_LEFT)
    ax_left.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax_left.set_ylabel(YLABEL_B_LEFT)
    ax_left.set_xticks(x_all, LABELS_B)
    ax_left.axhline(0, color=TEXT_COLOR, linewidth=0.7, zorder=1)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(axis="both", direction="out", length=3, width=0.7, pad=2)
    # Right axis: Steering-flow trend only
    ax_right = ax_left.twinx()
    ax_right.bar(
        [x_all[2]], [steering_result["Trend"]],
        yerr=[steering_result["CI95"]], width=BAR_WIDTH,
        color=STEER_AXIS_COLOR, edgecolor="none", zorder=4,
        error_kw={
            "ecolor": "black",
            "elinewidth": 0.7,
            "capthick": 0.7,
            "capsize": 2.2,
        },
    )
    add_significance(
        ax_right, x_all[2], steering_result,
        YLIM_B_RIGHT, STEER_AXIS_COLOR,
    )
    ax_right.set_ylim(YLIM_B_RIGHT)
    ax_right.set_yticks(YTICKS_B_RIGHT)
    ax_right.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax_right.set_ylabel(YLABEL_B_RIGHT, color=STEER_AXIS_COLOR)
    ax_right.tick_params(
        axis="y", direction="out", length=3, width=0.7, pad=2,
        colors=STEER_AXIS_COLOR,
    )
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color(STEER_AXIS_COLOR)
    ax_right.spines["right"].set_linewidth(0.7)
    for tick in ax_right.get_yticklabels():
        tick.set_color(STEER_AXIS_COLOR)
    return ax_right

def make_composite(results_a, translation_results, steering_result):
    fig = plt.figure(figsize=COMPOSITE_FIGSIZE, facecolor="white")
    grid = fig.add_gridspec(
        1, 2,
        left=0.075, right=0.920, bottom=0.17, top=0.885,
        wspace=0.34, width_ratios=[2, 3],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    plot_panel_a(ax_a, results_a)
    ax_b_right = plot_panel_b(ax_b, translation_results, steering_result)

    # Both letters are placed the SAME physical distance from their own axes
    # upper-left corners. Unlike axes fractions, points do not change with the
    # different widths of panels A and B. The top margin prevents clipping.
    fig_width_pt = fig.get_figwidth() * 72.0
    fig_height_pt = fig.get_figheight() * 72.0
    for ax, letter in zip((ax_a, ax_b), "AB"):
        box = ax.get_position()
        fig.text(
            box.x0 - PANEL_LABEL_LEFT_PT / fig_width_pt,
            box.y1 + PANEL_LABEL_UP_PT / fig_height_pt,
            letter.lower(),
            transform=fig.transFigure,
            ha="left", va="bottom", fontsize=10.5,
            fontweight="bold", color="#111111", clip_on=False,
        )
    return fig, {'A': [ax_a], 'B': [ax_b, ax_b_right]}


def save_panel(figure, selected_axes, path):
    """Export a panel without resizing it or dropping B's secondary axis."""
    visibility = [(artist, artist.get_visible())
                  for artist in [*figure.axes, *figure.texts]]
    try:
        for artist, _ in visibility:
            artist.set_visible(artist in selected_axes)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [axis.get_tightbbox(renderer) for axis in selected_axes]
        bounds = Bbox.union([box for box in boxes if box is not None])
        bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(.035)
        figure.savefig(path, dpi=DPI, facecolor='white', bbox_inches=bounds, pad_inches=0)
    finally:
        for artist, visible in visibility:
            artist.set_visible(visible)


def save_figures(results_a, translation_results, steering_result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, groups = make_composite(results_a, translation_results, steering_result)
    try:
        figure.savefig(output_dir / 'FigS3_composite.png', dpi=DPI, facecolor='white',
                       bbox_inches='tight', pad_inches=.05)
        save_panel(figure, groups['A'], output_dir / 'FigS3a_coast_tracking.png')
        save_panel(figure, groups['B'], output_dir / 'FigS3b_translation_and_steering_speed.png')
    finally:
        plt.close(figure)
    print(f'Saved two panels and one composite PNG: {output_dir}')


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=resolve_project_path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--output-dir', type=resolve_project_path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_file, output_dir = args.data_file.expanduser().resolve(), args.output_dir.expanduser().resolve()
    print(f'Plotting data: {data_file}')
    print(f'Results: {output_dir}')
    save_figures(*load_plot_data(data_file), output_dir=output_dir)


if __name__ == '__main__':
    main()
