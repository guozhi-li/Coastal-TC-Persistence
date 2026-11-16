#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR/'Database/CLTC'
DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Results/001_S2_CLTC_SENSITIVITY'


DPI = 600

BAR_PALETTE = ["#9DB7D5", "#5F83AD", "#244A73"]

LAND_MASK_PALETTE = ["#D7E3EF", "#9DB7D5", "#5F83AD", "#244A73"]

COMPACT_FORMATTER = FuncFormatter(lambda value, position: f"{value:g}")

DURATION_THRESHOLDS = (12, 24, 36)

DISTANCE_THRESHOLDS = (50, 100, 150)

LAND_AREAS = (1400, 2000, 5000, 10000)

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
})

BAR_COLUMNS = (
    'Panel', 'Condition', 'Method', 'Trend_events_decade',
    'CI95_events_decade', 'MK_P',
)

PDF_COLUMNS = ('Duration_h', 'Density_percent')

SHARE_COLUMNS = ('Threshold_h', 'Below_percent')

BAR_CASES = (
    ('A', '12 h', 'LMI'),
    ('A', '24 h', 'LMI'),
    ('A', '36 h', 'LMI'),
    ('B', '±50 km', 'LMI'),
    ('B', '±100 km', 'LMI'),
    ('B', '±150 km', 'LMI'),
    ('C', '1400', 'LMI'),
    ('C', '2000', 'LMI'),
    ('C', '5000', 'LMI'),
    ('C', '10000', 'LMI'),
    ('D', 'IBTrACS (3 h)', 'LMI'),
    ('D', 'IBTrACS (3 h)', 'MIN'),
    ('D', 'IBTrACS (6 h)', 'LMI'),
    ('D', 'IBTrACS (6 h)', 'MIN'),
    ('D', 'ADT-HURSAT (6 h)', 'LMI'),
    ('D', 'ADT-HURSAT (6 h)', 'MIN'),
    ('E', 'Original', 'LMI'),
    ('E', 'ENSO-removed', 'LMI'),
    ('E', 'AMO-removed', 'LMI'),
)

PDF_POINT_COUNT = 600

PDF_MARKERS = (12, 24, 36)

def validate_bars(table):
    """Return the 19 original bar estimates in panel/category/method order."""
    if table.columns.duplicated().any() or set(table.columns) != set(BAR_COLUMNS):
        raise ValueError(f'S2 bars require exactly these columns: {BAR_COLUMNS}')
    table = table.loc[:, list(BAR_COLUMNS)].copy()
    keys = ['Panel', 'Condition', 'Method']
    if len(table) != len(BAR_CASES) or table[keys].isna().any().any():
        raise ValueError('S2 requires 19 complete bar records: A=3, B=3, C=4, D=6, E=3')
    if table.duplicated(keys).any():
        raise ValueError('S2 bar panel/condition/method combinations must be unique')
    observed = set(table[keys].itertuples(index=False, name=None))
    if observed != set(BAR_CASES):
        raise ValueError('Unexpected or missing S2 bar panel/condition/method combinations')
    numeric = ['Trend_events_decade', 'CI95_events_decade', 'MK_P']
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError('S2 bar trends, confidence half-widths and MK P values must be finite')
    if table['CI95_events_decade'].lt(0).any():
        raise ValueError('S2 confidence half-widths must be nonnegative')
    if not table['MK_P'].between(0, 1).all():
        raise ValueError('S2 MK P values must lie within [0, 1]')
    order = pd.MultiIndex.from_tuples(BAR_CASES, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(BAR_COLUMNS)]

def validate_pdf(table):
    """Validate and order the saved curve without fitting or renormalizing it."""
    if table.columns.duplicated().any() or set(table.columns) != set(PDF_COLUMNS):
        raise ValueError(f'S2 PDF requires exactly these columns: {PDF_COLUMNS}')
    table = table.loc[:, list(PDF_COLUMNS)].copy()
    if len(table) != PDF_POINT_COUNT:
        raise ValueError(f'S2 PDF requires exactly {PDF_POINT_COUNT} saved curve points')
    for column in PDF_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('S2 PDF durations and densities must be finite')
    table = table.sort_values('Duration_h', kind='stable').reset_index(drop=True)
    x = table['Duration_h'].to_numpy(dtype=float)
    density = table['Density_percent'].to_numpy(dtype=float)
    if x[0] != 0 or not (np.diff(x) > 0).all() or x[-1] < 48:
        raise ValueError('S2 PDF must start at zero, increase strictly and extend to at least 48 h')
    if not np.allclose(np.diff(x), (x[-1] - x[0]) / (len(x) - 1), rtol=1e-10, atol=1e-12):
        raise ValueError('S2 PDF must retain its original evenly spaced grid')
    if (density < 0).any() or not (density > 0).any():
        raise ValueError('S2 PDF density must be nonnegative and not identically zero')
    return table

def validate_shares(table):
    """Validate the empirical percentages; never derive them from the PDF."""
    if table.columns.duplicated().any() or set(table.columns) != set(SHARE_COLUMNS):
        raise ValueError(f'S2 duration shares require exactly these columns: {SHARE_COLUMNS}')
    table = table.loc[:, list(SHARE_COLUMNS)].copy()
    if len(table) != len(PDF_MARKERS):
        raise ValueError('S2 requires the three empirical shares below 12, 24 and 36 h')
    for column in SHARE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('S2 empirical thresholds and percentages must be finite')
    if set(table['Threshold_h']) != set(PDF_MARKERS):
        raise ValueError('S2 empirical thresholds must be exactly 12, 24 and 36 h')
    table = table.sort_values('Threshold_h', kind='stable').reset_index(drop=True)
    shares = table['Below_percent'].to_numpy(dtype=float)
    if not ((shares >= 0) & (shares <= 100)).all() or (np.diff(shares) < 0).any():
        raise ValueError('S2 empirical shares must increase monotonically within [0, 100]')
    return table


def load_plot_data(data_dir):
    tables = []
    for filename, validate in (
        ('FigS2_bar_statistics.csv', validate_bars),
        ('FigS2_duration_pdf.csv', validate_pdf),
        ('FigS2_duration_shares.csv', validate_shares),
    ):
        path = data_dir/filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing plotting data: {path}. Run the S2 preprocessing script first.')
        tables.append(validate(pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')))
    bars, curve, shares = tables
    pdf = {'x': curve.Duration_h.to_numpy(float),
           'density_percent': curve.Density_percent.to_numpy(float),
           'xmax': float(curve.Duration_h.max()),
           'below': dict(zip(shares.Threshold_h.astype(int), shares.Below_percent.astype(float)))}
    return bars, pdf


def significance_mark(p_value):
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""

def _bar_limits(slopes, errors):
    upper = np.asarray(slopes, dtype=float) + np.asarray(errors, dtype=float)
    lower = np.asarray(slopes, dtype=float) - np.asarray(errors, dtype=float)
    data_max = float(np.nanmax(upper))
    data_min = float(np.nanmin(lower))
    span = max(data_max - min(0.0, data_min), 0.5)
    return data_max, span

def common_bc_axis(summary):
    """Return one y range and one tick sequence shared by panels b and c."""
    rows = summary[summary["Panel"].isin(["B", "C"])]
    slopes = rows["Trend_events_decade"].to_numpy(dtype=float)
    errors = rows["CI95_events_decade"].to_numpy(dtype=float)
    data_max, span = _bar_limits(slopes, errors)
    requested_top = data_max + 0.15 * span
    ticks = MaxNLocator(nbins=5).tick_values(0.0, requested_top)
    ticks = ticks[ticks >= 0.0]
    return ticks, float(ticks[-1])

def style_axis(ax):
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    ax.spines[["top", "right"]].set_visible(False)

def _physical_bar_width(ax, data_span, target_inches=0.38, maximum=0.70):
    """Convert a target printed width to data coordinates for consistent bars."""
    axes_width_inches = ax.get_position().width * ax.figure.get_figwidth()
    if axes_width_inches <= 0:
        return maximum
    return min(target_inches * data_span / axes_width_inches, maximum)

def draw_simple_bars(
    ax,
    table,
    conditions,
    xlabel,
    colors=BAR_PALETTE,
    tick_labels=None,
):
    rows = table.set_index("Condition").loc[list(conditions)]
    slopes = rows["Trend_events_decade"].to_numpy(dtype=float)
    errors = rows["CI95_events_decade"].to_numpy(dtype=float)
    mk_values = rows["MK_P"].to_numpy(dtype=float)
    x = np.arange(len(conditions), dtype=float)
    palette = list(colors)[:len(conditions)]
    x_left = -0.55
    x_right = len(conditions) - 0.45
    data_span = x_right - x_left
    bar_width = _physical_bar_width(ax, data_span)
    ax.bar(
        x, slopes, yerr=errors, width=bar_width,
        color=palette, edgecolor="none",
        error_kw={
            "ecolor": "#222222", "elinewidth": 0.65,
            "capthick": 0.65, "capsize": 2.0,
        },
    )
    data_max, span = _bar_limits(slopes, errors)
    for position, slope, error, p_value in zip(
        x, slopes, errors, mk_values
    ):
        mark = significance_mark(float(p_value))
        if mark:
            ax.text(
                position, slope + error + 0.025 * span, mark,
                ha="center", va="bottom", fontsize=7.2,
                fontweight="bold", clip_on=False,
            )
    ax.set_ylim(0.0, data_max + 0.15 * span)
    ax.set_xlim(x_left, x_right)
    if tick_labels is None:
        tick_labels = conditions
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Trend in CLTC count (decade$^{-1}$)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax.axhline(0, color="#222222", linewidth=0.55, zorder=0)
    style_axis(ax)

def draw_pdf(ax, pdf):
    """Compact PDF used as the upper-right inset of panel a."""
    x = pdf["x"]
    density = pdf["density_percent"]
    ax.set_facecolor("white")
    ax.patch.set_alpha(0.96)
    ax.plot(x, density, color="#20354F", linewidth=0.85)
    ax.fill_between(
        x, 0.0, density, color="#AFC1D5", alpha=0.34, linewidth=0
    )
    marker_colors = ["#9DB7D5", "#5F83AD", "#244A73"]
    for threshold, color in zip(PDF_MARKERS, marker_colors):
        ax.axvline(threshold, color=color, linestyle="--", linewidth=0.65)
    ax.set_xlim(0.0, pdf["xmax"])
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MultipleLocator(24))
    pdf_ylim = ax.get_ylim()
    ax.set_yticks([1, 3])
    ax.set_ylim(pdf_ylim)
    ax.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax.set_xlabel("Duration (h)", fontsize=4.7, labelpad=1.0)
    ax.set_ylabel("Probability\ndensity (%)", fontsize=4.7, labelpad=1.0)
    ax.text(
        0.97, 0.94,
        f"<12 h: {pdf['below'][12]:.1f}%\n"
        f"<24 h: {pdf['below'][24]:.1f}%\n"
        f"<36 h: {pdf['below'][36]:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=4.7, linespacing=1.12,
        bbox={
            "facecolor": "white", "edgecolor": "none",
            "alpha": 0.78, "pad": 0.5,
        },
    )
    ax.tick_params(
        direction="out", length=1.3, width=0.48,
        pad=0.7, labelsize=4.7,
    )
    ax.spines[["top", "right"]].set_visible(False)

def draw_dataset_method_bars(ax, table):
    dataset_order = [
        "IBTrACS (3 h)",
        "IBTrACS (6 h)",
        "ADT-HURSAT (6 h)",
    ]
    method_order = ["LMI", "MIN"]
    method_colors = {"LMI": "#9DB7D5", "MIN": "#244A73"}
    x = np.arange(len(dataset_order), dtype=float)
    x_left, x_right = -0.58, len(dataset_order) - 0.42
    data_span = x_right - x_left
    width = _physical_bar_width(ax, data_span)
    all_slopes, all_errors = [], []
    for method_index, method in enumerate(method_order):
        rows = (
            table[table["Method"].eq(method)]
            .set_index("Condition")
            .loc[dataset_order]
        )
        slopes = rows["Trend_events_decade"].to_numpy(dtype=float)
        errors = rows["CI95_events_decade"].to_numpy(dtype=float)
        mk_values = rows["MK_P"].to_numpy(dtype=float)
        positions = x + (method_index - 0.5) * width
        ax.bar(
            positions, slopes, yerr=errors, width=width,
            color=method_colors[method], edgecolor="none", label=f"{method} ≥ 34 kt",
            error_kw={
                "ecolor": "#222222", "elinewidth": 0.65,
                "capthick": 0.65, "capsize": 2.0,
            },
        )
        all_slopes.extend(slopes.tolist())
        all_errors.extend(errors.tolist())
        _, local_span = _bar_limits(slopes, errors)
        for position, slope, error, p_value in zip(
            positions, slopes, errors, mk_values
        ):
            mark = significance_mark(float(p_value))
            if mark:
                ax.text(
                    position, slope + error + 0.025 * local_span, mark,
                    ha="center", va="bottom", fontsize=7.2,
                    fontweight="bold", clip_on=False,
                )
    data_max, span = _bar_limits(all_slopes, all_errors)
    ax.set_ylim(0.0, data_max + 0.15 * span)
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(x, dataset_order)
    ax.set_xlabel("Dataset and temporal resolution")
    ax.set_ylabel(r"Trend in CLTC count (decade$^{-1}$)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax.axhline(0, color="#222222", linewidth=0.55, zorder=0)
    ax.legend(
        loc="upper right", ncol=1, frameon=False,
        handlelength=1.1, columnspacing=0.8,
        handletextpad=0.35, borderaxespad=0.25,
    )
    style_axis(ax)

def draw_climate_mode_bars(ax, table):
    """Plot Original, ENSO-removed, and AMO-removed CLTC-count trends."""
    conditions = ("Original", "ENSO-removed", "AMO-removed")
    rows = table.set_index("Condition").loc[list(conditions)]
    slopes = rows["Trend_events_decade"].to_numpy(dtype=float)
    errors = rows["CI95_events_decade"].to_numpy(dtype=float)
    mk_values = rows["MK_P"].to_numpy(dtype=float)
    positions = np.arange(len(conditions), dtype=float)
    colors = ["#9DB7D5", "#5F83AD", "#244A73"]
    x_left, x_right = -0.55, len(conditions) - 0.45
    width = _physical_bar_width(ax, x_right - x_left)
    ax.bar(
        positions, slopes, yerr=errors, width=width,
        color=colors, edgecolor="none",
        error_kw={
            "ecolor": "#222222", "elinewidth": 0.65,
            "capthick": 0.65, "capsize": 2.0,
        },
    )
    upper = slopes + errors
    lower = slopes - errors
    data_high = float(np.nanmax(np.maximum(upper, 0.0)))
    data_low = float(np.nanmin(np.minimum(lower, 0.0)))
    span = max(data_high - data_low, 0.5)
    ax.set_ylim(0.0, data_high + 0.18 * span)
    for position, slope, error, p_value in zip(
        positions, slopes, errors, mk_values
    ):
        mark = significance_mark(float(p_value))
        if mark:
            ax.text(
                position, slope + error + 0.035 * span, mark,
                ha="center", va="bottom", fontsize=7.2,
                fontweight="bold", clip_on=False,
            )
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(
        positions,
        ["Original", "ENSO-removed", "AMO-removed"],
    )
    ax.set_ylabel(r"Trend in CLTC count (decade$^{-1}$)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(COMPACT_FORMATTER)
    ax.axhline(0.0, color="#222222", linewidth=0.55, zorder=0)
    style_axis(ax)

def add_panel_label(ax, label):
    """Place a panel letter outside the upper-left corner without clipping."""
    box = ax.get_position()
    ax.figure.text(
        max(0.008, box.x0 - 0.045),
        min(0.965, box.y1 + 0.012),
        label.lower(),
        ha="left", va="bottom",
        fontsize=10.5, fontweight="bold", color="#111111",
        zorder=20,
    )

def add_pdf_inset(ax, pdf, bounds=None):
    if bounds is None:
        bounds = [0.61, 0.74, 0.35, 0.29]
    inset = ax.inset_axes(
        bounds,
        facecolor="white", zorder=8,
    )
    draw_pdf(inset, pdf)
    return inset

def make_composite(summary, pdf):
    """Compose A-C above, with wide D and compact E below."""
    fig = plt.figure(figsize=(9.10, 4.65), facecolor="white")
    grid = fig.add_gridspec(
        2, 3,
        left=0.060, right=0.992, bottom=0.105, top=0.930,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.32, hspace=0.38,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0:2])
    ax_e = fig.add_subplot(grid[1, 2])
    draw_simple_bars(
        ax_a, summary[summary["Panel"].eq("A")],
        [f"{value} h" for value in DURATION_THRESHOLDS],
        "Minimum duration (h)",
        tick_labels=[str(value) for value in DURATION_THRESHOLDS],
    )
    inset = add_pdf_inset(ax_a, pdf)
    draw_simple_bars(
        ax_b, summary[summary["Panel"].eq("B")],
        [f"±{value} km" for value in DISTANCE_THRESHOLDS],
        "Maximum distance to coast (km)",
        tick_labels=[f"±{value}" for value in DISTANCE_THRESHOLDS],
    )
    draw_simple_bars(
        ax_c, summary[summary["Panel"].eq("C")],
        [str(value) for value in LAND_AREAS],
        r"Minimum landmass size (km$^2$)",
        colors=LAND_MASK_PALETTE,
        tick_labels=[str(value) for value in LAND_AREAS],
    )
    bc_ticks, bc_top = common_bc_axis(summary)
    for axis in (ax_b, ax_c):
        axis.set_ylim(0.0, bc_top)
        axis.set_yticks(bc_ticks)
    draw_dataset_method_bars(ax_d, summary[summary["Panel"].eq("D")])
    draw_climate_mode_bars(ax_e, summary[summary["Panel"].eq("E")])
    for axis, label in (
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_e, "E"),
    ):
        add_panel_label(axis, label)
    groups = {'A': [ax_a, inset], 'B': [ax_b], 'C': [ax_c],
              'D': [ax_d], 'E': [ax_e]}
    return fig, groups



def save_panel(figure, selected_axes, path):
    """Include the child PDF axes and its annotations when exporting A."""
    # inset_axes creates a child that need not occur in figure.axes.
    artists = list(dict.fromkeys([*figure.axes, *figure.texts, *selected_axes]))
    visibility = [(artist, artist.get_visible()) for artist in artists]
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


def save_figures(summary, pdf, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, groups = make_composite(summary, pdf)
    filenames = {
        'A': 'FigS2a_duration_threshold_and_PDF.png',
        'B': 'FigS2b_distance_sensitivity.png',
        'C': 'FigS2c_island_area_sensitivity.png',
        'D': 'FigS2d_dataset_resolution_methods.png',
        'E': 'FigS2e_ENSO_AMO_removed.png',
    }
    try:
        figure.savefig(output_dir/'FigS2_composite.png', dpi=DPI, facecolor='white')
        for panel, axes in groups.items():
            save_panel(figure, axes, output_dir/filenames[panel])
    finally:
        plt.close(figure)
    print(f'Saved five panels and one composite PNG: {output_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_dir, output_dir = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    print(f'Plotting data: {data_dir}')
    save_figures(*load_plot_data(data_dir), output_dir=output_dir)


if __name__ == '__main__':
    main()
