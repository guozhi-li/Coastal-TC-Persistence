#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR/'Database/CLTC'
DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Results/001_M1_CAL_CLTC'

DPI = 600
PANEL_FIGSIZE = (2.65, 2.12)
PANEL_MARGINS = dict(left=0.20, right=0.80, bottom=0.24, top=0.96)
COMPOSITE_AXES_GAP = 0.60
DURATION_COLORS = ('#496181', '#496181')
FRACTION_BAR_COLOR = '#C25D34'
FRACTION_ACCENT_COLOR = '#C25D34'

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


MAIN_REGIONS = ('Global', 'NH20')
ALL_REGIONS = MAIN_REGIONS
SERIES_COLUMNS = ('Region', 'Year', 'Count', 'Fitted', 'CI95_half_width')
STAT_COLUMNS = ('Region', 'Trend_events_decade', 'CI95_events_decade', 'MK_P')
DURATION_METRICS = ('CLTC duration', 'Coastal-TC duration', 'Duration fraction')
DURATION_UNITS = ('h decade^-1', 'h decade^-1', 'percentage points decade^-1')
DURATION_COLUMNS = ('Metric', 'Trend_per_Decade', 'CI95_per_Decade', 'MK_P', 'Trend_Unit')


def validate_plot_tables(series, statistics):
    """Validate stored curves and genuine numerical Mann-Kendall P values."""
    if series.columns.duplicated().any() or set(series.columns) != set(SERIES_COLUMNS):
        raise ValueError(f'M1 time series require exactly {SERIES_COLUMNS}')
    if statistics.columns.duplicated().any() or set(statistics.columns) != set(STAT_COLUMNS):
        raise ValueError(f'M1 statistics require exactly {STAT_COLUMNS}')
    series = series[list(SERIES_COLUMNS)].copy()
    statistics = statistics[list(STAT_COLUMNS)].copy()
    for column in SERIES_COLUMNS[1:]:
        series[column] = pd.to_numeric(series[column], errors='raise')
    for column in STAT_COLUMNS[1:]:
        statistics[column] = pd.to_numeric(statistics[column], errors='raise')
    if not np.isfinite(series[list(SERIES_COLUMNS[1:])].to_numpy(float)).all():
        raise ValueError('M1 time series contain nonfinite values')
    if not np.isfinite(statistics[list(STAT_COLUMNS[1:])].to_numpy(float)).all():
        raise ValueError('M1 statistics contain nonfinite values')
    expected = {(region, year) for region in MAIN_REGIONS for year in range(1980, 2025)}
    keys = ['Region', 'Year']
    if len(series) != 90 or series.duplicated(keys).any() or set(series[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('M1 requires one record per year, 1980-2024, for each of Global/NH20')
    if len(statistics) != 2 or statistics.Region.duplicated().any() or set(statistics.Region) != set(ALL_REGIONS):
        raise ValueError('M1 requires two regional statistics for Global/NH20')
    if (series.Count.lt(0).any() or not np.equal(series.Count, np.floor(series.Count)).all()
            or series.CI95_half_width.lt(0).any() or statistics.CI95_events_decade.lt(0).any()):
        raise ValueError('Counts must be nonnegative integers and confidence half-widths nonnegative')
    if not statistics.MK_P.between(0, 1).all():
        raise ValueError('Mann-Kendall P values must lie within [0, 1]')
    ordered = pd.MultiIndex.from_product([MAIN_REGIONS, range(1980, 2025)], names=keys)
    series = series.set_index(keys).reindex(ordered).reset_index()
    statistics = statistics.set_index('Region').reindex(ALL_REGIONS).reset_index()
    return series, statistics

def validate_duration_statistics(table):
    """Require the three NH >20N duration trends and numerical MK P values."""
    if table.columns.duplicated().any() or set(table.columns) != set(DURATION_COLUMNS):
        raise ValueError(f'M1 duration statistics require exactly {DURATION_COLUMNS}')
    table = table[list(DURATION_COLUMNS)].copy()
    if len(table) != 3 or table.Metric.duplicated().any() or set(table.Metric) != set(DURATION_METRICS):
        raise ValueError('M1 requires three NH >20N duration statistics')
    numeric = ['Trend_per_Decade', 'CI95_per_Decade', 'MK_P']
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError('M1 duration statistics contain nonfinite values')
    if table.CI95_per_Decade.lt(0).any() or not table.MK_P.between(0, 1).all():
        raise ValueError('Duration confidence half-widths must be nonnegative and MK P values within [0, 1]')
    table = table.set_index('Metric').reindex(DURATION_METRICS).reset_index()
    if tuple(table.Trend_Unit) != DURATION_UNITS:
        raise ValueError('M1 duration units must be hours and percentage points per decade')
    return table


def load_plot_data(data_dir):
    paths = [data_dir/name for name in (
        'FigM1_annual_curves.csv', 'FigM1_trend_statistics.csv',
        'FigM1_duration_statistics.csv')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing plotting data: {path}. Run Calculate_codes/001_M1_CAL_CLTC.py, or download the updated plotting Database.')
    tables = [pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip') for path in paths]
    series, statistics = validate_plot_tables(*tables[:2])
    duration = validate_duration_statistics(tables[2])
    statistics = statistics.set_index('Region')
    annual, trends = {}, {}
    for region in MAIN_REGIONS:
        rows = series.loc[series.Region.eq(region)]
        estimate = statistics.loc[region]
        annual[region] = (rows.Year.to_numpy(int), rows.Count.to_numpy(float), int(rows.Count.sum()))
        trends[region] = (float(estimate.Trend_events_decade), rows.Fitted.to_numpy(float),
                          rows.CI95_half_width.to_numpy(float), float(estimate.CI95_events_decade),
                          float(estimate.MK_P))
    duration_results = [
        {'PLOTTED_TREND_PER_DECADE': float(row.Trend_per_Decade),
         'PLOTTED_CI95_PER_DECADE': float(row.CI95_per_Decade), 'MK_P': float(row.MK_P)}
        for row in duration.itertuples(index=False)
    ]
    return annual, trends, duration_results


def significance_mark(p_value):
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""

def style_spines(ax):
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.7)

def plot_fig1a(years, counts, n_events, trend, output_dir):
    OUTPUT_FILE = output_dir / "FigM1a_Global_CLTC_frequency.png"
    TREND_COLOR = "#496181"
    slope_decade, fitted, band, slope_ci, mk_p = trend

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    fig.subplots_adjust(**PANEL_MARGINS)
    ax.plot(years, counts, color=TREND_COLOR, lw=1.10)
    ax.plot(years, fitted, color=TREND_COLOR, lw=1.05, ls="--")
    ax.fill_between(years, fitted - band, fitted + band,
                    color=TREND_COLOR, alpha=0.14, linewidth=0)
    p_label = (r"$P<0.01$" if mk_p < 0.01 else rf"$P={mk_p:.2f}$")
    ax.text(0.975, 0.975,
            rf"Trend = {slope_decade:.1f} decade$^{{-1}}$" "\n" f"{p_label}",
            transform=ax.transAxes, ha="right", va="top")
    ax.text(0.98, 0.035, "Global", transform=ax.transAxes,
            ha="right", va="bottom")
    ax.set(xlim=(1978, 2026), ylim=(10, 50), xlabel="Year",
           ylabel="Count of CLTC")
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_xticks([1980, 2000, 2020])
    style_spines(ax)
    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Events: {n_events}; saved: {OUTPUT_FILE}")

def plot_fig1b(years, counts, n_events, trend, output_dir):
    OUTPUT_FILE = output_dir / "FigM1b_NH_above20_CLTC_frequency.png"
    TREND_COLOR = "#496181"
    slope_decade, fitted, band, slope_ci, mk_p = trend
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, facecolor="white")
    fig.subplots_adjust(**PANEL_MARGINS)
    ax.plot(years, counts, color=TREND_COLOR, lw=1.10)
    ax.plot(years, fitted, color=TREND_COLOR, lw=1.05, ls="--")
    ax.fill_between(years, fitted - band, fitted + band,
                    color=TREND_COLOR, alpha=0.14, linewidth=0)
    p_label = r"$P<0.01$" if mk_p < 0.01 else rf"$P={mk_p:.2f}$"
    ax.text(0.975, 0.975,
            rf"Trend = {slope_decade:.1f} decade$^{{-1}}$" "\n" f"{p_label}",
            transform=ax.transAxes, ha="right", va="top")
    ax.text(0.98, 0.035, "> 20°N", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8))
    ax.set(xlim=(1978, 2026), ylim=(0, 24), xlabel="Year", ylabel="Count of CLTC")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_xticks([1980, 2000, 2020])
    style_spines(ax)
    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"NH20 events: {n_events}; saved: {OUTPUT_FILE}")

def plot_fig1c(results, output_dir):
    """Two blue duration bars and an orange duration-fraction bar north of 20N."""
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, facecolor="white")
    fig.subplots_adjust(**PANEL_MARGINS)
    fraction_ax = ax.twinx()
    values = np.array([r["PLOTTED_TREND_PER_DECADE"] for r in results])
    errors = np.array([r["PLOTTED_CI95_PER_DECADE"] for r in results])
    ax.bar([0, 1], values[:2], yerr=errors[:2], width=0.45,
           color=DURATION_COLORS, edgecolor="none", alpha=1.0, zorder=2,
           error_kw={"ecolor": "#222222", "elinewidth": 0.7,
                     "capthick": 0.7, "capsize": 2.0})
    fraction_ax.bar([2], values[2:], yerr=errors[2:], width=0.45,
                    color=FRACTION_BAR_COLOR, edgecolor="none", alpha=1.0, zorder=3,
                    error_kw={"ecolor": "black", "elinewidth": 0.75,
                              "capthick": 0.75, "capsize": 2.0})

    def nice_step(required):
        magnitude = 10.0 ** math.floor(math.log10(max(float(required), 1.0e-12)))
        fraction = required / magnitude
        return next(x for x in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= x) * magnitude

    duration_step = nice_step(max(max(0.0, float(np.max(errors[:2] - values[:2]))) * 1.05,
                                  max(0.0, float(np.max(values[:2] + errors[:2]))) * 1.15 / 3.0))
    fraction_step = nice_step(max(max(0.0, float(errors[2] - values[2])) * 1.05,
                                  max(0.0, float(values[2] + errors[2])) * 1.15 / 3.0))
    for axis, step in ((ax, duration_step), (fraction_ax, fraction_step)):
        ticks = step * np.arange(0.0, 4.0)
        axis.set_ylim(ticks[0], ticks[-1])
        axis.set_yticks(ticks)
    ax.axhline(0, color="#222222", linewidth=0.7, zorder=1)
    ax.set_ylabel(r"Trend in duration (h decade$^{-1}$)", labelpad=2)
    fraction_ax.set_ylabel("Trend in duration fraction\n" r"(% decade$^{-1}$)",
                           color=FRACTION_ACCENT_COLOR, labelpad=2)
    ax.set_xticks([0, 1, 2], ["CLTC\nduration", "Coastal-TC\nduration", "Duration\nfraction"])
    ax.get_xticklabels()[2].set_color("black")
    ax.set_xlim(-0.65, 2.65)
    for i, result in enumerate(results):
        axis, step = (ax, duration_step) if i < 2 else (fraction_ax, fraction_step)
        mark = significance_mark(result["MK_P"])
        if mark:
            axis.text(i, values[i] + errors[i] + 0.035 * 3.0 * step, mark,
                      ha="center", va="bottom", fontsize=7, fontweight="bold",
                      color="black")
    style_spines(ax)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)
    fraction_ax.spines[["top", "left", "bottom"]].set_visible(False)
    fraction_ax.spines["right"].set_color(FRACTION_ACCENT_COLOR)
    fraction_ax.spines["right"].set_alpha(1.0)
    fraction_ax.spines["right"].set_linewidth(0.7)
    fraction_ax.tick_params(axis="y", direction="out", length=3, width=0.7,
                            pad=2, colors=FRACTION_ACCENT_COLOR)
    path = output_dir / "FigM1c_NH_above20_duration_and_fraction.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

def compose_fig1(output_dir):
    output_composite = output_dir / "FigM1_composite.png"
    panels = [
        ("a", output_dir / "FigM1a_Global_CLTC_frequency.png"),
        ("b", output_dir / "FigM1b_NH_above20_CLTC_frequency.png"),
        ("c", output_dir / "FigM1c_NH_above20_duration_and_fraction.png"),
    ]
    missing = [str(path) for _, path in panels if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Fig. 1 panels:\n" + "\n".join(missing))

    panel_width, panel_height = PANEL_FIGSIZE
    header_height = 0.22
    # Remove only unused right-side whitespace from A/B in the composite.
    # Saved individual panels keep identical dimensions, and no panel is rescaled.
    original_gap = panel_width * (1 - PANEL_MARGINS["right"] + PANEL_MARGINS["left"])
    trim_right = max(0.0, original_gap - COMPOSITE_AXES_GAP)
    available_blank = panel_width * (1 - PANEL_MARGINS["right"])
    if trim_right > available_blank - 0.04:
        raise ValueError("COMPOSITE_AXES_GAP is too small to preserve panel contents")
    prepared = []
    for index, (letter, path) in enumerate(panels):
        pixels = plt.imread(path)
        if index < len(panels) - 1:
            keep = round(pixels.shape[1] * (panel_width - trim_right) / panel_width)
            pixels = pixels[:, :keep]
        width = panel_height * pixels.shape[1] / pixels.shape[0]
        prepared.append((letter, pixels, width))
    figure_width = sum(width for _, _, width in prepared)
    figure_height = panel_height + header_height
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    x_offset = 0.0
    for letter, pixels, width in prepared:
        panel = fig.add_axes([
            x_offset / figure_width, 0,
            width / figure_width, panel_height / figure_height,
        ])
        panel.imshow(pixels, interpolation="none")
        panel.set_axis_off()
        frame_left = x_offset + PANEL_MARGINS["left"] * panel_width
        frame_top = PANEL_MARGINS["top"] * panel_height
        fig.text(
            (frame_left - 0.35) / figure_width,
            (frame_top + 0.07) / figure_height,
            letter, ha="left", va="bottom", fontsize=10,
            fontweight="bold", color="#111111",
        )
        x_offset += width
    fig.savefig(output_composite, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_composite}")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_dir, output_dir = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    annual, trends, duration_results = load_plot_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fig1a(*annual['Global'], trends['Global'], output_dir)
    plot_fig1b(*annual['NH20'], trends['NH20'], output_dir)
    plot_fig1c(duration_results, output_dir)
    compose_fig1(output_dir)

if __name__ == '__main__':
    main()
