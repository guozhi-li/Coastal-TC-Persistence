#!/usr/bin/env python3
import argparse
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

BASIN_SPECS = [("WNP20", "WNP"), ("EP20", "EP"), ("WNA20", "WNA")]

BASIN_COLORS = ["#496181", "#C25D34", "#96B8A7"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


MAIN_REGIONS = ('Global', 'NH', 'NH20')
ALL_REGIONS = (*MAIN_REGIONS, 'WNP20', 'EP20', 'WNA20')
SERIES_COLUMNS = ('Region', 'Year', 'Count', 'Fitted', 'CI95_half_width')
STAT_COLUMNS = ('Region', 'Trend_events_decade', 'CI95_events_decade', 'MK_P')


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
    if len(series) != 135 or series.duplicated(keys).any() or set(series[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('M1 requires one record per year, 1980-2024, for each of Global/NH/NH20')
    if len(statistics) != 6 or statistics.Region.duplicated().any() or set(statistics.Region) != set(ALL_REGIONS):
        raise ValueError('M1 requires six regional statistics, including the three inset basins')
    if (series.Count.lt(0).any() or not np.equal(series.Count, np.floor(series.Count)).all()
            or series.CI95_half_width.lt(0).any() or statistics.CI95_events_decade.lt(0).any()):
        raise ValueError('Counts must be nonnegative integers and confidence half-widths nonnegative')
    if not statistics.MK_P.between(0, 1).all():
        raise ValueError('Mann-Kendall P values must lie within [0, 1]')
    ordered = pd.MultiIndex.from_product([MAIN_REGIONS, range(1980, 2025)], names=keys)
    series = series.set_index(keys).reindex(ordered).reset_index()
    statistics = statistics.set_index('Region').reindex(ALL_REGIONS).reset_index()
    return series, statistics



def load_plot_data(data_dir):
    series_path, statistics_path = data_dir/'FigM1_annual_curves.csv', data_dir/'FigM1_trend_statistics.csv'
    for path in (series_path, statistics_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing plotting data: {path}. Run 001_M1_CAL_CLTC_PREPROCESS.py first.')
    series, statistics = validate_plot_tables(
        pd.read_csv(series_path, encoding='utf-8-sig', float_precision='round_trip'),
        pd.read_csv(statistics_path, encoding='utf-8-sig', float_precision='round_trip'))
    statistics = statistics.set_index('Region')
    annual, trends = {}, {}
    for region in MAIN_REGIONS:
        rows = series.loc[series.Region.eq(region)]
        estimate = statistics.loc[region]
        annual[region] = (rows.Year.to_numpy(int), rows.Count.to_numpy(float), int(rows.Count.sum()))
        trends[region] = (float(estimate.Trend_events_decade), rows.Fitted.to_numpy(float),
                          rows.CI95_half_width.to_numpy(float), float(estimate.CI95_events_decade),
                          float(estimate.MK_P))
    basin_results = [(float(statistics.loc[region, 'Trend_events_decade']),
                      float(statistics.loc[region, 'CI95_events_decade']),
                      float(statistics.loc[region, 'MK_P'])) for region, _ in BASIN_SPECS]
    return annual, trends, basin_results


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

def draw_basin_inset(ax_basin, basin_results):
    slopes = np.asarray([item[0] for item in basin_results])
    errors = np.asarray([item[1] for item in basin_results])
    xpos = np.arange(len(BASIN_SPECS))
    ax_basin.bar(
        xpos, slopes, yerr=errors, width=0.54,
        color=BASIN_COLORS, edgecolor="none", linewidth=0.7,
        error_kw={"ecolor": "#222222", "elinewidth": 0.45,
                   "capthick": 0.45, "capsize": 1.4},
    )
    ax_basin.set_xticks(xpos, [label for _, label in BASIN_SPECS])
    ax_basin.set_ylabel("Trend\n" r"(decade$^{-1}$)", fontsize=5.0, labelpad=1.0)
    ax_basin.yaxis.tick_left()
    ax_basin.yaxis.set_label_position("left")
    ax_basin.tick_params(direction="out", length=1.8, width=0.55,
                         pad=1, labelsize=5.0)
    ax_basin.spines[["top", "right"]].set_visible(False)
    lower = min(0.0, float(np.nanmin(slopes - errors)))
    upper = max(0.0, float(np.nanmax(slopes + errors)))
    span = max(upper - lower, 0.1)
    ax_basin.set_ylim(lower - 0.05 * span, upper + 0.24 * span)
    for xi, (slope, error, p_value) in enumerate(basin_results):
        mark = significance_mark(p_value)
        if mark:
            ax_basin.text(
                xi, slope + error + 0.035 * span, mark,
                ha="center", va="bottom", fontsize=6.0,
                fontweight="bold", clip_on=False,
            )

def plot_panel(region, annual, trend, output_dir, basin_results=None):
    specs = {
        "Global": ("FigM1a_Global_CLTC_frequency.png", 2.05, 0.27,
                   "#496181", "Global", [10, 20, 30, 40, 50]),
        "NH": ("FigM1b_NH_CLTC_frequency.png", 2.05, 0.27,
               "#C25D34", "NH", [0, 10, 20, 30, 40]),
        "NH20": ("FigM1c_NH_above20_CLTC_frequency.png", 2.55, 0.21,
                 "#96B8A7", "NH (> 20°N)", [0, 6, 12, 18, 24]),
    }
    filename, width, left, color, label, yticks = specs[region]
    years, counts, n_events = annual
    slope_decade, fitted, band, _, mk_p = trend
    fig, ax = plt.subplots(figsize=(width, 2.12), facecolor="white")
    fig.subplots_adjust(left=left, right=0.97, bottom=0.24, top=0.96)
    if basin_results is not None:
        ax_basin = ax.inset_axes([0.15, 0.70, 0.30, 0.22], facecolor="white", zorder=6)
        ax_basin.patch.set_alpha(1.0)
    ax.plot(years, counts, color=color, lw=1.10)
    ax.plot(years, fitted, color=color, lw=1.05, ls="--")
    ax.fill_between(years, fitted - band, fitted + band,
                    color=color, alpha=0.14, linewidth=0)
    p_label = r"$P<0.01$" if mk_p < 0.01 else rf"$P={mk_p:.2f}$"
    ax.text(0.975, 0.975,
            rf"Trend = {slope_decade:.1f} decade$^{{-1}}$" "\n" f"{p_label}",
            transform=ax.transAxes, ha="right", va="top")
    label_style = {}
    if region == "NH20":
        label_style = {"fontsize": 7.0, "bbox": dict(
            facecolor="white", edgecolor="none", alpha=0.82, pad=0.8)}
    ax.text(0.98, 0.035, label, transform=ax.transAxes,
            ha="right", va="bottom", **label_style)
    ax.set(xlim=(1978, 2026), ylim=(yticks[0], yticks[-1]), xlabel="Year",
           ylabel="Count of CLTC" if region == "Global" else "")
    ax.set_yticks(yticks)
    ax.set_xticks([1980, 2000, 2020])
    style_spines(ax)
    if basin_results is not None:
        draw_basin_inset(ax_basin, basin_results)
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"{region} events: {n_events}; saved: {output_file}")

def compose_fig1(output_dir):
    panels = [
        ("A", output_dir / "FigM1a_Global_CLTC_frequency.png", 2.05, 0.27),
        ("B", output_dir / "FigM1b_NH_CLTC_frequency.png", 2.05, 0.27),
        ("C", output_dir / "FigM1c_NH_above20_CLTC_frequency.png", 2.55, 0.21),
    ]
    panel_height = 2.12
    header_height = 0.22
    figure_width = sum(width for _, _, width, _ in panels)
    figure_height = panel_height + header_height
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    x_offset = 0.0
    for letter, path, width, axes_left in panels:
        panel = fig.add_axes([
            x_offset / figure_width, 0,
            width / figure_width, panel_height / figure_height,
        ])
        panel.imshow(plt.imread(path), interpolation="none")
        panel.set_axis_off()
        frame_left = x_offset + axes_left * width
        frame_top = 0.96 * panel_height
        fig.text(
            (frame_left - 0.35) / figure_width,
            (frame_top + 0.07) / figure_height,
            letter, ha="left", va="bottom", fontsize=10,
            fontweight="bold", color="#111111",
        )
        x_offset += width
    output_file = output_dir / "FigM1_composite.png"
    fig.savefig(output_file, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_dir, output_dir = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    annual, trends, basin_results = load_plot_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in MAIN_REGIONS:
        plot_panel(region, annual[region], trends[region], output_dir,
                   basin_results if region == 'NH20' else None)
    for (_, label), (slope, error, p_value) in zip(BASIN_SPECS, basin_results):
        print(f'{label}: trend={slope:.3f} ± {error:.3f} decade^-1, MK P={p_value:.17g}')
    compose_fig1(output_dir)


if __name__ == '__main__':
    main()
