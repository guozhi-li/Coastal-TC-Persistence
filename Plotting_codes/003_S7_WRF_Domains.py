#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database/WRF'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/003_S7_WRF_Domains'
INTENSITY_COLUMNS = ['sst', 'initial_wind_ms', 'distance', 'forecast_hour', 'mws_ms', 'mcp_hpa']
PANEL_NAMES = {
    'C':'FigS7c_maximum_10m_wind_300K.png', 'D':'FigS7d_maximum_10m_wind_302K.png',
    'E':'FigS7e_minimum_central_pressure_300K.png', 'F':'FigS7f_minimum_central_pressure_302K.png',
}
SSTS = ("302K-2K", "302K")

DISPLAY_SST = {"302K-2K": "300 K", "302K": "302 K"}

WINDS_MS = tuple(range(34, 41))

INITIAL_DISTANCES_KM = (-50, 0, 50)

DISTANCES = tuple(str(x) for x in INITIAL_DISTANCES_KM) + ("CTRL",)

HOURS = np.arange(0, 25, 1, dtype=int)

DPI = 600

FIGURE_WIDTH = 8.3

FONT_ENLARGEMENT = 1.25

FONT_SCALE = FIGURE_WIDTH / 6.65 * FONT_ENLARGEMENT


DISTANCE_COLORS = {-50: "#496181", 0: "#C25D34",
                   50: "#96B8A7", "CTRL": "#1F1F1F"}

def distance_label(distance_km):
    if distance_km < 0:
        return f"−{abs(distance_km)} km"
    if distance_km > 0:
        return f"+{distance_km} km"
    return "0 km"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7 * FONT_SCALE,
    "axes.labelsize": 7.5 * FONT_SCALE,
    "xtick.labelsize": 7 * FONT_SCALE,
    "ytick.labelsize": 7 * FONT_SCALE,
    "axes.linewidth": .7,
})

def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def normalize_distance(series):
    return (series.astype(str).str.strip()
            .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True))

def read_case_table(path, value_columns):
    keys = ["sst", "initial_wind_ms", "distance", "forecast_hour"]
    columns = keys + list(value_columns)
    table = pd.read_csv(require_file(path), usecols=columns, low_memory=False, float_precision="round_trip")
    table["sst"] = table["sst"].astype(str).str.strip()
    table["distance"] = normalize_distance(table["distance"])
    for name in ["initial_wind_ms", "forecast_hour", *value_columns]:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table = table.loc[
        table["sst"].isin(SSTS)
        & table["initial_wind_ms"].isin(WINDS_MS)
        & table["distance"].isin(DISTANCES)
        & table["forecast_hour"].isin(HOURS)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [SSTS, WINDS_MS, DISTANCES, HOURS], names=keys
    )
    actual = pd.MultiIndex.from_frame(table[keys])
    if actual.has_duplicates:
        raise ValueError(f"{path}: duplicate case/time rows")
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(f"{path}: missing case/time rows; first five: {list(missing[:5])}")
    if len(table) != len(expected):
        raise ValueError(f"{path}: expected {len(expected)} selected rows, found {len(table)}")
    return table.sort_values(keys).reset_index(drop=True)

def style_axis(ax):
    ax.tick_params(direction="out", length=3, width=.75, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(.75)

def add_letter(fig, ax, letter):
    box = ax.get_position()
    return fig.text(box.x0 - .035, box.y1 + .014, letter.lower(),
             fontsize=10 * FONT_SCALE, fontweight="bold",
             ha="left", va="bottom")


def case_series(table, sst, wind, distance, column):
    sub = table.loc[
        table["sst"].eq(sst)
        & table["initial_wind_ms"].eq(wind)
        & table["distance"].eq(str(distance))
    ].sort_values("forecast_hour")
    if len(sub) != len(HOURS) or not np.array_equal(
        sub["forecast_hour"].to_numpy(int), HOURS
    ):
        raise ValueError(f"Incomplete series: {sst}/{wind}/{distance}")
    return sub[column].to_numpy(float)

def limits(table, column, minimum_pad):
    values = table[column].to_numpy(float)
    pad = max(minimum_pad, .07 * (values.max() - values.min()))
    return float(values.min() - pad), float(values.max() + pad)

def draw_metric(ax, table, sst, column, ylim):
    for distance in DISTANCES:
        color = DISTANCE_COLORS["CTRL" if distance == "CTRL" else int(distance)]
        members = []
        for wind in WINDS_MS:
            values = case_series(table, sst, wind, distance, column)
            members.append(values)
            ax.plot(HOURS, values, color=color, lw=.45, alpha=.18)
        ax.plot(HOURS, np.mean(members, axis=0), color=color, lw=1.45)
    ax.set(xlim=(0, 24), ylim=ylim, xlabel="Time (h)")
    ax.set_xticks(np.arange(0, 25, 3))
    style_axis(ax)

def distance_legend(fig, anchor):
    return fig.legend(
        handles=[
            *[Line2D([], [], color=DISTANCE_COLORS[d], lw=1.5,
                     label=distance_label(d))
              for d in INITIAL_DISTANCES_KM],
            Line2D([], [], color=DISTANCE_COLORS["CTRL"],
                   lw=1.5, label="CTRL"),
        ],
        loc="center", bbox_to_anchor=anchor,
        ncol=4, frameon=False, fontsize=7 * FONT_SCALE,
        columnspacing=1.5, handlelength=2.0,
    )


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def read_plot_data(data_dir):
    path = data_dir/'FigS7_intensity.csv'
    if list(pd.read_csv(path,nrows=0).columns) != INTENSITY_COLUMNS:
        raise ValueError(f'Expected six intensity columns: {INTENSITY_COLUMNS}')
    intensity = read_case_table(path, ('mws_ms','mcp_hpa'))
    if not np.isfinite(intensity[['mws_ms','mcp_hpa']].to_numpy(float)).all():
        raise ValueError('Nonfinite intensity values')
    return intensity


def save_panel(fig, path, axis, extras, legend=None, legend_anchor=None):
    """Use the composite artists, preserving size and restoring shared artwork."""
    artists = list(fig.axes)+list(fig.texts)+list(fig.legends)+list(fig.artists)
    states = [(artist,artist.get_visible()) for artist in artists]
    try:
        keep = [axis]+list(extras)
        for artist,visible in states:
            artist.set_visible(visible and artist in keep)
        box = axis.get_position()
        center, top = box.x0+box.width/2, box.y1+.055
        if legend is not None:
            legend.set_bbox_to_anchor((center,top),transform=fig.transFigure)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if legend is not None:
            # Place the standalone legend above the full panel, including its
            # vertical title, so the long wind-speed title is clear.
            panel_boxes = [axis.get_tightbbox(renderer),
                           axis.yaxis.label.get_window_extent(renderer)] + [
                artist.get_window_extent(renderer)
                for artist in extras if artist is not legend
            ]
            panel_top = Bbox.union([b for b in panel_boxes if b is not None]).y1
            legend_height = legend.get_window_extent(renderer).height
            top = (panel_top + .08 * fig.dpi + legend_height / 2) / fig.bbox.height
            legend.set_bbox_to_anchor((center, top), transform=fig.transFigure)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        boxes = [axis.get_tightbbox(renderer),
                 axis.yaxis.label.get_window_extent(renderer)] + [
            artist.get_window_extent(renderer) for artist in extras
        ]
        bounds = Bbox.union([box for box in boxes if box is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(.06)
        fig.savefig(path,dpi=DPI,facecolor='white',bbox_inches=bounds,pad_inches=0)
    finally:
        if legend is not None:
            legend.set_bbox_to_anchor(legend_anchor,transform=fig.transFigure)
        for artist,visible in states:
            artist.set_visible(visible)


def draw_figures(output_dir, intensity):
    output_dir.mkdir(parents=True, exist_ok=True)
    wind_limits = limits(intensity, 'mws_ms', .8)
    pressure_limits = limits(intensity, 'mcp_hpa', 1.5)
    # Preserve the original intensity-panel dimensions and font sizes.
    figure_height = 5.6
    panel_height = .18 * 9.85 / figure_height
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height), facecolor='white')
    try:
        left, right, gap = .11, .97, .14
        width = (right - left - gap) / 2
        x2 = left + width + gap
        axes = {
            'C': fig.add_axes([left, .575, width, panel_height]),
            'D': fig.add_axes([x2, .575, width, panel_height]),
            'E': fig.add_axes([left, .09, width, panel_height]),
            'F': fig.add_axes([x2, .09, width, panel_height]),
        }
        for letter, sst, column, ylim in (
            ('C', SSTS[0], 'mws_ms', wind_limits),
            ('D', SSTS[1], 'mws_ms', wind_limits),
            ('E', SSTS[0], 'mcp_hpa', pressure_limits),
            ('F', SSTS[1], 'mcp_hpa', pressure_limits),
        ):
            draw_metric(axes[letter], intensity, sst, column, ylim)
        axes['C'].set_ylabel('Maximum 10-m wind speed (m s$^{-1}$)')
        axes['E'].set_ylabel('Minimum central pressure (hPa)')
        fig.align_ylabels([axes['C'], axes['E']])
        for letter, ax in axes.items():
            add_letter(fig, ax, letter)
        legend_anchor = (.54, .97)
        line_legend = distance_legend(fig, legend_anchor)
        fig.savefig(output_dir / 'FigS7_composite.png', dpi=DPI, facecolor='white',
                    bbox_inches='tight', pad_inches=.06)
        for letter in 'CDEF':
            save_panel(fig, output_dir / PANEL_NAMES[letter], axes[letter],
                       [line_legend], legend=line_legend,
                       legend_anchor=legend_anchor)
    finally:
        plt.close(fig)
    print(f'Saved composite and four individual panels (c-f): {output_dir}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=resolve_project_path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',type=resolve_project_path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args()
    intensity=read_plot_data(args.data_dir)
    draw_figures(args.output_dir,intensity)


if __name__=='__main__':
    main()