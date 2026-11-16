#!/usr/bin/env python3
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
DEFAULT_DATA_FILE = PROJECT_DIR / 'Database/WRF/FigS8_track_coordinates.csv'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/003_S8_WRF_Tracks'
PANEL_FILENAMES = {
    'A': 'FigS8a_EXP_tracks_300K.png', 'B': 'FigS8b_CTRL_tracks_300K.png',
    'C': 'FigS8c_EXP_tracks_302K.png', 'D': 'FigS8d_CTRL_tracks_302K.png',
}


SSTS = ("302K-2K", "302K")

DISPLAY_SST = {"302K-2K": "300 K", "302K": "302 K"}

WINDS_MS = tuple(range(34, 41))

INITIAL_DISTANCES_KM = (-50, 0, 50)

DISTANCES = tuple(str(x) for x in INITIAL_DISTANCES_KM) + ("CTRL",)

HOURS = np.arange(0, 25, 3, dtype=int)

DPI = 600

FIGURE_WIDTH = 8.3

FONT_SCALE = FIGURE_WIDTH / 6.65

DISTANCE_COLORS = {-50: "#496181", 0: "#C25D34",
                   50: "#96B8A7", "CTRL": "#1F1F1F"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7 * FONT_SCALE,
    "axes.labelsize": 7.5 * FONT_SCALE,
    "xtick.labelsize": 7 * FONT_SCALE,
    "ytick.labelsize": 7 * FONT_SCALE,
    "axes.linewidth": .7,
})

def normalize_distance(series):
    return (series.astype(str).str.strip()
            .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True))


PANEL_CASES = {
    'A': ('302K-2K', 'EXP'), 'B': ('302K-2K', 'CTRL'),
    'C': ('302K', 'EXP'), 'D': ('302K', 'CTRL'),
}
TRACK_COLUMNS = ('panel', 'initial_distance_km', 'initial_wind_ms',
                 'time_h', 'series_type', 'x_km')


def validate_tracks(table):
    """Validate all 504 member and 72 stored-mean coordinates without refitting."""
    if table.columns.duplicated().any() or set(table.columns) != set(TRACK_COLUMNS):
        raise ValueError(f'S8 requires exactly these plotting columns: {TRACK_COLUMNS}')
    table = table.loc[:, list(TRACK_COLUMNS)].copy()
    table['initial_distance_km'] = normalize_distance(table['initial_distance_km'])
    for name in ('initial_wind_ms', 'time_h', 'x_km'):
        table[name] = pd.to_numeric(table[name], errors='raise')
    if len(table) != 576 or not table['series_type'].isin(('member', 'mean')).all():
        raise ValueError('S8 requires 576 coordinates: 504 members and 72 means')
    if not np.isfinite(table[['time_h', 'x_km']].to_numpy(float)).all():
        raise ValueError('S8 time and zonal coordinates must be finite')
    members = table.loc[table.series_type.eq('member')]
    means = table.loc[table.series_type.eq('mean')]
    member_keys = ['panel', 'initial_distance_km', 'initial_wind_ms', 'time_h']
    mean_keys = ['panel', 'initial_distance_km', 'time_h']
    expected_members, expected_means = [], []
    for panel, (_, kind) in PANEL_CASES.items():
        distances = tuple(str(value) for value in INITIAL_DISTANCES_KM) if kind == 'EXP' else ('CTRL',)
        expected_members.extend((panel, distance, wind, hour)
            for distance in distances for wind in WINDS_MS for hour in HOURS)
        expected_means.extend((panel, distance, hour)
            for distance in distances for hour in HOURS)
    if (len(members) != 504 or members.duplicated(member_keys).any()
            or set(members[member_keys].itertuples(index=False, name=None)) != set(expected_members)):
        raise ValueError('S8 requires every panel/distance/wind/hour member exactly once')
    if (len(means) != 72 or means.duplicated(mean_keys).any()
            or set(means[mean_keys].itertuples(index=False, name=None)) != set(expected_means)
            or not means.initial_wind_ms.isna().all()):
        raise ValueError('S8 requires each stored mean once, with initial_wind_ms blank')
    table['_series_order'] = table.series_type.map({'member': 0, 'mean': 1})
    return (table.sort_values(['panel', 'initial_distance_km', '_series_order', 'initial_wind_ms', 'time_h'],
                              kind='stable', na_position='last')
            .drop(columns='_series_order').reset_index(drop=True))


def distance_label(distance_km):
    if distance_km < 0:
        return f"−{abs(distance_km)} km"
    if distance_km > 0:
        return f"+{distance_km} km"
    return "0 km"

def style_axis(ax):
    ax.tick_params(direction="out", length=3, width=.75, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(.75)

def add_letter(fig, ax, letter):
    box = ax.get_position()
    fig.text(box.x0 - .035, box.y1 + .014, letter.lower(),
             fontsize=10 * FONT_SCALE, fontweight="bold",
             ha="left", va="bottom")

def track_xlim(table, column, subset_mask):
    """Use a common zero-centered scale for the two SST panels."""
    values = table.loc[subset_mask, column].to_numpy(float)
    extent = float(np.max(np.abs(values))) + 10.0
    extent = 110.0 if extent <= 110.0 else 50.0 * np.ceil(extent / 50.0)
    return -extent, extent



def load_plot_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing S8 plotting data: {path}; run preprocessing first')
    return validate_tracks(pd.read_csv(path, float_precision='round_trip'))


def plotted_series(table, panel, distance, wind=None):
    selected = table.loc[table.panel.eq(panel) & table.initial_distance_km.eq(str(distance))]
    if wind is None:
        selected = selected.loc[selected.series_type.eq('mean')]
    else:
        selected = selected.loc[selected.series_type.eq('member') & selected.initial_wind_ms.eq(wind)]
    selected = selected.sort_values('time_h')
    if len(selected) != 9 or not np.array_equal(selected.time_h.values, HOURS):
        raise ValueError(f'Incomplete saved S8 curve: {panel}/{distance}/{wind}')
    return selected.x_km.to_numpy(float)


def draw_tracks(ax, table, panel, xlim):
    kind = PANEL_CASES[panel][1]
    ax.axvline(0, color='.35', lw=.75, zorder=1)
    distances = INITIAL_DISTANCES_KM if kind == 'EXP' else ('CTRL',)
    for distance in distances:
        for wind in WINDS_MS:
            x = plotted_series(table, panel, distance, wind)
            ax.plot(x, HOURS, color=DISTANCE_COLORS[distance],
                    linestyle='-', lw=.7, alpha=.25, zorder=2)
        mean = plotted_series(table, panel, distance)
        ax.plot(mean, HOURS, color=DISTANCE_COLORS[distance],
                linestyle='-', lw=1.8, zorder=3)
    ax.set_xlabel('Zonal distance to coast (km)' if kind == 'EXP'
                  else 'Zonal distance to initial center (km)')
    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(50 * np.ceil(xlim[0] / 50), xlim[1] + 1, 50))
    ax.set_ylim(0, 24)
    ax.set_yticks(HOURS)
    ax.set_ylabel('Time (h)')
    style_axis(ax)


def distance_handles():
    return [Line2D([], [], color=DISTANCE_COLORS[distance], lw=1.8,
                   label=distance_label(distance)) for distance in INITIAL_DISTANCES_KM]


def make_composite(table):
    members = table.series_type.eq('member')
    exp_xlim = track_xlim(table, 'x_km', members & table.panel.isin(('A', 'C')))
    ctrl_xlim = track_xlim(table, 'x_km', members & table.panel.isin(('B', 'D')))
    fig = plt.figure(figsize=(FIGURE_WIDTH, 7.0), facecolor='white')
    left, right, gap = .09, .96, .09
    width = (right - left - gap) / 2
    x2 = left + width + gap
    axes = {
        'A': fig.add_axes([left, .57, width, .34]),
        'B': fig.add_axes([x2, .57, width, .34]),
        'C': fig.add_axes([left, .11, width, .34]),
        'D': fig.add_axes([x2, .11, width, .34]),
    }
    for panel, (_, kind) in PANEL_CASES.items():
        draw_tracks(axes[panel], table, panel, exp_xlim if kind == 'EXP' else ctrl_xlim)
        add_letter(fig, axes[panel], panel)
    shared_legend = fig.legend(
        handles=distance_handles() + [Line2D([], [], color=DISTANCE_COLORS['CTRL'], lw=1.8, label='CTRL')],
        loc='center', bbox_to_anchor=(.5, .965), ncol=4, frameon=False,
        fontsize=7 * FONT_SCALE, columnspacing=1.5, handlelength=2.0,
    )
    return fig, axes, shared_legend


def save_figures(table, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, shared_legend = make_composite(table)
    try:
        fig.savefig(output_dir / 'FigS8_composite.png', dpi=DPI, facecolor='white',
                    bbox_inches='tight', pad_inches=.06)
        for text in fig.texts:
            text.set_visible(False)
        shared_legend.set_visible(False)
        for panel, axis in axes.items():
            for other in axes.values():
                other.set_visible(other is axis)
            legend = None
            if PANEL_CASES[panel][1] == 'EXP':
                legend = axis.legend(handles=distance_handles(), loc='lower center',
                    bbox_to_anchor=(.5, 1.025), ncol=3, frameon=False,
                    fontsize=7 * FONT_SCALE, columnspacing=1.5, handlelength=2.0)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [axis.get_tightbbox(renderer)]
            if legend is not None:
                boxes.append(legend.get_window_extent(renderer))
            extent = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.06)
            path = output_dir / PANEL_FILENAMES[panel]
            fig.savefig(path, dpi=DPI, facecolor='white', bbox_inches=extent, pad_inches=0)
            if legend is not None:
                legend.remove()
            print(f'Saved: {path}')
        print(f'Saved: {output_dir / "FigS8_composite.png"}')
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    save_figures(load_plot_data(args.data_file.expanduser().resolve()),
                 args.output_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
