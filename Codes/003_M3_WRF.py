#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "Database/WRF"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Results/003_M3_WRF"

SSTS = ("302K-2K", "302K")
DISPLAY_SST = {"302K": "302 K", "302K-2K": "300 K"}
DISTANCES = (-50, 0, 50)
TRACK_HOURS = np.arange(0, 25, 3, dtype=float)
PVT_MEAN_LABEL = "PVT (3–10 km)"
BOX_LABELS = (PVT_MEAN_LABEL, "PVT (1 km)", "HA", "VA", "DH")

BOX_DISPLAY_LABELS = {PVT_MEAN_LABEL: "PVT\n(3–10 km)", "PVT (1 km)": "PVT\n(1 km)"}
SST_LINESTYLES = {"302K-2K": "--", "302K": "-"}
DISTANCE_COLORS = {-50: "#496181", 0: "#C25D34", 50: "#96B8A7"}
BOX_COLORS = {"302K-2K": "#355C84", "302K": "#C25D34"}
DPI = 600
FIGSIZE = (8.3, 3.8)
FONT_SCALE = FIGSIZE[0] / 6.65
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7 * FONT_SCALE,
    "axes.labelsize": 7.5 * FONT_SCALE,
    "xtick.labelsize": 7 * FONT_SCALE,
    "ytick.labelsize": 7 * FONT_SCALE,
    "axes.linewidth": 0.7,
})


def read_table(path, keys, numeric_columns, expected_keys):
    require_file(path)
    table = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    missing = set(keys + numeric_columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all():
            raise ValueError(f"{path.name}: non-finite values in {column}")
    if table.duplicated(list(keys)).any():
        raise ValueError(f"{path.name}: duplicate plotting records")
    actual = set(table[list(keys)].itertuples(index=False, name=None))
    if actual != set(expected_keys):
        raise ValueError(f"{path.name}: unexpected or missing plotting records")
    return table


def load_plot_data(data_dir):
    trajectories = read_table(
        data_dir / "FigM3A_track_coordinates.csv",
        ("source_sst", "initial_distance_km", "time_h"),
        ("initial_distance_km", "time_h", "plot_x_km"),
        [(sst, distance, hour) for sst in SSTS for distance in DISTANCES for hour in TRACK_HOURS],
    )
    lines = []
    for distance in DISTANCES:
        for sst in SSTS:
            part = trajectories.loc[trajectories.source_sst.eq(sst)
                                    & trajectories.initial_distance_km.eq(distance)].sort_values("time_h")
            # plot_x_km already includes the initial distance; do not add it again.
            lines.append((sst, distance, part.plot_x_km.to_numpy(float)))
    boxes = read_table(
        data_dir / "FigM3B_motion_boxstats.csv", ("source_sst", "component"),
        ("whisker_low_ms", "q1_ms", "median_ms", "q3_ms", "whisker_high_ms", "mean_ms"),
        [(sst, component) for sst in SSTS for component in BOX_LABELS],
    )
    ordered = boxes[["whisker_low_ms", "q1_ms", "median_ms", "q3_ms", "whisker_high_ms"]].to_numpy(float)
    if (np.diff(ordered, axis=1) < 0).any():
        raise ValueError("Invalid ordering of box quartiles/whiskers")
    return lines, boxes


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def plot_panel_a(ax, lines):
    lo, hi = -110.0, 110.0
    ax.axvline(0, color="0.35", lw=.75, zorder=0)
    for sst, distance, values in lines:
        ax.plot(values, TRACK_HOURS, color=DISTANCE_COLORS[distance],
                linestyle=SST_LINESTYLES[sst], lw=1.8, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(-100, 101, 50))
    ax.set_ylim(0, 24)
    ax.set_yticks(TRACK_HOURS)
    ax.set_xlabel("Zonal distance to coast (km)")
    ax.set_ylabel("Time (h)")
    sst_handles = [
        Line2D([0], [0], color="0.15", linestyle=SST_LINESTYLES[s],
               lw=1.8, label=DISPLAY_SST[s]) for s in SSTS
    ]
    legend = ax.legend(handles=sst_handles, loc="lower right",
                       bbox_to_anchor=(1.025, .02), frameon=False,
                       fontsize=7 * FONT_SCALE)
    legend._legend_box.align = "left"


def plot_panel_b(ax, paired):
    labels = BOX_LABELS
    centers = np.arange(len(labels), 0, -1, dtype=float)
    offsets = {"302K-2K": +.18, "302K": -.18}
    max_abs_displayed = 0.0
    for sst in SSTS:
        subset = paired.loc[paired["source_sst"].eq(sst)]
        subset = subset.set_index("component")
        for label, center in zip(labels, centers):
            row = subset.loc[label]
            stats = {"whislo": row.whisker_low_ms, "q1": row.q1_ms,
                     "med": row.median_ms, "q3": row.q3_ms,
                     "whishi": row.whisker_high_ms}
            mean = float(row.mean_ms)
            max_abs_displayed = max(
                max_abs_displayed, abs(stats["whislo"]),
                abs(stats["whishi"]), abs(mean),
            )
            pos = center + offsets[sst]
            ax.bxp(
                [stats], positions=[pos], widths=.30, vert=False,
                patch_artist=True, showfliers=False,
                boxprops={"facecolor": BOX_COLORS[sst], "edgecolor": "0.15",
                          "linewidth": .7, "alpha": 1.0},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "0.25", "linewidth": .7},
                capprops={"color": "0.25", "linewidth": .7},
            )
            ax.scatter(mean, pos, marker="D", s=13,
                       facecolor="white", edgecolor="black",
                       linewidth=.65, zorder=5)
    tick_step = 2.0
    limit = max(
        tick_step, tick_step * np.ceil((max_abs_displayed + .5) / tick_step)
    )
    ax.set_xlim(-limit, limit)
    ax.set_xticks(np.arange(-limit, limit + tick_step / 2, tick_step))
    ax.axvline(0, color="0.25", lw=.8, zorder=0)
    ax.set_yticks(centers)
    ax.set_yticklabels([BOX_DISPLAY_LABELS.get(label, label) for label in labels])
    ax.set_ylim(.5, 5.5)
    ax.set_xlabel("Cross-shore motion contribution (m s$^{-1}$)")
    ax.legend(handles=[
        Patch(facecolor=BOX_COLORS[s], edgecolor="0.15",
              label=DISPLAY_SST[s]) for s in SSTS
    ] + [
        Line2D([], [], marker="D", linestyle="none", markersize=4,
               markerfacecolor="white", markeredgecolor="black",
               label="Mean")
    ], loc="lower right", frameon=False,
              fontsize=7 * FONT_SCALE)


def style_axis(ax):
    ax.tick_params(direction="out", length=3, width=.7,
                   labelsize=7 * FONT_SCALE, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(.7)


def save_standalone_panels(offset_lines, paired, output_dir):
    fig_a, ax_a = plt.subplots(figsize=(3.7, 3.8))
    plot_panel_a(ax_a, offset_lines)
    style_axis(ax_a)
    fig_a.subplots_adjust(left=.17, right=.97, bottom=.17, top=.91)
    fig_a.savefig((output_dir / "FigM3a_zonal_track_offset.png"), dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(4.4, 3.8))
    plot_panel_b(ax_b, paired)
    style_axis(ax_b)
    fig_b.subplots_adjust(left=.28, right=.98, bottom=.17, top=.91)
    fig_b.savefig((output_dir / "FigM3b_cross_shore_velocity_contributions.png"), dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig_b)


def draw_figures(output_dir, offset_lines, paired):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE,
                             gridspec_kw={"width_ratios": [1.0, 1.2]})
    plot_panel_a(axes[0], offset_lines)
    plot_panel_b(axes[1], paired)
    for ax in axes:
        style_axis(ax)
    fig.subplots_adjust(left=.08, right=.985, bottom=.17, top=.91, wspace=.38)
    for label, ax in zip("AB", axes):
        box = ax.get_position()
        fig.text(box.x0 - .048, box.y1 + .023, label,
                 fontsize=10 * FONT_SCALE,
                 fontweight="bold", ha="left", va="bottom")
    fig.savefig((output_dir / "FigM3_composite.png"), dpi=DPI, facecolor="white", bbox_inches="tight")
    save_standalone_panels(offset_lines, paired, output_dir)
    plt.close(fig)
    print(f"Saved three PNG files to: {output_dir}")



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing the two M3 plotting CSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    print(f"Plotting data: {data_dir}")
    lines, boxes = load_plot_data(data_dir)
    draw_figures(output_dir, lines, boxes)


if __name__ == "__main__":
    main()
