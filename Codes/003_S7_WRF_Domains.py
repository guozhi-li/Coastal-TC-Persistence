#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database/WRF'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/003_S7_WRF_Domains'
INTENSITY_COLUMNS = ['sst', 'initial_wind_ms', 'distance', 'forecast_hour', 'mws_ms', 'mcp_hpa']
DOMAIN_COLUMNS = ['domain', 'x0_km', 'y0_km', 'width_km', 'height_km']
PANEL_NAMES = {
    'A':'FigS7a_domain_configuration.png', 'B':'FigS7b_initial_coastal_positions.png',
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

LAND_COLOR = "#C99B4A"

OCEAN_COLOR = "#68A9D5"

COAST_COLOR = "#25364A"

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
    fig.text(box.x0 - .035, box.y1 + .014, letter,
             fontsize=10 * FONT_SCALE, fontweight="bold",
             ha="left", va="bottom")

def draw_domains(ax, geometry):
    w1, h1, left2, bottom2, w2, h2 = geometry
    ax.add_patch(Rectangle((-w1 / 2, -h1 / 2), w1 / 2, h1,
                           facecolor=LAND_COLOR, edgecolor="none"))
    ax.add_patch(Rectangle((0, -h1 / 2), w1 / 2, h1,
                           facecolor=OCEAN_COLOR, edgecolor="none"))
    ax.axvline(0, color=COAST_COLOR, lw=.9)
    ax.add_patch(Rectangle((left2, bottom2), w2, h2, fill=False,
                           edgecolor="black", lw=1.35))
    ax.text(.025, .975, "d01", transform=ax.transAxes, va="top")
    ax.text(left2 + .04 * w2, bottom2 + .95 * h2, "d02", va="top")
    ax.set(xlim=(-w1 / 2, w1 / 2), ylim=(-h1 / 2, h1 / 2),
           xlabel="Zonal distance (km)", ylabel="Meridional distance (km)")
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    style_axis(ax)

def load_tc_icon(icon_file):
    """Load the original TC icon, preserving its existing alpha crop."""
    if not icon_file.is_file():
        raise FileNotFoundError(f"Missing prepared icon: {icon_file}; run preprocessing first")
    icon = plt.imread(icon_file)
    if icon.ndim == 3 and icon.shape[2] == 4:
        visible = icon[..., 3] > .01
        if visible.any():
            jj, ii = np.where(visible)
            icon = icon[jj.min():jj.max() + 1, ii.min():ii.max() + 1]
    return icon

def draw_initial_distances(ax, icon):
    x_min, x_max = -100.0, 100.0
    for row, distance in enumerate(INITIAL_DISTANCES_KM):
        coast_x = -float(distance)
        y = float(len(INITIAL_DISTANCES_KM) - row - 1)
        ax.add_patch(Rectangle((x_min, y - .47), coast_x - x_min, .94,
                               facecolor=LAND_COLOR, edgecolor="none"))
        ax.add_patch(Rectangle((coast_x, y - .47), x_max - coast_x, .94,
                               facecolor=OCEAN_COLOR, edgecolor="none"))
        ax.plot([coast_x, coast_x], [y - .47, y + .47],
                color=COAST_COLOR, lw=1.0)
        if distance:
            ax.annotate("", xy=(0, y - .18), xytext=(coast_x, y - .18),
                        arrowprops={"arrowstyle": "<->", "lw": .9, "color": "black"})
        if icon is None:
            ax.plot(0, y, marker="*", markersize=8, color="#BA1E2D",
                    markeredgecolor="white", markeredgewidth=.5, zorder=5)
        else:
            ax.add_artist(AnnotationBbox(
                OffsetImage(icon, zoom=15.0 / max(icon.shape[:2])),
                (0, y), frameon=False, pad=0, zorder=5
            ))
    ax.set(xlim=(x_min, x_max), ylim=(-.49, 2.49),
           xlabel="Zonal distance (km)", ylabel="Initial distance to coast")
    ax.set_yticks((2, 1, 0), ("−50 km", "0 km", "+50 km"))
    ax.yaxis.labelpad = 8
    ax.axvline(0, color="black", lw=.6, ls=":")
    style_axis(ax)

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
    domains = pd.read_csv(data_dir/'FigS7_domains.csv', float_precision='round_trip')
    if list(domains.columns) != DOMAIN_COLUMNS or domains.domain.tolist() != ['d01','d02']:
        raise ValueError('Expected ordered d01/d02 domain rows')
    values = domains[DOMAIN_COLUMNS[1:]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:,2:] <= 0):
        raise ValueError('Domain extents must be finite, with positive width/height')
    outer, inner = values
    if not np.allclose(outer[:2], -outer[2:]/2, atol=1e-10, rtol=0):
        raise ValueError('d01 must be centered at the origin')
    if np.any(inner[:2] < outer[:2]) or np.any(inner[:2]+inner[2:] > outer[:2]+outer[2:]):
        raise ValueError('d02 must lie inside d01')
    geometry = (outer[2], outer[3], inner[0], inner[1], inner[2], inner[3])
    return intensity, geometry, load_tc_icon(data_dir/'FigS7_TC_center.png')


def save_panel(fig, path, axis, extras, legend=None, legend_anchor=None, tc_key=None):
    """Use the composite artists, preserving size and restoring shared artwork."""
    artists = list(fig.axes)+list(fig.texts)+list(fig.legends)+list(fig.artists)
    states = [(artist,artist.get_visible()) for artist in artists]
    key_positions = None
    try:
        keep = [axis]+list(extras)
        for artist,visible in states:
            artist.set_visible(visible and artist in keep)
        box = axis.get_position()
        center, top = box.x0+box.width/2, box.y1+.055
        if legend is not None:
            legend.set_bbox_to_anchor((center,top),transform=fig.transFigure)
        if tc_key is not None:
            icon,text = tc_key
            key_positions = (icon.xy,icon.xybox,text.get_position())
            icon.xy = icon.xybox = (center-.055,top)
            text.set_position((center-.030,top))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [axis.get_tightbbox(renderer)]+[artist.get_window_extent(renderer) for artist in extras]
        bounds = Bbox.union([box for box in boxes if box is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(.06)
        fig.savefig(path,dpi=DPI,facecolor='white',bbox_inches=bounds,pad_inches=0)
    finally:
        if legend is not None:
            legend.set_bbox_to_anchor(legend_anchor,transform=fig.transFigure)
        if key_positions is not None:
            tc_key[0].xy,tc_key[0].xybox = key_positions[:2]
            tc_key[1].set_position(key_positions[2])
        for artist,visible in states:
            artist.set_visible(visible)


def draw_figures(output_dir, intensity, geometry, tc_icon):
    output_dir.mkdir(parents=True,exist_ok=True)
    wind_limits = limits(intensity,'mws_ms',.8)
    pressure_limits = limits(intensity,'mcp_hpa',1.5)
    fig = plt.figure(figsize=(FIGURE_WIDTH,9.85),facecolor='white')
    try:
        left, right, gap = .11, .97, .14
        width = (right - left - gap) / 2
        x2 = left + width + gap
        top_h = width * fig.get_figwidth() / fig.get_figheight()
        axes = {
            "A": fig.add_axes([left, .62, width, top_h]),
            "B": fig.add_axes([x2, .62, width, top_h]),
            "C": fig.add_axes([left, .315, width, .18]),
            "D": fig.add_axes([x2, .315, width, .18]),
            "E": fig.add_axes([left, .035, width, .18]),
            "F": fig.add_axes([x2, .035, width, .18]),
        }
        draw_domains(axes["A"], geometry)
        draw_initial_distances(axes["B"], tc_icon)
        for letter, sst, column, ylim in (
            ("C", SSTS[0], "mws_ms", wind_limits),
            ("D", SSTS[1], "mws_ms", wind_limits),
            ("E", SSTS[0], "mcp_hpa", pressure_limits),
            ("F", SSTS[1], "mcp_hpa", pressure_limits),
        ):
            draw_metric(axes[letter], intensity, sst, column, ylim)
        for letter in "CE":
            axes[letter].set_ylabel(
                "Maximum 10-m wind speed (m s$^{-1}$)" if letter == "C"
                else "Minimum central pressure (hPa)"
            )
        # Align all left-column vertical titles in figure coordinates.
        # Preserve A's natural label position, then reuse its centre for C and E.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        reference_label = axes["A"].yaxis.label
        reference_x = reference_label.get_transform().transform(
            reference_label.get_position()
        )[0] / fig.bbox.width
        for letter in "CE":
            axis = axes[letter]
            box = axis.get_position()
            label_x = (reference_x - box.x0) / box.width
            axis.yaxis.set_label_coords(label_x, 0.5)
        for letter, ax in axes.items():
            add_letter(fig, ax, letter)
        land_legend = fig.legend(
            handles=[Patch(facecolor=LAND_COLOR,label='Land'),Patch(facecolor=OCEAN_COLOR,label='Ocean')],
            loc='center',bbox_to_anchor=(.43,.985),ncol=2,frameon=False,
            fontsize=8*FONT_SCALE,columnspacing=2.5,
        )
        tc_key = AnnotationBbox(
            OffsetImage(tc_icon,zoom=15.0/max(tc_icon.shape[:2])),(.60,.985),
            xycoords=fig.transFigure,frameon=False,pad=0,
        )
        fig.add_artist(tc_key)
        tc_text = fig.text(.625,.985,'TC center',va='center',fontsize=10*FONT_ENLARGEMENT)
        line_legend = distance_legend(fig,(.53,.545))
        fig.savefig(output_dir/'FigS7_composite.png',dpi=DPI,facecolor='white',bbox_inches='tight',pad_inches=.06)
        save_panel(fig,output_dir/PANEL_NAMES['A'],axes['A'],[land_legend],
                   legend=land_legend,legend_anchor=(.43,.985))
        save_panel(fig,output_dir/PANEL_NAMES['B'],axes['B'],[tc_key,tc_text],tc_key=(tc_key,tc_text))
        for letter in 'CDEF':
            save_panel(fig,output_dir/PANEL_NAMES[letter],axes[letter],[line_legend],
                       legend=line_legend,legend_anchor=(.53,.545))
    finally:
        plt.close(fig)
    print(f'Saved composite and six individual panels: {output_dir}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=resolve_project_path,default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir',type=resolve_project_path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args()
    intensity,geometry,icon=read_plot_data(args.data_dir)
    draw_figures(args.output_dir,intensity,geometry,icon)


if __name__=='__main__':
    main()
