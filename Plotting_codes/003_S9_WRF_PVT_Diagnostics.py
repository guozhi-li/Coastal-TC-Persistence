#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Database/WRF'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results/003_S9_WRF_PVT_Diagnostics'

SOURCE_SSTS = ("302K-2K", "302K")

DISPLAY_SSTS = {"302K-2K": "300 K", "302K": "302 K"}

AXIS_KM = np.arange(-120.0, 121.0, 5.0)

DPI = 600

VECTOR_COLORS = {
    "dh": "#151515", "dh_x": "#355F9B",
    "dh_y": "#528056", "dh_z": "#C25D34",
}

VECTOR_LABELS = {
    "dh": "DH",
    "dh_x": r"DH-$X$",
    "dh_y": r"DH-$Y$",
    "dh_z": r"DH-$Z$",
}

PV_TENDENCY_UNIT = (
    r"10$^{-6}$ K m$^{2}$ kg$^{-1}$ s$^{-1}$ h$^{-1}$"
)

CMAP = LinearSegmentedColormap.from_list(
    "s9_red_blue",
    ["#3B79AE", "#5F9FC0", "#8ABBD0", "#B7D3E3", "#E5EFF7",
     "#FFFFFF", "#FBEDE7", "#F6C9C0", "#EE9E97", "#DA848D", "#C07278"],
    N=256,
)

X_KM, Y_KM = np.meshgrid(AXIS_KM, AXIS_KM)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.linewidth": .8,
})

def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_plot_data(data_dir):
    nc_path = data_dir / 'FigS9_DH_spatial.nc'
    csv_path = data_dir / 'FigS9_DH_vectors.csv'
    for path in (nc_path, csv_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing S9 plotting data: {path}')
    with xr.open_dataset(nc_path, engine='h5netcdf') as stored:
        if set(stored.data_vars) != {'dh1'} or stored.dh1.dims != ('sst', 'y_km', 'x_km'):
            raise ValueError('S9 NC must contain dh1 on (sst, y_km, x_km)')
        np.testing.assert_array_equal(stored.sst, [300, 302])
        np.testing.assert_array_equal(stored.x_km, AXIS_KM)
        np.testing.assert_array_equal(stored.y_km, AXIS_KM)
        if (stored.attrs.get('source_sst_300K') != '302K-2K'
                or stored.attrs.get('source_sst_302K') != '302K'
                or stored.attrs.get('height_m') != 1000.
                or stored.attrs.get('case_count') != 21):
            raise ValueError('Incorrect source experiment or aggregation metadata')
        fields = stored.dh1.values.copy()
        scale = float(stored.attrs['arrow_km_per_ms'])
    if np.isinf(fields).any() or not all(np.isfinite(field).any() for field in fields):
        raise ValueError('Invalid DH fields')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid saved arrow scale')
    vectors = pd.read_csv(csv_path, float_precision='round_trip')
    if list(vectors.columns) != ['SST_K', 'Term', 'CX_m_s', 'CY_m_s']:
        raise ValueError('Incorrect S9 vector columns')
    terms = ('DH-X', 'DH-Y', 'DH-Z', 'DH')
    expected = [(sst, term) for sst in (300, 302) for term in terms]
    if list(vectors[['SST_K', 'Term']].itertuples(index=False, name=None)) != expected:
        raise ValueError('Expected eight ordered DH component vectors')
    if not np.isfinite(vectors[['CX_m_s', 'CY_m_s']].to_numpy(float)).all():
        raise ValueError('Nonfinite DH vectors')
    results = {}
    for index, (temperature, source_sst) in enumerate(zip((300, 302), SOURCE_SSTS)):
        panel = vectors.loc[vectors.SST_K == temperature].set_index('Term')
        data = {'dh1': fields[index]}
        for term, stem in zip(terms, ('dh_x', 'dh_y', 'dh_z', 'dh_reconstructed')):
            data[stem] = panel.loc[term, ['CX_m_s', 'CY_m_s']].to_numpy(float)
        np.testing.assert_allclose(data['dh_reconstructed'], data['dh_x'] + data['dh_y'] + data['dh_z'],
                                   rtol=1e-6, atol=1e-6)
        results[source_sst] = data
    return results, scale


def color_scale(results, field):
    finite = np.concatenate([
        np.abs(results[sst][field][np.isfinite(results[sst][field])])
        for sst in SOURCE_SSTS
    ])
    if finite.size == 0:
        raise ValueError(f"No finite {field} values")
    upper = max(1, int(np.ceil(np.max(finite))))
    tick_step = max(1, int(np.ceil(upper / 4)))
    upper = tick_step * int(np.ceil(upper / tick_step))
    levels = np.linspace(-upper, upper, 17)
    ticks = np.arange(-upper, upper + 1, tick_step)
    return levels, ticks, TwoSlopeNorm(vmin=-upper, vcenter=0, vmax=upper)

def draw_arrows(ax, data, km_per_ms):
    for stem in ("dh_x", "dh_y", "dh_z", "dh_reconstructed"):
        vector = data[stem]
        color_key = "dh" if stem == "dh_reconstructed" else stem
        endpoint = tuple(km_per_ms * vector)
        length = float(np.hypot(*endpoint))
        ax.add_patch(FancyArrowPatch(
            (0, 0), endpoint, arrowstyle="-|>",
            mutation_scale=9 if length < 8 else 13,
            linewidth=3.0 if stem == "dh_reconstructed" else 2.3,
            color=VECTOR_COLORS[color_key],
            shrinkA=0, shrinkB=0, zorder=7,
        ))
    ax.plot(0, 0, "ko", ms=4.5, zorder=9)

def draw_vector_key(ax, km_per_ms):
    """Key arrows depict 1 m/s using the same scale as the map arrows."""
    x0 = .71
    dx = km_per_ms / (AXIS_KM[-1] - AXIS_KM[0])
    x1 = x0 + dx
    ax.text((x0 + x1) / 2, .31, r"1 m s$^{-1}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    for stem, y in zip(("dh", "dh_x", "dh_y", "dh_z"),
                       (.25, .19, .13, .07)):
        ax.add_patch(FancyArrowPatch(
            (x0, y), (x1, y), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=11, linewidth=2.2,
            color=VECTOR_COLORS[stem], shrinkA=0, shrinkB=0, zorder=10,
        ))
        ax.text(x1 + .022, y, VECTOR_LABELS[stem],
                transform=ax.transAxes, va="center", fontsize=10)

def decorate(ax, col=0):
    ax.set(xlim=(-120, 120), ylim=(-120, 120),
           xticks=(-120, -60, 0, 60, 120),
           yticks=(-120, -60, 0, 60, 120))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=3.5, width=.8, labelsize=10)
    if col:
        ax.tick_params(labelleft=False)

def make_figure(results, scale):
    fig = plt.figure(figsize=(11.2, 4.8), facecolor="white")
    lefts = (.08, .50)
    bottom = .15
    panel_size_in = 3.942
    width = panel_size_in / fig.get_figwidth()
    height = panel_size_in / fig.get_figheight()
    levels_dh, ticks_dh, norm_dh = color_scale(results, "dh1")
    fills, axes, panel_labels = {}, {}, []
    for col, sst in enumerate(SOURCE_SSTS):
        panel = "AB"[col]
        ax = fig.add_axes((lefts[col], bottom, width, height))
        axes[panel] = ax
        fills[panel] = ax.contourf(
            X_KM, Y_KM, results[sst]["dh1"],
            levels=levels_dh, norm=norm_dh, cmap=CMAP, extend="both",
        )
        decorate(ax, col=col)
        if col == 0:
            ax.set_ylabel("Meridional distance to TC center (km)", labelpad=8)
        ax.set_xlabel("Zonal distance to TC center (km)", labelpad=6)
        label = ax.text(-.06, 1.02, panel.lower(), transform=ax.transAxes,
                fontsize=16, fontweight="bold",
                ha="left", va="bottom", clip_on=False)
        panel_labels.append(label)
        draw_arrows(ax, results[sst], scale)
        if col == 1:
            draw_vector_key(ax, scale)
    cax = fig.add_axes((.88, bottom, .018, height))
    bar = fig.colorbar(fills["B"], cax=cax, ticks=ticks_dh)
    bar.set_label(f"DH contribution to PVT ({PV_TENDENCY_UNIT})", fontsize=10)
    bar.ax.tick_params(labelsize=9, length=3)
    return fig, axes, cax, panel_labels

def save_panel(fig, axis, color_axis, panel_labels, right_axis, scale, path):
    """Reuse a composite panel at its original physical scale with a nearby colorbar."""
    visibility = [(item, item.get_visible()) for item in [*fig.axes, *panel_labels]]
    color_position = color_axis.get_position(original=True).frozen()
    gap = color_position.x0 - right_axis.get_position().x1
    xlabel = axis.get_ylabel()
    ticks = axis.yaxis.get_major_ticks()
    tick_visibility = [tick.label1.get_visible() for tick in ticks]
    added_key_artists = []
    try:
        for item, _ in visibility:
            item.set_visible(item is axis or item is color_axis)
        axis.set_ylabel('Meridional distance to TC center (km)', labelpad=8)
        axis.tick_params(labelleft=True)
        box = axis.get_position()
        color_axis.set_position([box.x1 + gap, color_position.y0,
                                 color_position.width, color_position.height])
        if axis is not right_axis:
            before = set([*axis.texts, *axis.patches])
            draw_vector_key(axis, scale)
            added_key_artists = [item for item in [*axis.texts, *axis.patches] if item not in before]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = Bbox.union([axis.get_tightbbox(renderer), color_axis.get_tightbbox(renderer)])
        bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(.05)
        fig.savefig(path, dpi=DPI, facecolor='white', bbox_inches=bounds, pad_inches=0)
    finally:
        for item in added_key_artists:
            item.remove()
        axis.set_ylabel(xlabel)
        for tick, visible in zip(ticks, tick_visibility):
            tick.label1.set_visible(visible)
        color_axis.set_position(color_position)
        for item, visible in visibility:
            item.set_visible(visible)


def save_figures(results, scale, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, color_axis, labels = make_figure(results, scale)
    try:
        fig.savefig(output_dir / 'FigS9_DH_EXP_minus_CTRL_2Panel.png', dpi=DPI,
                    facecolor='white', bbox_inches='tight', pad_inches=.05)
        for panel, temperature in (('A', 300), ('B', 302)):
            save_panel(fig, axes[panel], color_axis, labels, axes['B'], scale,
                       output_dir / f'FigS9{panel.lower()}_DH_{temperature}K_EXP_minus_CTRL.png')
    finally:
        plt.close(fig)
    print(f'Saved two DH panels and one composite PNG: {output_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=resolve_project_path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=resolve_project_path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(f'Plotting data: {args.data_dir}')
    save_figures(*load_plot_data(args.data_dir), output_dir=args.output_dir)


if __name__ == '__main__':
    main()
