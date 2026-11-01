#!/usr/bin/env python3
import argparse
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_ROOT = PROJECT_DIR / 'Database'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Results' / '002_M2_SST'
SST_STYLES = {'OISST': ('#244A73', 'o'), 'ERSST': ('#D69E79', 'o')}
SPATIAL_OCEAN_MASK_ALPHA = 0.93
SST_PROFILE_EDGES_KM = np.arange(0.0, 801.0, 200.0)
SST_EXCLUSION_POLYGON = np.asarray([
    (-26.0, 40.0), (60.0, 40.0), (40.0, 10.0), (23.0, -32.0),
    (23.0, -40.0), (-105.0, -40.0), (-105.0, 7.0),
    (-25.0, 7.0), (-26.0, 40.0),
], dtype=float)
SENSITIVITY_LABELS = (
    'Original (1980–2024)', 'ERSST-removed',
    'Original (1982–2024)', 'OISST-removed',
)
CMIP6_EXPERIMENT_COLORS = {
    'AER': '#DDAA9E', 'GHG': '#456D9F', 'NAT': '#88ADC1', 'HIST': '#9B5F5B',
}
CMIP6_EXPERIMENT_MARKERS = {'AER': 'o', 'GHG': 's', 'NAT': '^', 'HIST': None}
FIGURE_DPI = 600
FONT_SIZE = 9.0
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
})


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')


def load_spatial_data(path):
    """Load the native-grid map without interpolation or recomputation."""
    require_file(path)
    fields = ('sst_trend', 'MK_P', 'distance_to_coast')
    with xr.open_dataset(path, engine='h5netcdf') as dataset:
        missing = set(fields) - set(dataset.data_vars)
        if missing:
            raise ValueError(f'{path.name}: missing variables {sorted(missing)}')
        data = dataset[list(fields)].load()
    for name in ('lat', 'lon'):
        if name not in data.coords or data[name].ndim != 1:
            raise ValueError(f'{path.name}: expected one-dimensional {name}')
        coordinate = data[name].values
        if not np.isfinite(coordinate).all() or not (np.diff(coordinate) > 0).all():
            raise ValueError(f'{path.name}: {name} must be finite and increasing')
    for name in fields:
        if set(data[name].dims) != {'lat', 'lon'}:
            raise ValueError(f'{path.name}: {name} must have lat/lon dimensions')
    finite_p = data.MK_P.values[np.isfinite(data.MK_P.values)]
    if np.isinf(data.MK_P.values).any() or ((finite_p < 0) | (finite_p > 1)).any():
        raise ValueError(f'{path.name}: MK_P values must be within [0, 1] or NaN')
    trend = data.sst_trend.transpose('lat', 'lon')
    pvalue = data.MK_P.transpose('lat', 'lon')
    if bool((np.isfinite(trend) & ~np.isfinite(pvalue)).any()):
        raise ValueError(f'{path.name}: a finite SST trend is missing its MK P value')
    significant = (pvalue < .05) & np.isfinite(trend)
    distance = data.distance_to_coast.transpose('lat', 'lon')
    if bool((significant & ~np.isfinite(trend)).any()):
        raise ValueError(f'{path.name}: a significant cell has a missing trend')
    return trend, significant, distance


def load_table(path, keys, expected_keys, numeric_columns, *, significance=False):
    """Validate panel categories, numbers and numerical MK P values."""
    require_file(path)
    table = pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')
    required = set(keys) | set(numeric_columns)
    if significance:
        required.add('MK_P')
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    if table.duplicated(list(keys)).any():
        raise ValueError(f'{path.name}: duplicate plotting records')
    actual_keys = set(table[list(keys)].itertuples(index=False, name=None))
    if actual_keys != set(expected_keys):
        raise ValueError(f'{path.name}: unexpected or missing categories/distance bins')
    for name in numeric_columns:
        table[name] = pd.to_numeric(table[name], errors='raise')
        if not np.isfinite(table[name]).all():
            raise ValueError(f'{path.name}: {name} must be finite')
        if 'CI95' in name and (table[name] < 0).any():
            raise ValueError(f'{path.name}: CI95 half-width must be nonnegative')
    if significance:
        pvalues = pd.to_numeric(table.MK_P, errors='raise')
        if not np.isfinite(pvalues).all() or not pvalues.between(0, 1).all():
            raise ValueError(f'{path.name}: MK_P values must be finite and within [0, 1]')
        table['MK_P'] = pvalues.astype(float)
    return table


def load_plot_data(data_dir):
    spatial = load_spatial_data(data_dir / 'FigM2A_SST_spatial.nc')
    centers = (100.0, 300.0, 500.0, 700.0)
    profiles = load_table(
        data_dir / 'FigM2BCD_SST_profiles.csv',
        ('Region', 'Source', 'Distance_Center_km'),
        [(region, source, center) for region in ('5-20N', '20-40N', '5-40S')
         for source in ('OISST', 'ERSST') for center in centers],
        ('Distance_Center_km', 'OLS_Trend_K_per_Decade', 'OLS_CI95_K_per_Decade'),
        significance=True,
    )
    sensitivity = load_table(
        data_dir / 'FigM2E_CLTC_SST_sensitivity.csv',
        ('Condition', 'Metric'),
        [(condition, metric) for condition in SENSITIVITY_LABELS
         for metric in ('CLTC count', 'Duration fraction')],
        ('Trend', 'CI95'), significance=True,
    )
    attribution = load_table(
        data_dir / 'FigM2F_CMIP6_SST_attribution.csv',
        ('Forcing', 'Distance_Center_km'),
        [(forcing, center) for forcing in ('AER', 'GHG', 'NAT', 'HIST')
         for center in centers],
        ('Distance_Center_km', 'Median_K_per_Decade'),
    )
    return spatial, profiles, sensitivity, attribution


def sst_exclusion_mask(distance):
    lon2d, lat2d = np.meshgrid(distance.lon.values, distance.lat.values)
    locations = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    inside = MplPath(SST_EXCLUSION_POLYGON, closed=True).contains_points(
        locations, radius=1e-10).reshape(lon2d.shape)
    return xr.DataArray(inside | (np.abs(lat2d) <= 5.0),
                        coords={'lat': distance.lat, 'lon': distance.lon},
                        dims=('lat', 'lon'))


def draw_spatial_sst_trend(fig,axis,cax,trend,significant,distance,coastline_path):
    def cyclic(array):
        # Order plotting columns eastward from Greenwich before closing the seam.
        array = array.assign_coords(lon=np.mod(array.lon.values, 360.0)).sortby('lon')
        values,lon=add_cyclic_point(np.asarray(array.values,float),
                                    coord=np.asarray(array.lon.values,float))
        return values,lon,np.asarray(array.lat.values,float)
    data,lon,lat=cyclic(trend)
    levels=np.linspace(0,.4,25)
    colors = [
        '#3B79AE','#5F9FC0','#8ABBD0','#B7D3E3','#E5EFF7',
        '#FFFFFF',
        '#FBEDE7','#F6C9C0','#EE9E97','#DA848D','#C07278',
    ]
    color_values=np.asarray(
        np.linspace(0,.4,len(colors)),
        dtype=float,
    )
    color_positions=color_values/.4
    # Matplotlib requires exact 0 and 1 endpoints for mapped colors.
    color_positions[0]=0.0
    color_positions[-1]=1.0
    cmap=LinearSegmentedColormap.from_list(
        'M2_SST_reference_red_blue',
        list(zip(color_positions,colors)),N=256,
    )
    norm=Normalize(vmin=0,vmax=.4)
    filled=axis.contourf(lon,lat,data,levels=levels,cmap=cmap,norm=norm,
                          extend='both',transform=ccrs.PlateCarree(),zorder=2)
    # Use exactly the native-grid selection used for the B-D distance profiles.
    # Keep SST colours visible only in the unexcluded 0-800 km offshore domain.
    distance = distance.transpose('lat', 'lon')
    analysis_domain = (
        np.isfinite(distance) & (distance >= SST_PROFILE_EDGES_KM[0])
        & (distance <= SST_PROFILE_EDGES_KM[-1])
        & ~sst_exclusion_mask(distance)
    )
    # Sample on a periodic longitude grid, without duplicate -180/+180 points.
    sig_native = np.asarray(
        (significant.transpose('lat', 'lon')
         & np.isfinite(distance) & (distance >= 0)).values, bool
    )
    lon_native = np.asarray(significant.lon.values, float)
    lat_native = np.asarray(significant.lat.values, float)
    lon_sample = np.arange(0.0, 360.0, 2.0)
    lat_sample = np.arange(-40.0, 40.0 + 0.001, 2.0)
    angular_difference = np.abs(
        (lon_native[None, :] - lon_sample[:, None] + 180.0) % 360.0 - 180.0
    )
    lon_index = np.argmin(angular_difference, axis=1)
    # Also avoid repeated native cells if a coarser grid is supplied.
    _, first = np.unique(lon_index, return_index=True)
    lon_index = lon_index[np.sort(first)]
    lat_index = np.array([
        int(np.argmin(np.abs(lat_native - value))) for value in lat_sample
    ])
    lat_index = np.unique(lat_index)
    sampled_sig = sig_native[np.ix_(lat_index, lon_index)]
    sampled_lon, sampled_lat = np.meshgrid(lon_native[lon_index], lat_native[lat_index])
    axis.scatter(sampled_lon[sampled_sig], sampled_lat[sampled_sig], s=.80, marker='o',
                 c='black', linewidths=0, transform=ccrs.PlateCarree(),
                 zorder=5, rasterized=True)
    # Use the signed coast-distance grid for a land mask consistent with analysis.
    land_field = xr.where(np.isfinite(distance) & (distance < 0), 1.0, 0.0)
    land_values, land_lon, land_lat = cyclic(land_field)
    axis.contourf(land_lon, land_lat, land_values,
                  levels=[0.5, 1.5], colors=['#D2D2D2'],
                  transform=ccrs.PlateCarree(), zorder=3, antialiased=False)
    # De-emphasize all ocean cells outside the profile analysis domain.
    # Non-finite distances are also masked; land is kept grey, not overlaid.
    is_land = np.isfinite(distance) & (distance < 0)
    # Explicitly include the entire equatorial exclusion band, including the
    # Indonesian seas, regardless of their small distance to nearby islands.
    equatorial_band = (np.abs(distance.lat) <= 5.0).broadcast_like(distance)
    ocean_outside = (~analysis_domain | equatorial_band) & ~is_land
    mask_values, mask_lon, mask_lat = cyclic(ocean_outside.astype(float))
    # Draw whole grid cells rather than interpolated binary-mask contours.
    # Unmasked cells remain transparent; all excluded ocean cells get one overlay.
    mask_values = np.ma.masked_where(mask_values < 0.5, mask_values)
    axis.pcolormesh(
        mask_lon, mask_lat, mask_values, shading='nearest',
        cmap=mpl.colors.ListedColormap(['white']), vmin=0.0, vmax=1.0,
        alpha=SPATIAL_OCEAN_MASK_ALPHA, transform=ccrs.PlateCarree(),
        zorder=7, edgecolors='none', antialiased=False, rasterized=True,
    )
    # Outline the selected analysis region independently of the display overlay.
    boundary_values, boundary_lon, boundary_lat = cyclic(analysis_domain.astype(float))
    axis.contour(
        boundary_lon, boundary_lat, boundary_values,
        levels=[0.5], colors='#444444', linewidths=0.35,
        linestyles=[(0, (3, 2))], transform=ccrs.PlateCarree(), zorder=6,
    )
    # Draw GSHHG coastlines below stippling and the ocean overlay.
    require_file(coastline_path)
    coastline_reader = shapereader.Reader(str(coastline_path))
    try:
        axis.add_geometries(
            list(coastline_reader.geometries()), crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='#555555', linewidth=0.4, zorder=4,
        )
    finally:
        coastline_reader.close()
    transform=ccrs.PlateCarree()
    # On a 180-degree-centred map, x=-180..180 corresponds to longitude 0..360.
    axis.set_xlim(-180, 180)
    axis.set_ylim(-40, 40)
    axis.set_xticks([-180, -120, -60, 0, 60, 120])
    axis.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W'])
    axis.set_yticks(np.arange(-40,41,20),crs=transform)
    axis.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    axis.tick_params(direction='out',length=3,width=.75,pad=2,labelsize=FONT_SIZE)
    for spine in axis.spines.values():
        spine.set_linewidth(.75);spine.set_color('#222222')
        spine.set_zorder(10)
    # Heading first, then a longer horizontal colorbar above panel A.
    bar_box=cax.get_position()
    fig.text(bar_box.x0-.016,bar_box.y0+bar_box.height/2,
             'Trend in SST (K decade$^{-1}$)',
             ha='right',va='center',fontsize=FONT_SIZE)
    bar=fig.colorbar(filled,cax=cax,orientation='horizontal',
                     ticks=(0,.1,.2,.3,.4),extend='both')
    bar.ax.xaxis.set_ticks_position('bottom')
    bar.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _position: '0' if np.isclose(value, 0)
        else f'{value:g}'.replace('-', '−')
    ))
    bar.ax.tick_params(labelsize=FONT_SIZE,width=.7,length=2.5,pad=2)
    bar.outline.set_linewidth(.7)


def draw_sst_profile(axis, table, region_name, ylabel=False, legend=False,
                     *, y_limits, y_ticks):
    for source in ('OISST', 'ERSST'):
        sample = table[(table.Region == region_name) & (table.Source == source)]
        sample = sample.sort_values('Distance_Center_km')
        x = sample.Distance_Center_km.to_numpy(float)
        y = sample.OLS_Trend_K_per_Decade.to_numpy(float)
        ci = sample.OLS_CI95_K_per_Decade.to_numpy(float)
        sig = sample.MK_P.to_numpy(float) < .05
        color, marker = SST_STYLES[source]
        axis.fill_between(x, y-ci, y+ci, color=color, alpha=.13, linewidth=0)
        axis.plot(x, y, color=color, linewidth=1.25, zorder=3)
        axis.scatter(x[sig], y[sig], s=11, marker=marker,
                     facecolor=color, edgecolor=color, linewidth=.4, zorder=4)
        axis.scatter(x[~sig], y[~sig], s=17, marker=marker,
                     facecolor='white', edgecolor=color, linewidth=.8, zorder=4)
    axis.set_xlim(0, 800);axis.set_xticks((0, 200, 400, 600, 800))
    axis.set_ylim(*y_limits);axis.set_yticks(y_ticks)
    axis.set_xlabel('Distance to coast (km)',fontsize=FONT_SIZE)
    if ylabel:axis.set_ylabel('Trend in SST (K decade$^{-1}$)',fontsize=FONT_SIZE)
    else:axis.tick_params(axis='y',labelleft=False)
    axis.tick_params(direction='out',length=3,width=.7,pad=2,labelsize=FONT_SIZE)
    for spine in axis.spines.values():spine.set_linewidth(.7);spine.set_color('#333333')
    if legend:
        handles = [Line2D([0],[0],color=SST_STYLES[source][0],marker=SST_STYLES[source][1],
                          markersize=3,linewidth=1.25,label=source) for source in ('OISST','ERSST')]
        axis.legend(handles=handles,loc='upper right',frameon=False,fontsize=FONT_SIZE,
                    handlelength=1.7,borderaxespad=.3)


def draw_cltc_sst_sensitivity(axis, table):
    count_color, fraction_color, fraction_error = '#244A73', '#D69E79', '#C25D34'
    x = np.arange(4, dtype=float)
    width = .26
    counts = table.loc[table.Metric.eq('CLTC count')].set_index('Condition').loc[list(SENSITIVITY_LABELS)]
    fractions = table.loc[table.Metric.eq('Duration fraction')].set_index('Condition').loc[list(SENSITIVITY_LABELS)]
    count_v, count_ci = counts.Trend.to_numpy(float), counts.CI95.to_numpy(float)
    fraction_v, fraction_ci = fractions.Trend.to_numpy(float), fractions.CI95.to_numpy(float)
    right = axis.twinx()
    count_bars = axis.bar(x-width/2, count_v, width, color=count_color,
                          edgecolor='none', zorder=3)
    fraction_bars = right.bar(x+width/2, fraction_v, width, color=fraction_color,
                               edgecolor='none', zorder=3)
    axis.errorbar(x-width/2, count_v, yerr=count_ci, fmt='none',
                  ecolor='#333333', elinewidth=.8, capsize=2.4, zorder=4)
    right.errorbar(x+width/2, fraction_v, yerr=fraction_ci, fmt='none',
                   ecolor=fraction_error, elinewidth=.8, capsize=2.4, zorder=4)
    low = min(-1.0, float(np.min(count_v-count_ci))-.12,
              float(np.min(fraction_v-fraction_ci))/3-.12)
    high = max(3.0, float(np.max(count_v+count_ci))+.20,
               float(np.max(fraction_v+fraction_ci))/3+.20)
    low, high = np.floor(low*2)/2, np.ceil(high*2)/2
    axis.set_ylim(low, high); right.set_ylim(low*3, high*3)
    count_ticks = np.arange(np.ceil(low), np.floor(high)+1, 1)
    axis.set_yticks(count_ticks);right.set_yticks(count_ticks*3)
    axis.set_xlim(-.55, 3.55)
    axis.axhline(0, color='#666666', lw=.65, zorder=1)
    axis.set_xticks(x, ('Original\n(1980–2024)', 'ERSST-removed\n(1980–2024)',
                        'Original\n(1982–2024)', 'OISST-removed\n(1982–2024)'))
    axis.set_ylabel('Trend in CLTC count (decade$^{-1}$)',fontsize=FONT_SIZE)
    right.set_ylabel('Trend in duration fraction (% decade$^{-1}$)',color=fraction_error,fontsize=FONT_SIZE)
    right.tick_params(axis='y', colors=fraction_error)
    right.spines['right'].set_color(fraction_error)
    for ax in (axis, right):
        ax.tick_params(direction='out', length=3, width=.75, pad=2,
                       labelsize=FONT_SIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(.75)
    from matplotlib.patches import Patch
    axis.legend(handles=(Patch(facecolor=count_color, label='CLTC count'),
                         Patch(facecolor=fraction_color, label='Duration fraction')),
                loc='upper right', bbox_to_anchor=(1.0, .99), ncol=1,
                alignment='left', frameon=False, fontsize=FONT_SIZE,
                handlelength=1.3, labelspacing=.25)
    for bars, estimates, errors, flags, ax, color in (
        (count_bars, count_v, count_ci, counts.MK_P.to_numpy(float) < .05, axis, '#111111'),
        (fraction_bars, fraction_v, fraction_ci, fractions.MK_P.to_numpy(float) < .05,
         right, fraction_error),
    ):
        for bar, value, error, significant in zip(bars, estimates, errors, flags):
            if significant:
                ax.text(bar.get_x()+bar.get_width()/2,
                        value+error+.025*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                        '**', color=color, ha='center', va='bottom',
                        fontsize=FONT_SIZE, fontweight='bold')
    return right


def draw_cmip6_attribution(axis,table):
    for forcing in ('AER','GHG','NAT','HIST'):
        part=table.loc[table.Forcing.eq(forcing)].sort_values('Distance_Center_km')
        if len(part)!=4:raise ValueError(f'Missing four CMIP6 bins for {forcing}')
        axis.plot(part.Distance_Center_km.to_numpy(float),
                  part.Median_K_per_Decade.to_numpy(float),
                  color=CMIP6_EXPERIMENT_COLORS[forcing],
                  marker=CMIP6_EXPERIMENT_MARKERS[forcing],
                  markersize=3.4,markeredgewidth=.6,linewidth=1.15,
                  label=forcing,zorder=3)
    axis.axhline(0,color='#777777',linewidth=.55,zorder=1)
    axis.set_xlim(0,800);axis.set_ylim(-.05,.35)
    axis.set_xticks((0,200,400,600,800));axis.set_yticks((0,.1,.2,.3))
    axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _position: '0' if np.isclose(value, 0) else f'{value:.1f}'
    ))
    axis.set_xlabel('Distance to coast (km)',fontsize=FONT_SIZE)
    axis.set_ylabel('Trend in SST (K decade$^{-1}$)',fontsize=FONT_SIZE)
    axis.legend(loc='upper right',ncol=2,frameon=False,fontsize=FONT_SIZE,
                handlelength=1.55,columnspacing=.55,handletextpad=.35,
                borderaxespad=.25)
    axis.tick_params(direction='out',length=3,width=.75,pad=2,labelsize=FONT_SIZE)
    for spine in axis.spines.values():
        spine.set_visible(True);spine.set_linewidth(.75);spine.set_color('#222222')


def draw_figures(output_dir, coastline_path, spatial, sst_profiles,
                 cltc_sst_sensitivity, cmip6_attribution):
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_sst_trend, spatial_significance, spatial_distance = spatial
    trend = sst_profiles.OLS_Trend_K_per_Decade.to_numpy(float)
    ci = sst_profiles.OLS_CI95_K_per_Decade.to_numpy(float)
    profile_y_limits = (min(.08, float(np.min(trend-ci))-.015),
                        max(.35, float(np.max(trend+ci))+.015))
    profile_y_ticks = [t for t in (.1,.2,.3,.4,.5)
                       if profile_y_limits[0] < t < profile_y_limits[1]]
    profile_regions = ('5-20N', '20-40N', '5-40S')

    panel_a_file=output_dir/'FigM2a_SST_spatial.png'
    fig=plt.figure(figsize=(7.25,2.55),facecolor='white')
    map_width=.81;map_height=map_width*7.25/(4.5*2.55)
    map_top=.84;map_bottom=map_top-map_height
    axis=fig.add_axes([.08,map_bottom,map_width,map_height],projection=ccrs.PlateCarree(central_longitude=180))
    cax=fig.add_axes([.46,.90,.43,.023])
    draw_spatial_sst_trend(fig,axis,cax,spatial_sst_trend,spatial_significance,spatial_distance,coastline_path)
    fig.savefig(panel_a_file,dpi=FIGURE_DPI,facecolor='white',
                bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

    profile_annotations = {
        'B': 'NH (5°–20°N)',
        'C': 'NH (20°–40°N)',
        'D': 'SH (5°–40°S)',
    }

    for letter,region in zip('BCD',profile_regions):
        fig,axis=plt.subplots(figsize=(2.72,2.38),facecolor='white')
        fig.subplots_adjust(left=.205,right=.985,bottom=.225,top=.915)
        draw_sst_profile(axis,sst_profiles,region,ylabel=True,legend=(letter=='D'),
                         y_limits=profile_y_limits,y_ticks=profile_y_ticks)
        axis.text(.03,.96,profile_annotations[letter],transform=axis.transAxes,
                  ha='left',va='top',fontsize=FONT_SIZE)
        path=output_dir/f'FigM2{letter.lower()}_OISST_ERSST_{region}.png'
        fig.savefig(path,dpi=FIGURE_DPI,facecolor='white',
                    bbox_inches='tight',pad_inches=.035)
        plt.close(fig)
        print(f'Saved: {path}')

    panel_e_file=output_dir/'FigM2e_CLTC_SST_sensitivity.png'
    fig,axis=plt.subplots(figsize=(5.82,2.55),facecolor='white')
    fig.subplots_adjust(left=.14,right=.85,bottom=.25,top=.92)
    draw_cltc_sst_sensitivity(axis,cltc_sst_sensitivity)
    fig.savefig(panel_e_file,dpi=FIGURE_DPI,facecolor='white',
                bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

    panel_f_file=output_dir/'FigM2f_CMIP6_SST_attribution.png'
    fig,axis=plt.subplots(figsize=(2.72,2.38),facecolor='white')
    fig.subplots_adjust(left=.205,right=.985,bottom=.225,top=.915)
    draw_cmip6_attribution(axis,cmip6_attribution)
    fig.savefig(panel_f_file,dpi=FIGURE_DPI,facecolor='white',
                bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

    # Shared axes geometry, with all artwork drawn by Matplotlib.
    fig=plt.figure(figsize=(8.4,7.3),facecolor='white')
    left,right,top=.08,.94,.91
    map_height=(right-left)*8.4/(4.5*7.3)
    a_bottom=top-map_height
    axis_a=fig.add_axes([left,a_bottom,right-left,map_height],projection=ccrs.PlateCarree(central_longitude=180))
    cax=fig.add_axes([.49,.948,.45,.017])
    draw_spatial_sst_trend(fig,axis_a,cax,spatial_sst_trend,spatial_significance,spatial_distance,coastline_path)
    fig.text(.055,.935,'A',fontsize=10.5,fontweight='bold',va='bottom')
    middle_top=a_bottom-.068
    middle_bottom=middle_top-.235
    bottom_top=middle_bottom-.089
    bottom_bottom=.078
    columns=((.08,.24),(.39,.24),(.70,.24))
    for (letter,region),(x0,width) in zip(zip('BCD',profile_regions),columns):
        axis=fig.add_axes([x0,middle_bottom,width,middle_top-middle_bottom])
        draw_sst_profile(axis,sst_profiles,region,
                          ylabel=(letter=='B'),legend=(letter=='D'),
                          y_limits=profile_y_limits,y_ticks=profile_y_ticks)
        axis.text(.03,.96,profile_annotations[letter],transform=axis.transAxes,
                  ha='left',va='top',fontsize=FONT_SIZE)
        fig.text(x0-.023,middle_top+.010,letter,fontsize=10.5,
                 fontweight='bold',va='bottom')
    axis_e=fig.add_axes([.08,bottom_bottom,.49,bottom_top-bottom_bottom])
    axis_e_right=draw_cltc_sst_sensitivity(axis_e,cltc_sst_sensitivity)
    axis_e_right.yaxis.set_label_coords(1.05,.5)
    fig.text(.057,bottom_top+.010,'E',fontsize=10.5,fontweight='bold',va='bottom')
    axis_f=fig.add_axes([.70,bottom_bottom,.24,bottom_top-bottom_bottom])
    draw_cmip6_attribution(axis_f,cmip6_attribution)
    fig.text(.677,bottom_top+.010,'F',fontsize=10.5,fontweight='bold',va='bottom')
    composite_file=output_dir/'FigM2_composite.png'
    fig.savefig(composite_file,dpi=FIGURE_DPI,facecolor='white',
                bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    print(f'Saved: {composite_file}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-root', type=Path, default=DEFAULT_DATABASE_ROOT,
                        help='Database directory alongside Codes.')
    parser.add_argument('--data-dir', type=Path,
                        help='Four preprocessed files; default: Database/SST.')
    parser.add_argument('--coastline', type=Path,
                        help='GSHHG shapefile; default: Database/GSHHG/GSHHS_f_L1_1400.shp.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='PNG output directory; default: Results/002_M2_SST.')
    args = parser.parse_args(argv)
    database_root = args.database_root.expanduser().resolve()
    data_dir = (args.data_dir or database_root / 'SST').expanduser().resolve()
    coastline_path = (args.coastline or database_root / 'GSHHG/GSHHS_f_L1_1400.shp').expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require_file(coastline_path)
    print(f'Plotting data: {data_dir}')
    print(f'Results: {output_dir}')
    plot_data = load_plot_data(data_dir)
    draw_figures(output_dir, coastline_path, *plot_data)

if __name__ == '__main__':
    main()
