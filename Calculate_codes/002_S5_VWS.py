#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from scipy import stats

VARIABLE = 'VWS'
FIGURE = 'S5'
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database' / VARIABLE
PROFILE_COLUMNS = ['Region', 'Source', 'Distance_Center_km',
                   'OLS_Trend_per_Decade', 'OLS_CI95_per_Decade', 'MK_P']

def resolve_project_path(value):
    return Path(value).expanduser().resolve()

def open_netcdf(path):
    # h5netcdf also supports Windows Unicode paths in NetCDF4/HDF5 files.
    return xr.open_dataset(path, engine='h5netcdf', decode_times=True)

START_YEAR, END_YEAR = 1980, 2024
NH_MONTHS = (7, 8, 9, 10, 11)

SH_MONTHS = (12, 1, 2, 3, 4)

RH_LEVEL = 600.0

UPPER_LEVEL = 200.0

LOWER_LEVEL = 850.0

DISTANCE_EDGES = np.arange(-400.0, 801.0, 100.0)

REGIONS = (
    {"name": "5-20N", "label": "NH (5–20°N)", "south": 5.0,
      "north": 20.0, "months": NH_MONTHS},
    {"name": "20-40N", "label": "NH (20–40°N)", "south": 20.0,
      "north": 40.0, "months": NH_MONTHS},
    {"name": "5-40S", "label": "SH (5–40°S)", "south": -40.0,
      "north": -5.0, "months": SH_MONTHS},
)

EXCLUSION_POLYGON = np.asarray(
    [
        (-26, 40), (60, 40), (40, 10), (23, -32), (23, -40),
        (-105, -40), (-105, 7), (-25, 7), (-26, 40),
    ],
    dtype=float,
)

COORDINATE_CANDIDATES = {
    "time": ("time", "valid_time", "date"),
    "level": ("level", "lev", "pressure_level", "pressure", "plev", "isobaricInhPa"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude"),
}

COORDS = COORDINATE_CANDIDATES

def require(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')

def find_name(names, candidates, label, required=True):
    value = next((item for item in candidates if item in names), None)
    if value is None and required:
        raise KeyError(f'Cannot find {label}; available: {sorted(names)}')
    return value

def pick_variable(dataset, candidates):
    name = next((item for item in candidates if item in dataset.data_vars), None)
    if name:
        return name
    found = [name for name, item in dataset.data_vars.items() if item.ndim >= 3]
    if len(found) != 1:
        raise KeyError(f'Cannot identify data variable: {list(dataset.data_vars)}')
    return found[0]

def canonicalize(array, with_level=True):
    names = set(array.dims) | set(array.coords)
    rename = {}
    for key in ('time', 'lat', 'lon'):
        old = find_name(names, COORDS[key], key)
        if old != key:
            rename[old] = key
    old = find_name(names, COORDS['level'], 'pressure level', required=with_level)
    if old and old != 'level':
        rename[old] = 'level'
    array = array.rename(rename)
    allowed = {'time', 'lat', 'lon'} | ({'level'} if with_level else set())
    for dim in tuple(array.dims):
        if dim not in allowed:
            if array.sizes[dim] != 1:
                raise ValueError(f'Unexpected dimension {dim}={array.sizes[dim]}')
            array = array.isel({dim: 0}, drop=True)
    lon = (array.lon.astype(float) + 180) % 360 - 180
    return array.assign_coords(lon=lon).sortby('lon').sortby('lat').sortby('time')

def select_level(array, hpa):
    values = np.asarray(array.level.values, float)
    units = str(array.level.attrs.get('units', '')).lower()
    pa = units in {'pa', 'pascal', 'pascals'} or np.nanmax(abs(values)) > 2000
    target = hpa * 100 if pa else hpa
    selected = array.sel(level=target, method='nearest')
    actual = float(selected.level) / 100 if pa else float(selected.level)
    if abs(actual - hpa) > 1:
        raise ValueError(f'Requested {hpa:g} hPa; nearest is {actual:g} hPa')
    return selected.drop_vars('level', errors='ignore')

def open_distance(path, template):
    require(path)
    with open_netcdf(path) as ds:
        name = next((x for x in ('distance_to_coast', 'signed_distance_to_coast') if x in ds.data_vars), None)
        if name is None:
            raise KeyError(f'No signed distance variable: {path}')
        distance = ds[name].squeeze(drop=True)
        names = set(distance.dims) | set(distance.coords)
        rename = {find_name(names, COORDS['lat'], 'latitude'): 'lat', find_name(names, COORDS['lon'], 'longitude'): 'lon'}
        distance = distance.rename(rename)
        lon = (distance.lon.astype(float) + 180) % 360 - 180
        distance = distance.assign_coords(lon=lon).sortby('lon').sortby('lat')
        units = str(distance.attrs.get('units', '')).strip().lower()
        if units in {'m', 'meter', 'meters', 'metre', 'metres'}:
            distance = distance / 1000
        sign = str(distance.attrs.get('sign_convention', '')).lower()
        if any((x in sign for x in ('positive over land', 'land positive', 'land > 0'))):
            distance = -distance
        elif not any((x in sign for x in ('negative over land', 'land negative', 'land < 0'))):
            raise ValueError(f'Unknown distance sign convention: {path}')
        try:
            distance = distance.sel(lat=template.lat, lon=template.lon)
        except KeyError as exc:
            raise ValueError(f'Field/distance native grids differ: {path}') from exc
        np.testing.assert_allclose(distance.lat, template.lat, rtol=0, atol=1e-10)
        np.testing.assert_allclose(distance.lon, template.lon, rtol=0, atol=1e-10)
        return distance.load()

def exclusion_mask(distance):
    lon, lat = np.meshgrid(distance.lon.values, distance.lat.values)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    inside = MplPath(EXCLUSION_POLYGON, closed=True).contains_points(points, radius=1e-10).reshape(lon.shape)
    return xr.DataArray(inside | (np.abs(lat) <= 5), coords={'lat': distance.lat, 'lon': distance.lon}, dims=('lat', 'lon'))

def spherical_cell_area(lat, lon, earth_radius_m=6_371_000.0):
    """Spherical cell areas (m2) from rectilinear cell-centre coordinates."""
    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.asarray(lon.values, dtype=float)
    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError('Expected one-dimensional latitude and longitude')
    if lat_values.size < 2 or lon_values.size < 2:
        raise ValueError('At least two coordinates are needed in each direction')
    lat_steps = np.diff(lat_values)
    lon_steps = np.diff(lon_values)
    if (not np.all(np.isfinite(lat_values)) or
            not np.all(np.isfinite(lon_values)) or
            not (np.all(lat_steps > 0) or np.all(lat_steps < 0)) or
            not (np.all(lon_steps > 0) or np.all(lon_steps < 0))):
        raise ValueError('Latitude and longitude must be finite and monotonic')
    lat_edges = np.empty(lat_values.size + 1, dtype=float)
    lat_edges[1:-1] = (lat_values[:-1] + lat_values[1:]) / 2
    lat_edges[0] = lat_values[0] - lat_steps[0] / 2
    lat_edges[-1] = lat_values[-1] + lat_steps[-1] / 2
    lat_edges = np.clip(lat_edges, -90, 90)
    lon_edges = np.empty(lon_values.size + 1, dtype=float)
    lon_edges[1:-1] = (lon_values[:-1] + lon_values[1:]) / 2
    lon_edges[0] = lon_values[0] - lon_steps[0] / 2
    lon_edges[-1] = lon_values[-1] + lon_steps[-1] / 2
    area = (earth_radius_m ** 2 *
            np.abs(np.diff(np.sin(np.deg2rad(lat_edges))))[:, None] *
            np.abs(np.diff(np.deg2rad(lon_edges)))[None, :])
    if not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError('Invalid spherical cell areas')
    return xr.DataArray(area, coords={'lat': lat, 'lon': lon},
                        dims=(lat.dims[0], lon.dims[0]),
                        name='spherical_cell_area', attrs={'units': 'm2'})

def mann_kendall_p_columns(values):
    """Original two-sided MK P for time x series, including ties and continuity.

    Missing observations are skipped independently in each series. A normal
    survival function preserves tiny P values without rounding them to zero.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError('MK input must have dimensions (time, series)')
    valid = np.isfinite(values)
    n = valid.sum(axis=0).astype(float)
    score = np.zeros(values.shape[1], dtype=float)
    for i in range(values.shape[0]-1):
        difference = values[i+1:] - values[i]
        score += np.where(np.isfinite(difference), np.sign(difference), 0.).sum(axis=0)
    ordered = np.sort(np.where(valid, values, np.inf), axis=0)
    run = np.ones(values.shape[1], dtype=float)
    ties = np.zeros(values.shape[1], dtype=float)
    for i in range(1, values.shape[0]):
        same = np.isfinite(ordered[i]) & (ordered[i] == ordered[i-1])
        end = ~same & np.isfinite(ordered[i-1])
        ties += np.where(end, run*(run-1.)*(2.*run+5.), 0.)
        run = np.where(same, run+1., 1.)
    if values.shape[0]:
        ties += np.where(np.isfinite(ordered[-1]), run*(run-1.)*(2.*run+5.), 0.)
    variance = (n*(n-1.)*(2.*n+5.) - ties)/18.
    z = np.divide(score-np.sign(score), np.sqrt(np.maximum(variance, 0.)),
                  out=np.zeros_like(score), where=variance > 0.)
    p = 2.*stats.norm.sf(np.abs(z))
    p[n < 3] = np.nan
    return p

def mann_kendall_p(values):
    return float(mann_kendall_p_columns(np.asarray(values, dtype=float).reshape(-1, 1))[0])

def spatial_trend_and_p(smoothed):
    """Calculate OLS slopes and original MK P in bounded column batches."""
    years = np.asarray(smoothed.year.values, dtype=float)
    values = np.asarray(smoothed.transpose('year', 'lat', 'lon').values)
    shape = values.shape[1:]
    values = values.reshape(len(years), -1)
    slopes = np.full(values.shape[1], np.nan)
    pvalues = np.full(values.shape[1], np.nan)
    for start in range(0, values.shape[1], 8192):
        stop = min(start+8192, values.shape[1])
        y = values[:, start:stop].astype(float)
        valid = np.isfinite(y)
        n = valid.sum(axis=0)
        xm = np.divide(np.where(valid, years[:, None], 0.).sum(axis=0), n,
                       out=np.zeros(len(n)), where=n > 0)
        ym = np.divide(np.where(valid, y, 0.).sum(axis=0), n,
                       out=np.zeros(len(n)), where=n > 0)
        dx = np.where(valid, years[:, None]-xm, 0.)
        dy = np.where(valid, y-ym, 0.)
        slopes[start:stop] = np.divide((dx*dy).sum(axis=0), (dx*dx).sum(axis=0),
                                      out=np.full(len(n), np.nan), where=n >= 3)*10.
        pvalues[start:stop] = mann_kendall_p_columns(y)
    coords = {'lat': smoothed.lat, 'lon': smoothed.lon}
    return (xr.DataArray(slopes.reshape(shape), dims=('lat', 'lon'), coords=coords),
            xr.DataArray(pvalues.reshape(shape), dims=('lat', 'lon'), coords=coords))


def validate_months(field):
    """Select by coordinate indices only; where() would eagerly load raw data."""
    years = np.asarray(field.time.dt.year.values)
    indices = np.flatnonzero((years >= START_YEAR) & (years <= END_YEAR))
    field = field.isel(time=indices).sel(lat=slice(-40.0, 40.0))
    years = np.asarray(field.time.dt.year.values)
    calendar_months = np.asarray(field.time.dt.month.values)
    for months in (NH_MONTHS, SH_MONTHS):
        counts = pd.Series(years[np.isin(calendar_months, months)]).value_counts()
        bad = {year: int(counts.get(year, 0)) for year in range(START_YEAR, END_YEAR+1)
               if int(counts.get(year, 0)) != len(months)}
        if bad:
            raise ValueError(f'Incomplete months {months}: {bad}')
    return field


def rh_scale_factor(array):
    """Determine units without multiplying/loading the full monthly record."""
    units = str(array.attrs.get('units', '')).strip().lower()
    if units in {'1', 'fraction', 'dimensionless', '0-1'}:
        return 100
    if units not in {'%', 'percent', 'percentage'}:
        sample = array.isel(time=0,
                            lat=slice(None, None, max(1, array.sizes['lat']//20)),
                            lon=slice(None, None, max(1, array.sizes['lon']//20)))
        if float(np.nanmedian(sample.values)) <= 1.5:
            return 100
    return 1


def input_paths(database_root, product):
    directory = database_root / 'ENVIRONMENT' / product
    if product == 'ERA5':
        names = ('relative_humidity_1980-2024.nc',
                 'u_component_of_wind_1980-2024.nc', 'v_component_of_wind_1980-2024.nc')
    else:
        names = ('MERRA2_rh_1980-2024.nc', 'MERRA2_wind_u_1980-2024.nc',
                 'MERRA2_wind_v_1980-2024.nc')
    return dict(zip(('rh', 'u', 'v'), (directory / name for name in names)))


def load_product(database_root, product):
    """Preserve seasonal means with bounded year-by-year, level-selective reads."""
    files = input_paths(database_root, product)
    with ExitStack() as stack:
        def read_field(key, candidates):
            require(files[key])
            ds = stack.enter_context(open_netcdf(files[key]))
            return canonicalize(ds[pick_variable(ds, candidates)])
        if VARIABLE == 'RH':
            field = read_field('rh', ('r', 'rh', 'RH', 'relative_humidity'))
            field = select_level(field, RH_LEVEL)
            rh_factor = rh_scale_factor(field)
            fields = [field]
        else:
            u = read_field('u', ('u', 'U', 'ua', 'u_component_of_wind'))
            v = read_field('v', ('v', 'V', 'va', 'v_component_of_wind'))
            fields = list(xr.align(select_level(u, UPPER_LEVEL), select_level(u, LOWER_LEVEL),
                                   select_level(v, UPPER_LEVEL), select_level(v, LOWER_LEVEL), join='exact'))
        fields = [validate_months(item) for item in fields]
        template = fields[0]
        years = np.arange(START_YEAR, END_YEAR + 1)
        storage = {months: np.empty((len(years), template.sizes['lat'], template.sizes['lon']),
                                   dtype=np.float32) for months in (NH_MONTHS, SH_MONTHS)}
        for i, year in enumerate(years):
            for months in (NH_MONTHS, SH_MONTHS):
                index = np.flatnonzero((template.time.dt.year.values == year)
                                       & np.isin(template.time.dt.month.values, months))
                pieces = [np.asarray(item.isel(time=index).transpose('time', 'lat', 'lon').values)
                          for item in fields]
                values = pieces[0]*rh_factor if VARIABLE == 'RH' else np.hypot(pieces[0] - pieces[1], pieces[2] - pieces[3])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
                    storage[months][i] = np.nanmean(values, axis=0)
            if i % 10 == 0 or year == END_YEAR:
                print(f'{product} {VARIABLE}: seasonal means through {year}', flush=True)
        return {months: xr.DataArray(values, dims=('year', 'lat', 'lon'),
                                     coords={'year': years, 'lat': template.lat, 'lon': template.lon})
                for months, values in storage.items()}


def calculate_profiles(product, cache, distance):
    excluded = exclusion_mask(distance)
    area = spherical_cell_area(distance.lat, distance.lon)
    rows = []
    for region in REGIONS:
        annual, dist = xr.align(cache[region['months']], distance, join='exact')
        base = ((dist.lat >= region['south']) & (dist.lat <= region['north'])).broadcast_like(dist)
        base = base & ~excluded & np.isfinite(dist)
        years = np.asarray(annual.year, float)
        for index, (left, right) in enumerate(zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])):
            in_bin = (dist >= left) & (dist <= right if index == len(DISTANCE_EDGES)-2 else dist < right)
            weights = area.where(base & in_bin)
            # Reduce one year at a time instead of allocating multiple full
            # 45-year float64 weighted fields. Missing-cell weights are removed
            # independently in each year, as in the original xarray calculation.
            weight_values = np.asarray(weights.transpose('lat', 'lon').values, float)
            values = []
            for year in annual.year.values:
                field = np.asarray(annual.sel(year=year).transpose('lat', 'lon').values)
                numerator = np.nansum(field * weight_values, dtype=np.float64)
                denominator = np.nansum(np.where(np.isfinite(field), weight_values, np.nan), dtype=np.float64)
                values.append(numerator / denominator if denominator > 0 else np.nan)
            values = np.asarray(values, float)
            if not np.isfinite(values).all() or len(values) != END_YEAR - START_YEAR + 1:
                raise ValueError(f'Incomplete series: {product} {region["name"]} {left}..{right}')
            fit = stats.linregress(years, values)
            rows.append({'Region': region['name'], 'Source': product,
                         'Distance_Center_km': (left+right)/2,
                         'OLS_Trend_per_Decade': fit.slope*10,
                         'OLS_CI95_per_Decade': stats.t.ppf(.975, len(values)-2)*fit.stderr*10,
                         'MK_P': mann_kendall_p(values)})
    return pd.DataFrame(rows)


def calculate_map(cache):
    southern = cache[SH_MONTHS].isel(lat=np.flatnonzero(cache[SH_MONTHS].lat.values < 0))
    northern = cache[NH_MONTHS].isel(lat=np.flatnonzero(cache[NH_MONTHS].lat.values >= 0))
    annual = xr.concat((southern, northern), dim='lat').sortby('lat')
    smoothed = annual.rolling(year=5, center=True, min_periods=5).mean().dropna('year', how='all')
    return spatial_trend_and_p(smoothed)


def validate_profiles(table):
    if list(table.columns) != PROFILE_COLUMNS or len(table) != 72:
        raise ValueError(f'Expected 72 profile rows with columns {PROFILE_COLUMNS}')
    expected = [(region['name'], source, center) for source in ('ERA5', 'MERRA2')
                for region in REGIONS for center in (DISTANCE_EDGES[:-1]+DISTANCE_EDGES[1:])/2]
    if list(table[['Region', 'Source', 'Distance_Center_km']].itertuples(index=False, name=None)) != expected:
        raise ValueError('Unexpected profile row keys or ordering')
    numbers = table[PROFILE_COLUMNS[2:]].to_numpy(float)
    if not np.isfinite(numbers).all():
        raise ValueError('Profile data contain missing or nonfinite values')
    if (table.OLS_CI95_per_Decade < 0).any() or not table.MK_P.between(0, 1).all():
        raise ValueError('Invalid profile confidence interval or MK_P')


def assemble_outputs(profiles, slope, pvalue, distance):
    """Export newly calculated estimates; no previously calculated results are read."""
    table = profiles.loc[:, PROFILE_COLUMNS].copy()
    validate_profiles(table)
    slope, pvalue, distance = xr.align(slope, pvalue, distance, join='exact')
    spatial = xr.Dataset({'trend': slope.astype('float32'), 'MK_P': pvalue.astype('float64'),
                          'distance_to_coast': distance.astype('float32')})
    spatial.trend.attrs = {'units': '% decade^-1' if VARIABLE == 'RH' else 'm s^-1 decade^-1'}
    spatial.MK_P.attrs = {'long_name': 'Original two-sided Mann-Kendall P value',
                          'method': 'normal approximation with ties and continuity correction',
                          'valid_min': 0.0, 'valid_max': 1.0}
    spatial.distance_to_coast.attrs = {'units': 'km', 'sign_convention': 'negative over land; positive over ocean'}
    spatial.attrs = {'variable': VARIABLE, 'dataset': 'ERA5', 'period': '1980-2024',
                     'temporal_smoothing': 'centered 5-year running mean; five complete years',
                     'spatial_smoothing': 'none', 'trend_method': 'OLS slope per decade',
                     'significance_method': 'original two-sided Mann-Kendall with ties and continuity correction'}
    return spatial, table


def write_outputs(output_dir, spatial, table):
    validate_profiles(table)
    valid = spatial.MK_P.values[np.isfinite(spatial.MK_P.values)]
    if not len(valid) or np.any((valid < 0) | (valid > 1)):
        raise ValueError('Invalid spatial MK_P')
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / f'Fig{FIGURE}A_{VARIABLE}_spatial.nc'
    csv_path = output_dir / f'Fig{FIGURE}BCD_{VARIABLE}_profiles.csv'
    # Validate in temporary files before replacing prepared data.
    with tempfile.TemporaryDirectory(prefix=f'.{FIGURE}_', dir=output_dir) as tmp:
        nc_temp, csv_temp = Path(tmp)/nc_path.name, Path(tmp)/csv_path.name
        spatial.to_netcdf(nc_temp, engine='h5netcdf', encoding={
            name: {'zlib': True, 'complevel': 4} for name in spatial.data_vars})
        table.to_csv(csv_temp, index=False, float_format='%.17g')
        with open_netcdf(nc_temp) as check:
            xr.testing.assert_equal(check.load(), spatial)
        check_table = pd.read_csv(csv_temp, float_precision='round_trip', dtype={'Distance_Center_km': float})
        pd.testing.assert_frame_equal(table, check_table, check_exact=True)
        nc_temp.replace(nc_path)
        csv_temp.replace(csv_path)
    for path in (nc_path, csv_path):
        print(f'Saved: {path} ({path.stat().st_size:,} bytes)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir',
                        type=resolve_project_path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument('--output-dir', type=resolve_project_path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    profiles = []
    new_slope = new_p = era_distance = None
    for product in ('ERA5', 'MERRA2'):
        cache = load_product(args.raw_data_dir, product)
        distance = open_distance(args.raw_data_dir / 'DIST2LAND' / f'{product}_distance_to_coast.nc',
                                 cache[NH_MONTHS].isel(year=0, drop=True))
        profiles.append(calculate_profiles(product, cache, distance))
        if product == 'ERA5':
            new_slope, new_p = calculate_map(cache)
            era_distance = distance
        del cache
    profiles = pd.concat(profiles, ignore_index=True)
    spatial, profiles = assemble_outputs(profiles, new_slope, new_p, era_distance)
    write_outputs(output_dir, spatial, profiles)


if __name__ == '__main__':
    main()
