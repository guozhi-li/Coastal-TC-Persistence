#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.path import Path as MplPath
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/SST'


def configure_paths(database_root):
    """Set the raw-data paths without importing the plotting script."""
    global DATABASE_ROOT, SST_PRODUCTS, IBTRACS_FILE
    global CMIP6_ROOT, CMIP6_DISTANCE_ROOT
    DATABASE_ROOT = Path(database_root).expanduser().resolve()
    SST_PRODUCTS = {
        'OISST': (
            DATABASE_ROOT / 'ENVIRONMENT/OISST/oisst_sst_monmean_1982_2024.nc',
            DATABASE_ROOT / 'DIST2LAND/OISST_distance_to_coast.nc',
        ),
        'ERSST': (
            DATABASE_ROOT / 'ENVIRONMENT/ERSST/ersst_sst_monmean_1980_2024.nc',
            DATABASE_ROOT / 'DIST2LAND/ERSST_distance_to_coast.nc',
        ),
    }
    IBTRACS_FILE = DATABASE_ROOT / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv'
    CMIP6_ROOT = DATABASE_ROOT / 'CMIP6'
    CMIP6_DISTANCE_ROOT = DATABASE_ROOT / 'DIST2LAND'


configure_paths(DEFAULT_RAW_DATA_DIR)

SST_PROFILE_YEARS = {
    'OISST': np.arange(1982, 2025),
    'ERSST': np.arange(1980, 2025),
}

SST_PROFILE_EDGES_KM = np.arange(0.0, 801.0, 200.0)

SST_PROFILE_REGIONS = (
    ('B', '5-20N', 'NH (5–20°N)', 5.0, 20.0, (7, 8, 9, 10, 11), 'July–November'),
    ('C', '20-40N', 'NH (20–40°N)', 20.0, 40.0, (7, 8, 9, 10, 11), 'July–November'),
    ('D', '5-40S', 'SH (5–40°S)', -40.0, -5.0, (12, 1, 2, 3, 4), 'January–April and December'),
)

SST_EXCLUSION_POLYGON = np.asarray([
    (-26.0, 40.0), (60.0, 40.0), (40.0, 10.0), (23.0, -32.0),
    (23.0, -40.0), (-105.0, -40.0), (-105.0, 7.0),
    (-25.0, 7.0), (-26.0, 40.0),
], dtype=float)

SST_COORDINATE_CANDIDATES = {
    'time': ('time', 'valid_time', 'date'),
    'lat': ('lat', 'latitude'),
    'lon': ('lon', 'longitude'),
}

SENSITIVITY_LABELS = (
    'Original (1980–2024)', 'ERSST-removed',
    'Original (1982–2024)', 'OISST-removed',
)

CMIP6_EXCLUSION_POLYGON = SST_EXCLUSION_POLYGON

CMIP6_COORDINATE_CANDIDATES = {'time': ('time', 'valid_time', 'date'), 'lat': ('lat', 'latitude', 'nav_lat'), 'lon': ('lon', 'longitude', 'nav_lon')}

CMIP6_SST_CANDIDATES = ('tos', 'sst', 'SST', 'sea_surface_temperature')

CMIP6_DISTANCE_CANDIDATES = ('distance_to_coast', 'signed_distance_to_coast', 'distance', 'dist', 'CAL_DIST', 'signed_distance')

CMIP6_MODELS = ('ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-CM6-1', 'E3SM-2-0', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'NorESM2-LM')

CMIP6_EXPERIMENTS = (('hist-aer', 'AER'), ('hist-GHG', 'GHG'), ('hist-nat', 'NAT'), ('historical', 'HIST'))

CMIP6_FILE_START_YEAR, CMIP6_FILE_END_YEAR = (1980, 2014)

CMIP6_START_YEAR, CMIP6_END_YEAR = (1980, 2014)

CMIP6_MONTHS = (7, 8, 9, 10, 11)

CMIP6_LAT_MIN, CMIP6_LAT_MAX = (20.0, 40.0)

CMIP6_DISTANCE_EDGES_KM = np.arange(0.0, 800.0 + 200.0, 200.0)


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


def calculate_spatial_sst_trend():
    field_path, distance_path = SST_PRODUCTS['OISST'][:2]
    require_file(field_path)
    with open_netcdf(field_path, chunks='auto', decode_times=True) as ds:
        name = next((key for key in ('sst','SST','sea_surface_temperature')
                     if key in ds.data_vars), None)
        if name is None: raise KeyError(f'Missing SST variable: {field_path}')
        field = canonicalize_sst_field(ds[name], True)
        field = field.where((field.time.dt.year>=1982)&(field.time.dt.year<=2024),drop=True)
        field = field.sel(lat=slice(-40.,40.))
        pieces=[]
        for south,north,months,upper in (
            (-40.,0.,(12,1,2,3,4),False),
            (0.,40.,(7,8,9,10,11),True)):
            lat_test=(field.lat>=south)&((field.lat<=north) if upper else (field.lat<north))
            season=field.where(lat_test,drop=True)
            season=season.where(season.time.dt.month.isin(months),drop=True)
            years=pd.Series(np.asarray(season.time.dt.year.values,int))
            counts=years.value_counts().reindex(range(1982,2025),fill_value=0)
            if not counts.eq(5).all():
                raise ValueError(f'Incomplete M2A season: {counts[counts.ne(5)].to_dict()}')
            pieces.append(season.groupby('time.year').mean('time',skipna=True))
        annual=xr.concat(pieces,dim='lat').sortby('lat').astype('float32')
        distance=open_sst_distance(distance_path,field)
        annual=annual.where(distance>=0)
        print('Loading annual OISST fields for the five-year spatial trend ...')
        annual=annual.load()
    smoothed=annual.rolling(year=5,center=True,min_periods=5).mean().dropna('year',how='all')
    slope,pvalue=spatial_trend_and_p(smoothed)
    trend=slope.where(distance>=0).astype('float32')
    return trend,pvalue.where(np.isfinite(trend)),distance


def open_netcdf(path, **kwargs):
    """Read NetCDF, handling existing Unicode paths on Windows backends."""
    try:
        return xr.open_dataset(path, **kwargs)
    except FileNotFoundError:
        # Some netCDF4 Windows builds cannot resolve non-ASCII file paths.
        # ERSST is NetCDF3; OISST and most CMIP6 inputs use NetCDF4/HDF5.
        input_path = Path(path)
        if not input_path.is_file():
            raise
        with input_path.open('rb') as handle:
            signature = handle.read(8)
        if signature.startswith(b'CDF'):
            engine = 'scipy'
        elif signature == b'\x89HDF\r\n\x1a\n':
            engine = 'h5netcdf'
        else:
            raise
        return xr.open_dataset(path, engine=engine, **kwargs)


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')


def canonicalize_sst_field(array, with_time):
    names = set(array.dims) | set(array.coords)
    mapping = {}
    for name in ('lat', 'lon') + (('time',) if with_time else ()):
        original = next((alias for alias in SST_COORDINATE_CANDIDATES[name] if alias in names), None)
        if original is None:
            raise KeyError(f'No {name} coordinate in {array.name}: {sorted(names)}')
        if original != name:
            mapping[original] = name
    array = array.rename(mapping)
    allowed = {'lat', 'lon', 'time'} if with_time else {'lat', 'lon'}
    for dim in tuple(array.dims):
        if dim not in allowed:
            if array.sizes[dim] != 1:
                raise ValueError(f'Unexpected dimension {dim}={array.sizes[dim]}')
            array = array.isel({dim: 0}, drop=True)
    lon = ((array.lon.astype(float) + 180.0) % 360.0) - 180.0
    array = array.assign_coords(lon=lon).sortby('lon').sortby('lat')
    return array.sortby('time') if with_time else array


def open_sst_distance(path, sst):
    require_file(path)
    with open_netcdf(path) as ds:
        name = next((item for item in ('distance_to_coast',
                    'signed_distance_to_coast') if item in ds.data_vars), None)
        if name is None:
            raise KeyError(f'No signed distance variable in {path}')
        distance = canonicalize_sst_field(ds[name].squeeze(drop=True), False)
        unit = str(distance.attrs.get('units', '')).strip().lower()
        sign = str(distance.attrs.get('sign_convention', '')).lower()
        if unit in {'m', 'meter', 'meters', 'metre', 'metres'}:
            distance = distance / 1000.0
        if any(word in sign for word in ('negative over land', 'land negative', 'land < 0')):
            pass
        elif any(word in sign for word in ('positive over land', 'land positive', 'land > 0')):
            distance = -distance
        else:
            raise ValueError(f'Unknown distance sign convention: {path}')
        # The two SST products keep their own native grids. No interpolation.
        try:
            distance = distance.sel(lat=sst.lat, lon=sst.lon)
        except KeyError as error:
            raise ValueError(f'SST/distance coordinates differ: {path}') from error
        np.testing.assert_allclose(distance.lat.values, sst.lat.values, rtol=0, atol=1e-10)
        np.testing.assert_allclose(distance.lon.values, sst.lon.values, rtol=0, atol=1e-10)
        return distance.load()


def sst_exclusion_mask(distance):
    lon2d, lat2d = np.meshgrid(distance.lon.values, distance.lat.values)
    locations = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    inside = MplPath(SST_EXCLUSION_POLYGON, closed=True).contains_points(
        locations, radius=1e-10).reshape(lon2d.shape)
    return xr.DataArray(inside | (np.abs(lat2d) <= 5.0),
                        coords={'lat': distance.lat, 'lon': distance.lon},
                        dims=('lat', 'lon'))


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


def calculate_sst_profiles(source, field_path, distance_path):
    years_for_source = SST_PROFILE_YEARS[source]
    require_file(field_path)
    with open_netcdf(field_path, chunks='auto', decode_times=True) as ds:
        variable = next((name for name in ('sst', 'sea_surface_temperature')
                         if name in ds.data_vars), None)
        if variable is None:
            candidates = [name for name, item in ds.data_vars.items() if item.ndim >= 3]
            if len(candidates) != 1:
                raise KeyError(f'Cannot identify SST in {field_path}')
            variable = candidates[0]
        sst = canonicalize_sst_field(ds[variable], True)
        sst = sst.where(sst.time.dt.year.isin(years_for_source), drop=True)
        sst = sst.sel(lat=slice(-40.0, 40.0))
        sample = float(np.nanmedian(sst.isel(time=0,
                    lat=slice(None, None, 20), lon=slice(None, None, 20)).compute()))
        unit = str(sst.attrs.get('units', '')).strip().lower()
        is_kelvin = unit in {'k', 'kelvin', 'degrees_kelvin', 'degree_kelvin'} or sample > 100
        sst = sst.where((sst > 250) & (sst < 330)) if is_kelvin else \
              sst.where((sst > -5) & (sst < 50)) + 273.15
        distance = open_sst_distance(distance_path, sst)
        exclude = sst_exclusion_mask(distance)
        area = spherical_cell_area(distance.lat, distance.lon)
        rows = []
        for months in ((7, 8, 9, 10, 11), (12, 1, 2, 3, 4)):
            seasonal = sst.where(sst.time.dt.month.isin(months), drop=True)
            counts = pd.Series(seasonal.time.dt.year.values).value_counts()
            bad = {int(y): int(counts.get(y, 0)) for y in years_for_source
                   if int(counts.get(y, 0)) != len(months)}
            if bad:
                raise ValueError(f'{source} missing seasonal months: {bad}')
            annual = seasonal.groupby('time.year').mean('time', skipna=True).astype('float32').load()
            for panel, region_name, region_label, south, north, region_months, season in SST_PROFILE_REGIONS:
                if months != region_months:
                    continue
                region_mask = ((distance.lat >= south) & (distance.lat <= north)).broadcast_like(distance)
                base = region_mask & ~exclude & np.isfinite(distance)
                years = np.asarray(annual.year.values, dtype=float)
                for index, (left, right) in enumerate(zip(SST_PROFILE_EDGES_KM[:-1], SST_PROFILE_EDGES_KM[1:])):
                    in_bin = ((distance >= left) &
                              ((distance <= right) if index == len(SST_PROFILE_EDGES_KM)-2 else (distance < right)))
                    selected = base & in_bin
                    weights = area.where(selected)
                    yearly_mean = ((annual * weights).sum(('lat', 'lon'), skipna=True) /
                                   weights.where(annual.notnull()).sum(('lat', 'lon'), skipna=True))
                    values = np.asarray(yearly_mean.values, dtype=float)
                    valid = np.isfinite(years) & np.isfinite(values)
                    if valid.sum() != len(years_for_source):
                        raise ValueError(f'{source} {region_name} {left:g}–{right:g} km: '
                                         f'only {valid.sum()} valid years')
                    fit = stats.linregress(years[valid], values[valid])
                    ci = stats.t.ppf(.975, valid.sum()-2) * fit.stderr * 10
                    mk_p = mann_kendall_p(values[valid])
                    rows.append({'Panel': panel, 'Region': region_name,
                                 'Region_Label': region_label, 'Source': source,
                                 'Season': season, 'Start_Year': int(years_for_source[0]),
                                 'End_Year': int(years_for_source[-1]),
                                 'Distance_Center_km': (left+right)/2,
                                 'Grid_Cell_Count': int(selected.sum().item()),
                                 'OLS_Trend_K_per_Decade': float(fit.slope*10),
                                 'OLS_CI95_K_per_Decade': float(ci),
                                 'MK_P': float(mk_p)})
        return pd.DataFrame(rows)


def calculate_annual_cltc():
    require_file(IBTRACS_FILE)
    columns = ['SID', 'ISO_TIME', 'LAT', 'LON', 'CAL_DIST', 'USA_WIND']
    frame = pd.read_csv(IBTRACS_FILE, usecols=columns, keep_default_na=False,
                        low_memory=False)
    frame['SID'] = frame['SID'].astype(str).str.strip()
    frame['ISO_TIME'] = pd.to_datetime(frame['ISO_TIME'], errors='coerce')
    for name in ('LAT', 'LON', 'CAL_DIST', 'USA_WIND'):
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    frame = frame.loc[
        frame['SID'].ne('') & frame['ISO_TIME'].notna()
        & frame['ISO_TIME'].dt.year.between(1980, 2024)
    ].dropna(subset=['LAT', 'LON', 'CAL_DIST'])
    eligible = frame.assign(_WIND=frame.USA_WIND.fillna(0.0)).groupby('SID')['_WIND'].max()
    frame = frame.loc[frame.SID.isin(eligible.index[eligible >= 34.0])
                      & frame.LAT.gt(20.0) & frame.CAL_DIST.abs().lt(100.0)]
    coastal = frame.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    if coastal.empty:
        raise ValueError('No coastal TC points remain for panel E')
    coastal['_BLOCK'] = (coastal.groupby('SID', sort=False).ISO_TIME.diff()
                         .ne(pd.Timedelta(hours=3))
                         .groupby(coastal.SID, sort=False).cumsum().astype(int))
    events = []
    for (sid, block_id), block in coastal.groupby(['SID', '_BLOCK'], sort=False):
        block = block.sort_values('ISO_TIME').copy()
        hours = (block.ISO_TIME.iloc[-1] - block.ISO_TIME.iloc[0]).total_seconds()/3600.0
        if len(block) >= 9 and hours >= 24.0:
            block['Event_ID'] = f'{sid}_E{int(block_id):03d}'
            block['Event_Start'] = block.ISO_TIME.iloc[0]
            events.append(block)
    if not events:
        raise ValueError('No CLTC events remain for panel E')
    cltc = pd.concat(events, ignore_index=True)
    years = np.arange(1980, 2025)

    def yearly_hours(source, keys):
        hours = pd.Series(0.0, index=years)
        for _, block in source.groupby(keys, sort=False):
            if len(block) < 2:
                continue
            for year, count in block.sort_values('ISO_TIME').ISO_TIME.iloc[:-1].dt.year.value_counts().items():
                if year in hours.index:
                    hours.loc[year] += 3.0*int(count)
        return hours.to_numpy(float)

    coastal_hours = yearly_hours(coastal, ['SID', '_BLOCK'])
    cltc_hours = yearly_hours(cltc, ['Event_ID'])
    starts = cltc.groupby('Event_ID', sort=False).Event_Start.first()
    event_count = starts.dt.year.value_counts().reindex(years, fill_value=0).sort_index()
    annual = pd.DataFrame({
        'YEAR': years,
        'CLTC_COUNT': event_count.to_numpy(float),
        'CLTC_DURATION_HOURS': cltc_hours,
        'COASTAL_TC_DURATION_HOURS': coastal_hours,
        'DURATION_FRACTION': np.divide(cltc_hours, coastal_hours,
                                       out=np.full(len(years), np.nan),
                                       where=coastal_hours > 0),
    })
    if (annual.DURATION_FRACTION > 1.000001).any():
        raise ValueError('CLTC duration exceeds coastal-TC duration')
    return annual


def calculate_coastal_sst_index(source, start_year):
    field_path, distance_path = SST_PRODUCTS[source][:2]
    require_file(field_path)
    with open_netcdf(field_path, chunks={'time': 12}, decode_times=True) as dataset:
        name = next((key for key in ('sst', 'SST', 'sea_surface_temperature')
                     if key in dataset.data_vars), None)
        if name is None:
            raise KeyError(f'Missing SST variable: {field_path}')
        sst = canonicalize_sst_field(dataset[name], True).astype('float64')
        sst = sst.where(((sst.time.dt.year >= start_year) & (sst.time.dt.year <= 2024))
                        & sst.time.dt.month.isin((7, 8, 9, 10, 11)), drop=True)
        sst = sst.sel(lat=slice(20.0, 40.0))
        distance = open_sst_distance(distance_path, sst)
        exclude = sst_exclusion_mask(distance)
        sst = sst.where(~exclude & (np.abs(distance) < 200.0))
        counts = pd.Series(sst.time.dt.year.values).value_counts()
        missing = {year: int(counts.get(year, 0)) for year in range(start_year, 2025)
                   if int(counts.get(year, 0)) != 5}
        if missing:
            raise ValueError(f'{source} incomplete July–November SST years: {missing}')
        sample = float(np.nanmedian(sst.isel(time=0,
                        lat=slice(None, None, max(1, sst.sizes['lat']//8)),
                        lon=slice(None, None, max(1, sst.sizes['lon']//24))).compute()))
        unit = str(sst.attrs.get('units', '')).strip().lower()
        kelvin = unit in {'k', 'kelvin', 'degree_kelvin', 'degrees_kelvin'} or sample > 100
        sst = sst.where((sst > 250) & (sst < 330)) if kelvin else \
              sst.where((sst > -5) & (sst < 50)) + 273.15
        weights = spherical_cell_area(sst.lat, sst.lon)
        monthly = ((sst*weights).sum(('lat', 'lon'), skipna=True) /
                   weights.where(sst.notnull()).sum(('lat', 'lon'), skipna=True))
        annual = monthly.groupby('time.year').mean('time', skipna=True).load()
        table = pd.DataFrame({'YEAR': annual.year.values.astype(int),
                              'SST_K': np.asarray(annual.values, float)})
        table = table.set_index('YEAR').reindex(range(start_year, 2025)).rename_axis('YEAR').reset_index()
        if table.SST_K.isna().any():
            raise ValueError(f'{source} annual SST missing: {table.loc[table.SST_K.isna(), "YEAR"].tolist()}')
        return table


def remove_sst_component(years, response, sst):
    years, response, sst = (np.asarray(item, float) for item in (years, response, sst))
    valid = np.isfinite(years) & np.isfinite(response) & np.isfinite(sst)
    if valid.sum() < 3:
        raise ValueError('Too few matching years for SST removal')
    sst_sd = float(np.std(sst[valid], ddof=1))
    if not np.isfinite(sst_sd) or np.isclose(sst_sd, 0):
        raise ValueError('SST series has zero variance')
    sst_z = (sst[valid] - np.mean(sst[valid]))/sst_sd
    fit = stats.linregress(sst_z, response[valid])
    adjusted = response[valid] - (fit.intercept + fit.slope*sst_z) + np.mean(response[valid])
    return pd.DataFrame({'YEAR': years[valid].astype(int), 'VALUE': adjusted})


def calculate_sensitivity_trend(years, values):
    years, values = np.asarray(years, float), np.asarray(values, float)
    valid = np.isfinite(years) & np.isfinite(values)
    years, values = years[valid], values[valid]
    if len(values) < 3:
        raise ValueError('Too few annual observations for panel E')
    if np.unique(values).size <= 1:
        return 0.0, 0.0, 1.0, len(values)
    fit = stats.linregress(years, values)
    ci = stats.t.ppf(.975, len(values)-2)*fit.stderr*10.0
    return float(fit.slope*10.0), float(ci), mann_kendall_p(values), len(values)


def calculate_cltc_sst_sensitivity():
    annual = calculate_annual_cltc()
    series = {
        SENSITIVITY_LABELS[0]: annual[['YEAR', 'CLTC_COUNT', 'DURATION_FRACTION']],
        SENSITIVITY_LABELS[2]: annual.loc[annual.YEAR.ge(1982),
                                     ['YEAR', 'CLTC_COUNT', 'DURATION_FRACTION']],
    }
    for source, first_year, label in (('ERSST', 1980, SENSITIVITY_LABELS[1]),
                                      ('OISST', 1982, SENSITIVITY_LABELS[3])):
        sst = calculate_coastal_sst_index(source, first_year)
        matched = annual.loc[annual.YEAR.ge(first_year)].merge(sst, on='YEAR', validate='one_to_one')
        count = remove_sst_component(matched.YEAR, matched.CLTC_COUNT, matched.SST_K)
        fraction = remove_sst_component(matched.YEAR, matched.DURATION_FRACTION, matched.SST_K)
        series[label] = count.rename(columns={'VALUE': 'CLTC_COUNT'}).merge(
            fraction.rename(columns={'VALUE': 'DURATION_FRACTION'}),
            on='YEAR', validate='one_to_one')
    rows = []
    for label in SENSITIVITY_LABELS:
        for metric, column, factor, unit in (
            ('CLTC count', 'CLTC_COUNT', 1.0, 'events decade^-1'),
            ('Duration fraction', 'DURATION_FRACTION', 100.0, 'percentage points decade^-1'),
        ):
            table = series[label][['YEAR', column]].dropna()
            trend, ci, p, n = calculate_sensitivity_trend(table.YEAR, table[column])
            rows.append({'Condition': label, 'Metric': metric,
                         'Start_Year': int(table.YEAR.min()),
                         'End_Year': int(table.YEAR.max()), 'N_Years': n,
                         'Trend': trend*factor, 'CI95': ci*factor,
                         'MK_P': p, 'Unit': unit})
    return pd.DataFrame(rows)


def find_cmip6_name(names, candidates, description, required=True):
    """Find a coordinate/data name, case-insensitively if needed."""
    for candidate in candidates:
        if candidate in names:
            return candidate
    lowered = {str(name).lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise KeyError(f'Cannot find {description}; available names: {sorted(map(str, names))}')
    return None


def normalize_cmip6_longitude(array):
    if 'lon' not in array.coords:
        raise KeyError('Longitude coordinate is missing')
    lon = (array.lon.astype(float) + 180.0) % 360.0 - 180.0
    array = array.assign_coords(lon=lon)
    if array.lon.ndim == 1:
        array = array.sortby('lon')
        values = np.asarray(array.lon.values, dtype=float)
        _, unique = np.unique(np.round(values, 10), return_index=True)
        if len(unique) != len(values):
            array = array.isel(lon=np.sort(unique))
    return array


def choose_cmip6_data_variable(dataset, candidates, description):
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
    lowered = {str(name).lower(): name for name in dataset.data_vars}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    numeric = [name for name, data in dataset.data_vars.items() if data.ndim >= 2 and np.issubdtype(data.dtype, np.number)]
    if len(numeric) == 1:
        return numeric[0]
    raise KeyError(f'Cannot identify {description}; available variables: {list(dataset.data_vars)}')


def rename_cmip6_coordinates(array):
    names = set(array.dims) | set(array.coords)
    rename = {}
    for standard in ('time', 'lat', 'lon'):
        found = find_cmip6_name(names, CMIP6_COORDINATE_CANDIDATES[standard], standard, required=standard != 'time')
        if found is not None and found != standard:
            rename[found] = standard
    return array.rename(rename) if rename else array


def cmip6_horizontal_dims(array):
    dims = []
    for coordinate in ('lat', 'lon'):
        if coordinate in array.coords:
            for dim in array[coordinate].dims:
                if dim not in dims:
                    dims.append(dim)
    if not dims:
        raise ValueError('Cannot infer horizontal dimensions from lat/lon coordinates')
    return tuple(dims)


def cmip6_horizontal_coordinates(array):
    if array.lat.ndim == 1 and array.lon.ndim == 1:
        return xr.broadcast(array.lat, array.lon)
    if array.lat.ndim == 2 and array.lon.ndim == 2:
        return (array.lat, array.lon)
    raise ValueError('Latitude and longitude must both be 1-D or both be 2-D')


def open_cmip6_sst(path):
    require_file(path)
    dataset = open_netcdf(path, decode_times=True, use_cftime=True)
    variable = choose_cmip6_data_variable(dataset, CMIP6_SST_CANDIDATES, 'SST')
    array = normalize_cmip6_longitude(rename_cmip6_coordinates(dataset[variable]).squeeze(drop=True))
    if not {'time', 'lat', 'lon'}.issubset(array.coords):
        raise ValueError(f'{path.name}: SST must have time/lat/lon coordinates')
    extra = [dim for dim in array.dims if dim != 'time' and dim not in cmip6_horizontal_dims(array)]
    if extra:
        raise ValueError(f'{path.name}: unsupported non-horizontal dimensions: {extra}')
    array.name = 'SST'
    return array


def open_cmip6_distance(path, field):
    require_file(path)
    dataset = open_netcdf(path)
    variable = choose_cmip6_data_variable(dataset, CMIP6_DISTANCE_CANDIDATES, 'distance to coast')
    distance = normalize_cmip6_longitude(rename_cmip6_coordinates(dataset[variable]).squeeze(drop=True))
    if 'time' in distance.dims:
        if distance.sizes['time'] != 1:
            raise ValueError(f'{path.name}: distance field unexpectedly has multiple times')
        distance = distance.isel(time=0, drop=True)
    field_dims = cmip6_horizontal_dims(field)
    distance_dims = cmip6_horizontal_dims(distance)
    if len(field_dims) != len(distance_dims):
        raise ValueError(f'{path.name}: field horizontal dims {field_dims} and distance dims {distance_dims} differ')
    rename_dims = {}
    for source_dim, target_dim in zip(distance_dims, field_dims):
        if distance.sizes[source_dim] != field.sizes[target_dim]:
            raise ValueError(f'{path.name}: dimension {source_dim}={distance.sizes[source_dim]} does not match {target_dim}={field.sizes[target_dim]}')
        if source_dim != target_dim:
            rename_dims[source_dim] = target_dim
    if rename_dims:
        distance = distance.rename(rename_dims)
    for coordinate in ('lat', 'lon'):
        distance = distance.assign_coords({coordinate: field[coordinate]})
    distance = distance.transpose(*field_dims)
    if not (bool((distance < 0).any()) and bool((distance > 0).any())):
        raise ValueError(f'{path.name}: expected signed distance with land negative and ocean positive')
    return distance.astype(np.float32)


def resolve_case_insensitive_file(directory, expected_name):
    expected = directory / expected_name
    if expected.is_file():
        return expected
    matches = [p for p in directory.glob('*.nc') if p.name.casefold() == expected_name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(expected)


def cmip6_sst_file(model, experiment):
    return resolve_case_insensitive_file(CMIP6_ROOT, f'{model}_{experiment}_tos_{CMIP6_FILE_START_YEAR}-{CMIP6_FILE_END_YEAR}.nc')


def cmip6_distance_file(model):
    return resolve_case_insensitive_file(CMIP6_DISTANCE_ROOT, f'CMIP6_{model}_tos_distance_to_coast.nc')


def validate_cmip6_inventory():
    missing, resolved = ([], {})
    for model in CMIP6_MODELS:
        try:
            resolved[model, 'distance'] = cmip6_distance_file(model)
        except FileNotFoundError as exc:
            missing.append(str(exc))
        for experiment, _ in CMIP6_EXPERIMENTS:
            try:
                resolved[model, experiment] = cmip6_sst_file(model, experiment)
            except FileNotFoundError as exc:
                missing.append(str(exc))
    if missing:
        raise FileNotFoundError('Missing required CMIP6 inputs:\n' + '\n'.join(missing))
    print(f'Validated {len(CMIP6_MODELS)} models x {len(CMIP6_EXPERIMENTS)} experiments plus {len(CMIP6_MODELS)} distance files.')
    return resolved


def cmip6_exclusion_mask(field):
    lat2d, lon2d = cmip6_horizontal_coordinates(field)
    points = np.column_stack((lon2d.values.ravel(), lat2d.values.ravel()))
    inside = MplPath(CMIP6_EXCLUSION_POLYGON).contains_points(points)
    return xr.DataArray(inside.reshape(lat2d.shape), dims=lat2d.dims, coords=lat2d.coords, name='excluded')


def complete_cmip6_seasonal_annual(field, start_year, end_year, months, label):
    years = field.time.dt.year
    field = field.where((years >= start_year) & (years <= end_year), drop=True)
    if field.sizes.get('time', 0) == 0:
        raise ValueError(f'{label}: no data in requested years {start_year}-{end_year}')
    field = field.where(field.time.dt.month.isin(months), drop=True)
    counts = pd.Series(np.asarray(field.time.dt.year.values, dtype=int)).value_counts().reindex(range(start_year, end_year + 1), fill_value=0)
    incomplete = counts[counts != len(months)]
    if not incomplete.empty:
        raise ValueError(f'{label}: each year must contain {len(months)} months {months}; incomplete={incomplete.to_dict()}')
    annual = field.groupby('time.year').mean('time', skipna=True).sel(year=slice(start_year, end_year))
    with ProgressBar():
        annual = annual.load()
    return annual.astype(np.float32)


def cmip6_spherical_cell_area(field):
    """Spherical areas on a model's native rectilinear or curvilinear grid."""
    if field.lat.ndim == field.lon.ndim == 1:
        return spherical_cell_area(field.lat, field.lon)
    lat, lon = cmip6_horizontal_coordinates(field)
    latitude = np.asarray(lat.values, dtype=float)
    longitude = np.asarray(lon.values, dtype=float)
    if latitude.shape != longitude.shape or min(latitude.shape) < 2:
        raise ValueError('Invalid two-dimensional CMIP6 latitude/longitude grid')
    # Unwrap the dateline before inferring cell corners from neighbouring centres.
    longitude = np.rad2deg(np.unwrap(np.unwrap(
        np.deg2rad(longitude), axis=1), axis=0))

    def inferred_corners(centres):
        padded = np.empty((centres.shape[0] + 2, centres.shape[1] + 2))
        padded[1:-1, 1:-1] = centres
        padded[0, 1:-1] = 2 * centres[0] - centres[1]
        padded[-1, 1:-1] = 2 * centres[-1] - centres[-2]
        padded[:, 0] = 2 * padded[:, 1] - padded[:, 2]
        padded[:, -1] = 2 * padded[:, -2] - padded[:, -3]
        return (padded[:-1, :-1] + padded[1:, :-1] +
                padded[:-1, 1:] + padded[1:, 1:]) / 4

    corner_lat = np.clip(inferred_corners(latitude), -90, 90)
    corner_lon = inferred_corners(longitude)
    phi, lam = np.deg2rad(corner_lat), np.deg2rad(corner_lon)
    xyz = np.stack((np.cos(phi) * np.cos(lam),
                    np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    a, b = xyz[:-1, :-1], xyz[:-1, 1:]
    c, d = xyz[1:, 1:], xyz[1:, :-1]

    def triangle_area(p, q, r):
        triple = np.abs(np.sum(p * np.cross(q, r), axis=-1))
        denominator = (1 + np.sum(p * q, axis=-1) +
                       np.sum(q * r, axis=-1) + np.sum(r * p, axis=-1))
        return 2 * np.arctan2(triple, denominator)

    area = 6_371_000.0 ** 2 * (triangle_area(a, b, c) +
                               triangle_area(a, c, d))
    if not np.any(np.isfinite(area) & (area > 0)):
        raise ValueError('No valid spherical CMIP6 grid-cell areas')
    return xr.DataArray(area, coords={name: field[name] for name in ('lat', 'lon')},
                        dims=lat.dims, name='spherical_cell_area',
                        attrs={'units': 'm2',
                               'note': '2-D cell corners inferred from adjacent centres'})


def cmip6_distance_bin_series(field, distance, label):
    annual = complete_cmip6_seasonal_annual(field, CMIP6_START_YEAR, CMIP6_END_YEAR, CMIP6_MONTHS, label)
    lat2d, _ = cmip6_horizontal_coordinates(field)
    weights = cmip6_spherical_cell_area(field)
    base = ((lat2d >= CMIP6_LAT_MIN) & (lat2d <= CMIP6_LAT_MAX) &
            np.isfinite(distance) & ~cmip6_exclusion_mask(field) &
            np.isfinite(weights) & (weights > 0))
    spatial_dims = cmip6_horizontal_dims(field)
    rows = []
    for index, (left, right) in enumerate(zip(CMIP6_DISTANCE_EDGES_KM[:-1], CMIP6_DISTANCE_EDGES_KM[1:])):
        in_bin = (distance >= left) & (distance <= right) if index == len(CMIP6_DISTANCE_EDGES_KM) - 2 else (distance >= left) & (distance < right)
        mask = base & in_bin
        selected_cells = int(mask.sum().item())
        if selected_cells == 0:
            raise ValueError(f'{label}: no grid cells in offshore distance bin {left:g}-{right:g} km')
        bin_weights = weights.where(mask)
        valid_weights = bin_weights.where(annual.notnull())
        series = (annual * valid_weights).sum(spatial_dims, skipna=True) / valid_weights.sum(spatial_dims, skipna=True)
        table = series.to_dataframe(name='SST').reset_index()[['year', 'SST']]
        table = table.rename(columns={'year': 'Year'})
        if table['SST'].isna().any():
            missing = table.loc[table['SST'].isna(), 'Year'].astype(int).tolist()
            raise ValueError(f'{label}/{left:g}-{right:g} km: annual SST missing in years {missing}')
        table['Distance_Left_km'] = float(left)
        table['Distance_Right_km'] = float(right)
        table['Distance_Center_km'] = float((left + right) / 2.0)
        table['Grid_Cell_Count'] = selected_cells
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def calculate_cmip6_trend_statistics(table):
    years = table['Year'].to_numpy(float)
    values = table['SST'].to_numpy(float)
    fit = stats.linregress(years, values)
    tcrit = float(stats.t.ppf(0.975, len(values) - 2))
    mk_result = mk.original_test(values)
    tau, mk_p = mk_result.Tau, mann_kendall_p(values)
    return {'N_Years': int(len(values)), 'Mean_SST': float(np.mean(values)), 'OLS_Trend_K_per_Decade': float(fit.slope * 10.0), 'OLS_CI95_K_per_Decade': float(tcrit * fit.stderr * 10.0), 'OLS_P': float(fit.pvalue), 'MK_Tau': float(tau), 'MK_P': float(mk_p)}


def calculate_cmip6_model_trends(inventory):
    trend_rows = []
    for model in CMIP6_MODELS:
        distance_path = inventory[model, 'distance']
        for experiment, forcing in CMIP6_EXPERIMENTS:
            input_path = inventory[model, experiment]
            print(f'Processing {model}/{experiment}: {input_path.name}')
            field = open_cmip6_sst(input_path)
            distance = open_cmip6_distance(distance_path, field)
            annual_bins = cmip6_distance_bin_series(
                field, distance, f'{model}/{experiment}'
            )
            for (left, right, center), table in annual_bins.groupby(
                ['Distance_Left_km', 'Distance_Right_km', 'Distance_Center_km'],
                sort=True,
            ):
                trend_rows.append({
                    'Model': model,
                    'Experiment': experiment,
                    'Forcing': forcing,
                    'Distance_Left_km': float(left),
                    'Distance_Right_km': float(right),
                    'Distance_Center_km': float(center),
                    **calculate_cmip6_trend_statistics(table[['Year', 'SST']]),
                })
            field.close()

    trends = pd.DataFrame(trend_rows)
    expected_bins = len(CMIP6_DISTANCE_EDGES_KM) - 1
    counts = trends.groupby('Forcing').agg(
        Model_Count=('Model', 'nunique'),
        Bin_Count=('Distance_Center_km', 'nunique'),
    )
    bad = counts[
        (counts['Model_Count'] != len(CMIP6_MODELS))
        | (counts['Bin_Count'] != expected_bins)
    ]
    if not bad.empty:
        raise RuntimeError(f'Incomplete forcing ensembles: {bad}')
    return trends


def summarize_cmip6_ensemble(model_trends):
    rows = []
    for _, forcing in CMIP6_EXPERIMENTS:
        selected = model_trends[model_trends['Forcing'].eq(forcing)]
        for (left, right, center), group in selected.groupby(['Distance_Left_km', 'Distance_Right_km', 'Distance_Center_km'], sort=True):
            values = group['OLS_Trend_K_per_Decade'].to_numpy(float)
            if len(values) != len(CMIP6_MODELS):
                raise RuntimeError(f'{forcing}/{left:g}-{right:g} km contains {len(values)} models, expected {len(CMIP6_MODELS)}')
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append({'Record_Type': 'CMIP6 ensemble distance-bin summary', 'Forcing': forcing, 'Distance_Left_km': float(left), 'Distance_Right_km': float(right), 'Distance_Center_km': float(center), 'Model_Count': int(len(values)), 'Q05_K_per_Decade': float(q05), 'Q25_K_per_Decade': float(q25), 'Median_K_per_Decade': float(median), 'Q75_K_per_Decade': float(q75), 'Q95_K_per_Decade': float(q95), 'Mean_K_per_Decade': float(np.mean(values)), 'Standard_Deviation_K_per_Decade': float(np.std(values, ddof=1)), 'Minimum_K_per_Decade': float(np.min(values)), 'Maximum_K_per_Decade': float(np.max(values))})
    return pd.DataFrame(rows)

# =============================================================================
# Minimal plotting data exports
# =============================================================================
PROFILE_COLUMNS = (
    'Region', 'Source', 'Distance_Center_km', 'OLS_Trend_K_per_Decade',
    'OLS_CI95_K_per_Decade', 'MK_P',
)
SENSITIVITY_COLUMNS = ('Condition', 'Metric', 'Trend', 'CI95', 'MK_P')
ATTRIBUTION_COLUMNS = ('Forcing', 'Distance_Center_km', 'Median_K_per_Decade')


def save_spatial_data(output_dir, trend, pvalue, distance):
    """Store native-grid plotting fields with lossless NetCDF compression."""
    trend, pvalue, distance = xr.align(
        trend.transpose('lat', 'lon').reset_coords(drop=True),
        pvalue.transpose('lat', 'lon').reset_coords(drop=True),
        distance.transpose('lat', 'lon').reset_coords(drop=True),
        join='exact',
    )
    data = xr.Dataset({
        'sst_trend': trend,
        'MK_P': pvalue.astype('float64'),
        'distance_to_coast': distance,
    })
    data.sst_trend.attrs = {
        'long_name': 'OLS trend in five-year-smoothed seasonal mean SST',
        'units': 'K decade-1',
    }
    data.MK_P.attrs = {
        'long_name': 'Two-sided original Mann-Kendall P value',
        'units': '1',
        'method': 'Tie-corrected variance and continuity correction; normal survival function',
    }
    data.distance_to_coast.attrs = {
        'long_name': 'Signed distance to the coastline used for plotting masks',
        'units': 'km',
        'sign_convention': 'negative over land; positive over ocean',
    }
    data.lat.attrs = {'standard_name': 'latitude', 'units': 'degrees_north'}
    data.lon.attrs = {'standard_name': 'longitude', 'units': 'degrees_east'}
    data.attrs = {
        'title': 'Figure M2A plotting data',
        'source': 'OISST',
        'analysis_years': '1982-2024',
        'season_NH': 'July-November',
        'season_SH': 'January-April and December in the same calendar year',
        'smoother': 'centered five-year mean; five complete years required',
        'trend_method': 'ordinary least squares',
        'significance_method': 'Original Mann-Kendall test; plotting threshold P < 0.05',
    }
    encoding = {
        name: {'zlib': True, 'complevel': 4, 'shuffle': True}
        for name in data.data_vars
    }
    path = output_dir / 'FigM2A_SST_spatial.nc'
    temporary = path.with_suffix('.nc.tmp')
    try:
        data.to_netcdf(temporary, engine='h5netcdf', format='NETCDF4', encoding=encoding)
        temporary.replace(path)
    finally:
        data.close()
        if temporary.exists():
            temporary.unlink()
    print(f'Saved: {path}')


def save_csv(table, columns, expected_rows, path):
    """Export only the requested columns, retaining full numerical precision."""
    data = table.loc[:, list(columns)].copy()
    if len(data) != expected_rows:
        raise ValueError(f'{path.name}: expected {expected_rows} rows, got {len(data)}')
    if 'MK_P' in data and (not np.isfinite(data.MK_P).all() or not data.MK_P.between(0, 1).all()):
        raise ValueError(f'{path.name}: invalid MK P values')
    temporary = path.with_suffix('.csv.tmp')
    try:
        data.to_csv(temporary, index=False, encoding='utf-8', float_format='%.17g')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Saved: {path} ({len(data)} rows)')


def preprocess(output_dir, panels):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if 'A' in panels:
        print('Panel A: spatial SST trends and significance ...')
        save_spatial_data(output_dir, *calculate_spatial_sst_trend())
    if 'BCD' in panels:
        tables = []
        for source, paths in SST_PRODUCTS.items():
            years = SST_PROFILE_YEARS[source]
            print(f'Panels B-D: {source}, {years[0]}-{years[-1]} ...')
            tables.append(calculate_sst_profiles(source, *paths))
        profiles = pd.concat(tables, ignore_index=True)
        save_csv(profiles, PROFILE_COLUMNS, 24,
                 output_dir / 'FigM2BCD_SST_profiles.csv')
    if 'E' in panels:
        print('Panel E: CLTC trends after SST removal ...')
        sensitivity = calculate_cltc_sst_sensitivity()
        save_csv(sensitivity, SENSITIVITY_COLUMNS, 8,
                 output_dir / 'FigM2E_CLTC_SST_sensitivity.csv')
    if 'F' in panels:
        print('Panel F: CMIP6 SST attribution ...')
        inventory = validate_cmip6_inventory()
        trends = calculate_cmip6_model_trends(inventory)
        attribution = summarize_cmip6_ensemble(trends)
        save_csv(attribution, ATTRIBUTION_COLUMNS, 16,
                 output_dir / 'FigM2F_CMIP6_SST_attribution.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir', type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Directory containing the original input datasets.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for the preprocessed NC and CSV files.')
    parser.add_argument('--panels', nargs='+', choices=('A', 'BCD', 'E', 'F'),
                        default=('A', 'BCD', 'E', 'F'),
                        help='Groups to preprocess; default: all four groups.')
    args = parser.parse_args(argv)
    configure_paths(args.raw_data_dir)
    print(f'Raw inputs: {DATABASE_ROOT}')
    print(f'Preprocessed data: {args.output_dir.expanduser().resolve()}')
    preprocess(args.output_dir, args.panels)


if __name__ == '__main__':
    main()
