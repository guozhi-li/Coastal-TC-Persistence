#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import tempfile

from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/EXPOSURE'
TRACK_RELATIVE = Path('IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv')
GISA_RELATIVE = Path('GISA/GISA_impervious_fraction_1985_2021_0p01deg.nc')
TRACK_FILE = DEFAULT_RAW_DATA_DIR / TRACK_RELATIVE
GISA_NC_FILE = DEFAULT_RAW_DATA_DIR / GISA_RELATIVE
PRECIP_ROOT = DEFAULT_RAW_DATA_DIR / 'MSWEP'


GISA_YEAR_MIN, GISA_YEAR_MAX = 1985, 2021

GISA_YEARS = np.arange(GISA_YEAR_MIN, GISA_YEAR_MAX + 1, dtype=int)

TIME_STEP_HOURS = 3

CLTC_MIN_POINTS = 9

WIND_LIMIT_KT = 34.0

TC_RADIUS_KM = 500.0

HEAVY_RAIN_MM_3H = 30.0

MSWEP_RESOLUTION = 0.1

DISTANCE_LIMIT_KM = 100.0

MIN_EXPOSURE_FREQUENCY = 1

REPEAT_THRESHOLDS = (1, 2, 3)

MAX_EXPOSURE_GAP_HOURS = 0

EXPOSURE_RULE = 'consecutive_exposure_observations'

CONSECUTIVE_COUNTING_POLICY = 'qualifying_runs_only'

EXPOSURE_DISTANCE_CASES = (
    ('m300_m100', '−300–−100 km', '#C25D34'),
    ('pm100', '±100 km', '#496181'),
    ('p100_p300', '100–300 km', '#6A946F'),
)

TREND_METHOD = 'ols'

SIGNIFICANCE_METHOD = 'mk'

TREND_SMOOTHING = 'centered_5_year_full_window'

TREND_SMOOTH_WINDOW = 5

SEN_CHUNK_SIZE = 2048

PROGRESS_EVERY = 20

REGION_ORDER = ("NH (>20°N)", "East Asia", "North America")

MAP_EXTENTS = {
    "East Asia": (100.0, 145.0, 15.0, 45.0),
    "North America": (-130.0, -65.0, 15.0, 45.0),
}

WNA_POLYGON_VERTICES = [
    (-105, 30), (-98, 18), (-92, 16.6), (-90, 15), (-85, 13),
    (-83, 9), (-81, 8), (-79.3, 9.3), (-78, 8.8), (-76, 6),
    (-48, 6), (-48, 60), (-105, 60),
]

EP_POLYGON_VERTICES = [
    (-105, 30), (-98, 18), (-92, 16.6), (-90, 15), (-85, 13),
    (-83, 9), (-81, 8), (-79.3, 9.3), (-78, 8.8), (-76, 6),
    (-130, 6), (-130, 37), (-105, 37),
]

WNP_POLYGON_VERTICES = [
    (102, 15), (110, 8), (113, 5), (148, 5),
    (148, 55), (102, 55), (102, 15),
]

PATH_WNA = MplPath(WNA_POLYGON_VERTICES)

PATH_EP = MplPath(EP_POLYGON_VERTICES)

PATH_WNP = MplPath(WNP_POLYGON_VERTICES)

SERIES_COLUMNS = (
    'year', 'exposure_raw', 'raw_ols_fitted', 'raw_ols_ci95_lower',
    'raw_ols_ci95_upper', 'exposure_smooth5', 'smooth5_ols_fitted',
    'smooth5_ols_ci95_lower', 'smooth5_ols_ci95_upper',
)

STAT_COLUMNS = (
    'panel', 'series', 'distance_case', 'repeat_threshold', 'slope_decade',
    'slope_ci95_lower', 'slope_ci95_upper', 'trend_p',
)

YEARS = np.arange(1985, 2022)

REGIONS = {'C': 'East Asia', 'D': 'North America'}

DISTANCE_KEYS = ('m300_m100', 'pm100', 'p100_p300')

def open_source_netcdf(path, **kwargs):
    """Read NetCDF3/HDF5 paths safely, including non-ASCII Windows paths."""
    path = Path(path)
    with path.open('rb') as handle:
        signature = handle.read(8)
    kwargs.setdefault('engine', 'scipy' if signature.startswith(b'CDF') else 'h5netcdf')
    return xr.open_dataset(path, **kwargs)

def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')

def normalize_lon_180(values):
    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0

def point_region(lon, lat):
    point = (float(lon), float(lat))
    if PATH_WNA.contains_point(point):
        return 'WNA'
    if PATH_EP.contains_point(point):
        return 'EP'
    if PATH_WNP.contains_point(point):
        return 'WNP'
    return 'OTHER'

def load_tracks():
    require_file(TRACK_FILE, 'processed IBTrACS track file')
    required = ['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'CAL_DIST']
    frame = pd.read_csv(TRACK_FILE, usecols=required, keep_default_na=False, low_memory=False)
    frame.SID = frame.SID.astype(str).str.strip()
    frame.ISO_TIME = pd.to_datetime(frame.ISO_TIME, errors='coerce')
    for column in ['LAT', 'LON', 'USA_WIND', 'CAL_DIST']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame[frame.SID.ne('') & frame.ISO_TIME.notna() & frame.ISO_TIME.dt.year.between(1980, 2024)].dropna(subset=['LAT', 'LON', 'CAL_DIST']).copy()
    frame = frame.sort_values(['SID', 'ISO_TIME']).drop_duplicates(['SID', 'ISO_TIME'])
    frame.LON = normalize_lon_180(frame.LON)
    frame['YEAR'] = frame.ISO_TIME.dt.year.astype(int)
    frame['point_region'] = [point_region(lon, lat) for lon, lat in zip(frame.LON, frame.LAT)]
    sid_lmi = frame.groupby('SID').USA_WIND.max()
    eligible = sid_lmi[sid_lmi >= WIND_LIMIT_KT].index
    frame['LMI_WIND'] = frame.SID.map(sid_lmi)
    frame['LMI_ELIGIBLE'] = frame.SID.isin(eligible)
    print(f'Loaded {len(frame):,} points; LMI-eligible SIDs: {len(eligible):,}')
    return frame

def region_mask(frame, region):
    base = frame.LAT > 20
    if region == 'NH (>20°N)':
        return base
    if region == 'East Asia':
        return base & frame.point_region.eq('WNP')
    if region == 'North America':
        return base & frame.point_region.isin(['EP', 'WNA'])
    raise ValueError(region)

def identify_cltc_events(frame, region):
    """Identify contiguous 3-hour CLTC events within |CAL_DIST| < 100 km."""
    selected = frame[
        frame.LMI_ELIGIBLE
        & frame.YEAR.between(GISA_YEAR_MIN, GISA_YEAR_MAX)
        & region_mask(frame, region)
        & (frame.CAL_DIST.abs() < DISTANCE_LIMIT_KM)
    ].copy()
    blocks = []
    event_number = 0
    for sid, storm in selected.groupby('SID', sort=False):
        storm = storm.sort_values('ISO_TIME')
        ids = storm.ISO_TIME.diff().ne(pd.Timedelta(hours=TIME_STEP_HOURS)).cumsum()
        for _, block in storm.groupby(ids, sort=False):
            if len(block) < CLTC_MIN_POINTS:
                continue
            event_number += 1
            block = block.copy()
            block['EVENT_ID'] = f'pm100_{region}_{sid}_E{event_number:05d}'
            block['EVENT_YEAR'] = int(block.ISO_TIME.iloc[0].year)
            blocks.append(block)
    columns = list(frame.columns) + ['EVENT_ID', 'EVENT_YEAR']
    result = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=columns)
    print(f'CLTC events: ±100 km, {region}: {result.EVENT_ID.nunique():,}')
    return result

def precipitation_file_for_time(timestamp):
    timestamp = pd.Timestamp(timestamp)
    return PRECIP_ROOT / f'{timestamp.year:04d}' / f'{timestamp.year:04d}{timestamp.dayofyear:03d}.{timestamp.hour:02d}.nc'

def select_exact_precipitation(dataset, timestamp, path):
    if 'precipitation' not in dataset or 'time' not in dataset.coords:
        raise KeyError(f'Missing precipitation/time: {path}')
    target = np.datetime64(pd.Timestamp(timestamp).to_datetime64(), 'ns')
    times = np.asarray(dataset.time.values).astype('datetime64[ns]')
    matches = np.flatnonzero(times == target)
    if len(matches) != 1:
        raise KeyError(f'MSWEP time mismatch: {timestamp}, {path}')
    return dataset.precipitation.isel(time=int(matches[0]), drop=True)

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    phi1 = np.radians(float(lat1))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlambda = np.radians((np.asarray(lon2, dtype=float) - float(lon1) + 180) % 360 - 180)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def subset_tc_box(data, tc_lat, tc_lon):
    """Select a TC-centered box with longitudes ordered across the cyclic seam."""
    lat = np.asarray(data.lat.values, dtype=float)
    lon = np.asarray(data.lon.values, dtype=float)
    lat_half = TC_RADIUS_KM / 111 + 0.2
    lon_half = min(180, TC_RADIUS_KM / (111 * max(abs(math.cos(math.radians(float(tc_lat)))), 0.05)) + 0.2)
    ii = np.flatnonzero(abs(lat - float(tc_lat)) <= lat_half)
    delta_lon = (lon - float(tc_lon) + 180.0) % 360.0 - 180.0
    jj = np.flatnonzero(abs(delta_lon) <= lon_half)
    if not len(ii) or not len(jj):
        raise ValueError(f'Empty MSWEP subset at {tc_lat}, {tc_lon}')
    # A box crossing the longitude seam selects both ends of the native array.
    # Order by offset from the TC; isel reorders precipitation and coordinates together.
    jj = jj[np.argsort(delta_lon[jj], kind='stable')]
    return data.isel(lat=ii, lon=jj)

def coordinate_edges(values, circular=False):
    values = np.asarray(values, dtype=float)
    working = np.rad2deg(np.unwrap(np.deg2rad(values))) if circular else values.copy()
    if len(working) == 1:
        return np.array([working[0] - 0.05, working[0] + 0.05])
    edges = np.empty(len(working) + 1)
    edges[1:-1] = (working[:-1] + working[1:]) / 2
    edges[0] = working[0] - (working[1] - working[0]) / 2
    edges[-1] = working[-1] + (working[-1] - working[-2]) / 2
    return edges

def validate_regular_grid(lat, lon, spacing=MSWEP_RESOLUTION):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) > 1 and (not np.allclose(abs(np.diff(lat)), spacing, atol=0.0001, rtol=0)):
        raise ValueError(f'Latitude spacing is not {spacing} degrees')
    if len(lon) > 1 and (not np.allclose(abs(np.diff(np.rad2deg(np.unwrap(np.deg2rad(lon))))), spacing, atol=0.0001, rtol=0)):
        raise ValueError(f'Longitude spacing is not {spacing} degrees')

def city_counts_on_rain_grid(city_fraction, rain):
    """Count >0.2 impervious 0.01° cells in each native 0.1° rain cell."""
    rain_lat = np.asarray(rain.lat.values, float)
    rain_lon = normalize_lon_180(rain.lon.values)
    validate_regular_grid(rain_lat, rain_lon)
    all_lat = np.asarray(city_fraction.lat.values, float)
    all_lon = np.asarray(city_fraction.lon.values, float)
    row = np.flatnonzero((all_lat >= rain_lat.min() - .05) &
                         (all_lat < rain_lat.max() + .05))
    col = np.flatnonzero(np.any(
        abs((all_lon[:, None] - rain_lon[None, :] + 180) % 360 - 180) < .05,
        axis=1))
    out = np.zeros((len(rain_lat), len(rain_lon)), np.int32)
    if not len(row) or not len(col):
        return out
    part = city_fraction.isel(lat=slice(row[0], row[-1]+1), lon=col).load()
    urban = np.asarray(part.values > .2, np.uint8)
    lat = np.asarray(part.lat.values, float)
    lon = np.asarray(part.lon.values, float)
    for i, y in enumerate(rain_lat):
        r0 = np.searchsorted(lat, y-.05)
        r1 = np.searchsorted(lat, y+.05)
        for j, x in enumerate(rain_lon):
            in_cell = abs((lon-x+180)%360-180) < .05
            out[i,j] = urban[r0:r1, in_cell].sum(dtype=np.int32)
    return out

def grid_key(lat, lon):
    return (int(np.rint(float(lat) * 100)), int(np.rint(float(lon) * 100)))

def validate_gisa_netcdf():
    require_file(GISA_NC_FILE, 'GISA NetCDF')
    with open_source_netcdf(GISA_NC_FILE, decode_times=False, mask_and_scale=True) as ds:
        if 'impervious_fraction' not in ds:
            raise KeyError(f'Missing impervious_fraction: {GISA_NC_FILE}')
        if not {'year', 'lat', 'lon'}.issubset(ds.impervious_fraction.dims):
            raise ValueError(f'Unexpected GISA dimensions: {ds.impervious_fraction.dims}')
        missing = sorted(set(GISA_YEARS) - set(np.asarray(ds.year.values, dtype=int)))
        if missing:
            raise ValueError(f'GISA NetCDF missing years: {missing}')
        validate_regular_grid(ds.lat.values, ds.lon.values, spacing=0.01)
    print(f'Validated GISA NetCDF: {GISA_NC_FILE}')

def load_gisa_year(dataset, year):
    fraction = dataset.impervious_fraction.sel(year=int(year)).squeeze(drop=True)
    return fraction.assign_coords(lon=normalize_lon_180(fraction.lon.values)).sortby('lon').sortby('lat')

def regional_grid_template(mswep_lat, mswep_lon, region):
    west, east, south, north = MAP_EXTENTS[region]
    lat = np.sort(mswep_lat[(mswep_lat >= south) & (mswep_lat <= north)])
    lon180 = normalize_lon_180(mswep_lon)
    lon = np.sort(lon180[(lon180 >= west) & (lon180 <= east)])
    shape = (len(GISA_YEARS), len(lat), len(lon))
    return {'lat':lat, 'lon':lon,
            'lat_lookup':{grid_key(v,0)[0]:i for i,v in enumerate(lat)},
            'lon_lookup':{grid_key(0,v)[1]:j for j,v in enumerate(lon)},
            'annual':np.zeros(shape,np.float32)}

def mswep_grid_coordinates():
    sample = next(PRECIP_ROOT.glob('**/*.nc'), None)
    if sample is None:
        raise FileNotFoundError(f'No MSWEP file in {PRECIP_ROOT}')
    with open_source_netcdf(sample) as ds:
        return np.asarray(ds.lat.values, float), np.asarray(ds.lon.values, float)

def calculate_exposure(events_by_region, city_dataset):
    """Count exposure with frequency ≥ 1 using the original event-level weighting."""
    annual = pd.DataFrame({'YEAR': GISA_YEARS,
                           **{name: 0.0 for name in REGION_ORDER}})
    mswep_lat, mswep_lon = mswep_grid_coordinates()
    templates = {region: regional_grid_template(mswep_lat, mswep_lon, region) for region in ('East Asia', 'North America')}
    city_year = None
    city_count = None
    total = sum((frame.EVENT_ID.nunique() for frame in events_by_region.values()))
    completed = 0
    for region, points in events_by_region.items():
        for event_id, event in points.groupby('EVENT_ID', sort=False):
            year = int(event.EVENT_YEAR.iloc[0])
            if city_year != year:
                city_count = load_gisa_year(city_dataset, year)
                city_year = year
            repeat_counts = {}
            urban_counts = {}
            for point in event.itertuples(index=False):
                path = precipitation_file_for_time(point.ISO_TIME)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with open_source_netcdf(path, decode_times=True, mask_and_scale=True) as ds:
                    rain = subset_tc_box(select_exact_precipitation(ds, point.ISO_TIME, path), point.LAT, point.LON)
                    rain_values = np.asarray(rain.values, dtype=float)
                    lat = np.asarray(rain.lat.values, dtype=float)
                    lon = normalize_lon_180(rain.lon.values)
                    lon_grid, lat_grid = np.meshgrid(lon, lat)
                    city_numbers = city_counts_on_rain_grid(city_count, rain)
                    exposed = (haversine_km(point.LAT, point.LON, lat_grid, lon_grid) <= TC_RADIUS_KM) & (city_numbers > 0) & np.isfinite(rain_values) & (rain_values >= HEAVY_RAIN_MM_3H)
                    for i, j in zip(*np.where(exposed)):
                        key = grid_key(lat[i], lon[j])
                        repeat_counts[key] = repeat_counts.get(key, 0) + 1
                        number = int(city_numbers[i, j])
                        if key in urban_counts and urban_counts[key] != number:
                            raise ValueError(f'Inconsistent city count: {key}')
                        urban_counts[key] = number
            # Frequency ≥ 1: retain all exposed rain cells of this event.
            retained = {key: repetitions * urban_counts[key]
                        for key, repetitions in repeat_counts.items()
                        if repetitions >= MIN_EXPOSURE_FREQUENCY}
            row = year - GISA_YEAR_MIN
            annual.loc[row, region] += float(sum(retained.values()))
            if region in templates:
                template = templates[region]
                grid = template['annual']
                for (lat_key, lon_key), value in retained.items():
                    i = template['lat_lookup'].get(lat_key)
                    j = template['lon_lookup'].get(lon_key)
                    if i is not None and j is not None:
                        grid[row, i, j] += float(value)
            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == total:
                print(f'[pm100] GISA events {completed:,}/{total:,}')
    return annual, templates

def mann_kendall_p(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    s_stat = sum((np.sign(values[j] - values[i]) for i in range(n - 1) for j in range(i + 1, n)))
    _, ties = np.unique(values, return_counts=True)
    variance = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18
    if variance <= 0:
        return 1.0
    z = (s_stat - 1) / np.sqrt(variance) if s_stat > 0 else (s_stat + 1) / np.sqrt(variance) if s_stat < 0 else 0.0
    return float(2 * stats.norm.sf(abs(z)))

def trend_p_value(years, values):
    """Test monotonic change with MK or the OLS slope with a two-sided t test."""
    values = np.asarray(values, dtype=float)
    if np.ptp(values) == 0:
        return 1.0
    if SIGNIFICANCE_METHOD == 'mk':
        return mann_kendall_p(values)
    if SIGNIFICANCE_METHOD == 'ols_t':
        return float(stats.linregress(years, values).pvalue)
    raise ValueError(f'Unsupported significance method: {SIGNIFICANCE_METHOD}')

def trend_smoothing_metadata():
    half = TREND_SMOOTH_WINDOW // 2
    return {
        'trend_smoothing': TREND_SMOOTHING,
        'trend_window_years': TREND_SMOOTH_WINDOW,
        'trend_start_year': int(GISA_YEARS[half]),
        'trend_end_year': int(GISA_YEARS[-half-1]),
        'n_trend_years': len(GISA_YEARS)-2*half,
    }

def annual_trend(years, values):
    """Apply one centered five-year mean, then fit OLS and test with MK."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    window = TREND_SMOOTH_WINDOW
    half = window // 2
    if (years.ndim != 1 or values.ndim != 1 or years.shape != values.shape
            or len(years) < window+2 or not np.isfinite(years).all()
            or not np.isfinite(values).all() or not np.all(np.diff(years) == 1)):
        raise ValueError('Trend input requires finite consecutive annual values and at least three full smoothing windows')
    # Sum counts first, then divide once, preserving ties for the MK test.
    smooth = sum(values[offset:len(values)-window+1+offset]
                 for offset in range(window))/float(window)
    fit_years = years[half:-half]
    result = stats.linregress(fit_years, smooth)
    error = stats.t.ppf(.975, len(fit_years)-2)*result.stderr
    # Keep raw-year alignment. Incomplete end windows are not fitted or padded.
    smoothed = np.full(values.shape, np.nan, dtype=float)
    fitted = np.full(values.shape, np.nan, dtype=float)
    smoothed[half:-half] = smooth
    fitted[half:-half] = result.intercept+result.slope*fit_years
    return {
        'slope_decade': float(result.slope*10),
        'slope_ci95_lower': float((result.slope-error)*10),
        'slope_ci95_upper': float((result.slope+error)*10),
        'fitted': fitted, 'smoothed': smoothed,
        'trend_p': trend_p_value(fit_years, smooth),
        'trend_method': TREND_METHOD, 'significance_method': SIGNIFICANCE_METHOD,
        'trend_smoothing': TREND_SMOOTHING, 'trend_window_years': window,
        'trend_start_year': int(fit_years[0]), 'trend_end_year': int(fit_years[-1]),
        'n_trend_years': len(fit_years),
    }

def raw_annual_trend(years, values):
    """Fit OLS to unsmoothed annual exposures and apply the selected significance test."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    if (years.ndim != 1 or values.ndim != 1 or years.shape != values.shape
            or len(years) < 3 or not np.isfinite(years).all()
            or not np.isfinite(values).all() or not np.all(np.diff(years) == 1)):
        raise ValueError('Raw trend input requires at least three finite consecutive annual values')
    result = stats.linregress(years, values)
    error = stats.t.ppf(.975, len(years)-2)*result.stderr
    return {
        'slope_decade': float(result.slope*10),
        'slope_ci95_lower': float((result.slope-error)*10),
        'slope_ci95_upper': float((result.slope+error)*10),
        'fitted': result.intercept+result.slope*years,
        'trend_p': trend_p_value(years, values),
        'trend_method': 'ols', 'significance_method': SIGNIFICANCE_METHOD,
        'trend_smoothing': 'none', 'trend_window_years': 1,
        'trend_start_year': int(years[0]), 'trend_end_year': int(years[-1]),
        'n_trend_years': len(years),
    }

def ols_mean_confidence(years, values, evaluation_years=None):
    """Return the lower/upper pointwise 95% OLS mean-fit confidence bounds.

    The fitted sample determines n and residual variance. Evaluation years only
    specify where the bounds are evaluated; they do not change the sample size.
    These are mean-fit confidence intervals, not prediction intervals.
    For the five-year-mean series, pass only its complete 33 fitted years and
    pad the two incomplete edge years with NaN outside this function.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    if (years.ndim != 1 or values.ndim != 1 or years.shape != values.shape
            or len(years) < 3 or not np.isfinite(years).all()
            or not np.isfinite(values).all() or not np.all(np.diff(years) > 0)):
        raise ValueError('OLS confidence bounds require at least three finite observations and increasing years')
    evaluation = years if evaluation_years is None else np.asarray(evaluation_years, dtype=float)
    if evaluation.ndim != 1 or not np.isfinite(evaluation).all():
        raise ValueError('Evaluation years must be a finite one-dimensional array')
    result = stats.linregress(years, values)
    centered = years-years.mean()
    sxx = np.sum(centered**2)
    fitted_sample = values.mean()+result.slope*centered
    residual_variance = np.sum((values-fitted_sample)**2)/(len(years)-2)
    fitted_evaluation = values.mean()+result.slope*(evaluation-years.mean())
    standard_error = np.sqrt(residual_variance*(
        1.0/len(years)+(evaluation-years.mean())**2/sxx
    ))
    half_width = stats.t.ppf(.975, len(years)-2)*standard_error
    return fitted_evaluation-half_width, fitted_evaluation+half_width

def prepare_coastal_min1_series(annual_table):
    """Select the coastal minimum-one series without altering the annual cache table."""
    required = {'distance_case', 'repeat_threshold', 'year', 'exposure'}
    missing = sorted(required-set(annual_table.columns))
    if missing:
        raise ValueError(f'Annual exposure table missing columns: {missing}')
    distance = annual_table['distance_case'].astype(str).str.strip()
    threshold = pd.to_numeric(annual_table['repeat_threshold'], errors='coerce')
    selected = annual_table.loc[distance.eq('pm100') & threshold.eq(1),
                                ['year', 'exposure']].copy()
    for column in ['year', 'exposure']:
        selected[column] = pd.to_numeric(selected[column], errors='raise')
        if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
            raise ValueError(f'Coastal minimum-one series has nonfinite {column} values')
    years = selected['year'].to_numpy(dtype=float)
    if not np.equal(years, np.floor(years)).all():
        raise ValueError('Coastal minimum-one years must be integers')
    if selected['year'].duplicated().any():
        raise ValueError('Coastal minimum-one series contains duplicate years')
    if selected['exposure'].lt(0).any():
        raise ValueError('Coastal minimum-one series contains negative exposure')
    selected = selected.sort_values('year').reset_index(drop=True)
    years = selected['year'].to_numpy(dtype=int, copy=True)
    expected = np.asarray(GISA_YEARS, dtype=int)
    if not np.array_equal(years, expected):
        missing_years = sorted(set(expected)-set(years))
        extra_years = sorted(set(years)-set(expected))
        raise ValueError('Coastal minimum-one series must contain each study year once: '
                         f'missing={missing_years}, extra={extra_years}')
    raw = selected['exposure'].to_numpy(dtype=float, copy=True)
    raw_fit = raw_annual_trend(years, raw)
    smooth_fit = annual_trend(years, raw)
    raw_fit['fitted_ci95_lower'], raw_fit['fitted_ci95_upper'] = ols_mean_confidence(years, raw)
    valid = np.isfinite(smooth_fit['smoothed'])
    lower, upper = ols_mean_confidence(years[valid], smooth_fit['smoothed'][valid])
    for key, bounds in [('fitted_ci95_lower', lower), ('fitted_ci95_upper', upper)]:
        smooth_fit[key] = np.full(raw.shape, np.nan, dtype=float)
        smooth_fit[key][valid] = bounds
    return {'years': years, 'raw': raw, 'raw_fit': raw_fit, 'smooth_fit': smooth_fit}

def spatial_trend(annual_grid):
    annual_grid = np.asarray(annual_grid, dtype=float)
    if annual_grid.ndim != 3 or annual_grid.shape[0] != len(GISA_YEARS):
        raise ValueError('Annual map dimensions must be (GISA year, lat, lon)')
    # Preserve the centered five-year mean used in the original maps.
    smooth = sum(annual_grid[offset:len(annual_grid)-4+offset] for offset in range(5))/5.0
    years = GISA_YEARS[2:-2].astype(float)
    active = np.isfinite(smooth).all(axis=0) & (np.sum(annual_grid, axis=0)>0)
    shape = annual_grid.shape[1:]
    slopes = np.full(shape, np.nan, dtype=float)
    p = np.full(shape, np.nan, dtype=np.float32)
    flat = smooth.reshape(len(years), -1)
    indices = np.flatnonzero(active.ravel())
    if TREND_METHOD == 'sen':
        earlier, later = np.triu_indices(len(years), k=1)
        elapsed = (years[later]-years[earlier])[:, None]
        for start in range(0, len(indices), SEN_CHUNK_SIZE):
            chunk = indices[start:start+SEN_CHUNK_SIZE]
            series = flat[:, chunk]
            pair_slopes = (series[later]-series[earlier])/elapsed
            slopes.ravel()[chunk] = np.median(pair_slopes, axis=0)*10
    elif TREND_METHOD == 'ols':
        centered = years-years.mean()
        slopes.ravel()[indices] = (centered @ flat[:, indices])/np.sum(centered**2)*10
    else:
        raise ValueError(f'Unsupported trend method: {TREND_METHOD}')
    for index in indices:
        p.ravel()[index] = trend_p_value(years, flat[:, index])
    significant = p < .05
    return np.where(significant, slopes, np.nan), p, significant

def trend_summary(regional_fits, sensitivity):
    """Export estimator, CI bounds, and the actual significance method for every panel."""
    bars = sensitivity.copy()
    bars['panel'] = 'A'
    bars['record'] = 'repeated_exposure_trend'
    rows = []
    for panel, region in [('B', 'East Asia'), ('C', 'North America')]:
        fit = regional_fits[region]
        rows.append({
            'panel': panel, 'record': 'regional_trend',
            'region': region, 'distance_case': 'pm100', 'repeat_threshold': 1,
            **{name: fit[name] for name in ['slope_decade', 'slope_ci95_lower',
                                           'slope_ci95_upper', 'trend_p']},
            'trend_method': TREND_METHOD, 'significance_method': SIGNIFICANCE_METHOD,
            **trend_smoothing_metadata(),
        })
    return pd.concat([bars, pd.DataFrame(rows)], ignore_index=True)

def formal_trend_summary(regional_fits, sensitivity, coastal_series):
    """Use display panel numbers without changing the legacy cache summary."""
    legacy = trend_summary(regional_fits, sensitivity)
    formal = legacy.copy()
    formal['panel'] = formal['panel'].map({'A': 'B', 'B': 'C', 'C': 'D'})
    rows = []
    keys = ['slope_decade', 'slope_ci95_lower', 'slope_ci95_upper', 'trend_p',
            'trend_method', 'significance_method', 'trend_smoothing',
            'trend_window_years', 'trend_start_year', 'trend_end_year', 'n_trend_years']
    for name, key in [('raw', 'raw_fit'), ('smooth5', 'smooth_fit')]:
        fit = coastal_series[key]
        rows.append({'panel': 'A', 'record': 'annual_exposure_trend',
                     'series': name, 'region': 'NH (>20°N)',
                     'distance_case': 'pm100', 'repeat_threshold': 1,
                     **{field: fit[field] for field in keys}})
    return pd.concat([pd.DataFrame(rows), formal], ignore_index=True)

def load_exposure_tracks():
    required = ['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'CAL_DIST']
    frame = pd.read_csv(TRACK_FILE, usecols=required,
                        keep_default_na=False, low_memory=False)
    frame['SID'] = frame.SID.astype(str).str.strip()
    frame['ISO_TIME'] = pd.to_datetime(frame.ISO_TIME, errors='coerce')
    for column in ['LAT', 'LON', 'USA_WIND', 'CAL_DIST']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[
        frame.SID.ne('') & frame.ISO_TIME.notna()
        & frame.ISO_TIME.dt.year.between(1980, 2024)
    ].dropna(subset=['LAT', 'LON', 'CAL_DIST']).copy()
    frame = (frame.sort_values(['SID', 'ISO_TIME'])
                  .drop_duplicates(['SID', 'ISO_TIME']))
    frame['LON'] = normalize_lon_180(frame.LON)
    frame['YEAR'] = frame.ISO_TIME.dt.year.astype(int)
    sid_lmi = frame.groupby('SID').USA_WIND.max()
    frame['LMI_ELIGIBLE'] = frame.SID.map(sid_lmi).ge(WIND_LIMIT_KT)
    return frame

def identify_nh_cltc_events(tracks, distance_key):
    if distance_key == 'pm100':
        distance_mask = tracks.CAL_DIST.abs() < 100.0
    elif distance_key == 'm300_m100':
        distance_mask = tracks.CAL_DIST.between(
            -300.0, -100.0, inclusive='both'
        )
    elif distance_key == 'p100_p300':
        distance_mask = tracks.CAL_DIST.between(
            100.0, 300.0, inclusive='both'
        )
    else:
        raise ValueError(distance_key)

    selected = tracks.loc[
        tracks.LMI_ELIGIBLE
        & tracks.YEAR.between(GISA_YEAR_MIN, GISA_YEAR_MAX)
        & tracks.LAT.gt(20.0)
        & distance_mask
    ]
    blocks = []
    for sid, storm in selected.groupby('SID', sort=False):
        storm = storm.sort_values('ISO_TIME')
        block_ids = storm.ISO_TIME.diff().ne(
            pd.Timedelta(hours=TIME_STEP_HOURS)
        ).cumsum()
        for block_number, (_, block) in enumerate(
            storm.groupby(block_ids, sort=False), start=1
        ):
            if len(block) < CLTC_MIN_POINTS:
                continue
            block = block.copy()
            block['EVENT_ID'] = f'{distance_key}_{sid}_E{block_number:05d}'
            block['EVENT_YEAR'] = int(block.ISO_TIME.iloc[0].year)
            blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=[*tracks.columns, 'EVENT_ID', 'EVENT_YEAR'])
    return pd.concat(blocks, ignore_index=True)

def record_consecutive_exposure(state, event_id, grid_key, timestamp, urban_count,
                                step_hours=3, max_gap_hours=0):
    """Record exposed observations, allowing a bounded gap of unexposed time steps.

    Call only when the rainfall, TC-radius and urban-cell exposure conditions hold.
    At 3-hour resolution, max_gap_hours=3 permits one intervening unexposed
    observation, so exposed observations up to 6 hours apart remain in one run.
    Only actual exposures are counted. Every gap must satisfy this limit, and
    observations must lie on the time-step grid. Longer gaps close the run.
    Use fresh state for each event; explicit event keys prevent cross-event joins.
    """
    time = pd.Timestamp(timestamp)
    if pd.isna(time):
        raise ValueError('Exposure timestamp must not be missing')
    if int(urban_count) != urban_count or urban_count <= 0:
        raise ValueError('urban_count must be a positive integer')
    step = pd.Timedelta(hours=step_hours)
    if step <= pd.Timedelta(0):
        raise ValueError('step_hours must be positive')
    max_gap = pd.Timedelta(hours=max_gap_hours)
    if pd.isna(max_gap) or max_gap < pd.Timedelta(0) or max_gap % step:
        raise ValueError('max_gap_hours must be a nonnegative multiple of step_hours')
    key = (event_id, grid_key)
    number = int(urban_count)
    if key not in state:
        state[key] = {
            'last_time': time, 'current_run': 1, 'completed_runs': [],
            'urban_count': number, 'step': step, 'max_gap': max_gap,
        }
        return
    entry = state[key]
    if entry['urban_count'] != number:
        raise ValueError(f'Inconsistent urban count for event/grid {key}')
    if entry['step'] != step:
        raise ValueError(f'Inconsistent time step for event/grid {key}')
    if entry['max_gap'] != max_gap:
        raise ValueError(f'Inconsistent allowed gap for event/grid {key}')
    delta = time-entry['last_time']
    if delta <= pd.Timedelta(0):
        raise ValueError(f'Duplicate or out-of-order exposure time for {key}: {time}')
    if delta % step == pd.Timedelta(0) and delta <= step + max_gap:
        entry['current_run'] += 1
    else:
        entry['completed_runs'].append(entry['current_run'])
        entry['current_run'] = 1
    entry['last_time'] = time

def summarize_consecutive_exposure(state, event_id, thresholds, *, policy):
    """Return urban-cell exposure counts under the explicitly selected policy.

    qualifying_runs_only: sum all observations in runs of length >= threshold.
    qualified_event_all: if any run qualifies, count every exposed observation
    for that event/grid, including exposures outside the qualifying run.

    Counts include the entire qualifying run, not only observations after the
    threshold is reached, and do not count overlapping windows multiple times.
    This is read-only: the unfinished final run is included without mutating it.
    """
    if policy not in {'qualifying_runs_only', 'qualified_event_all'}:
        raise ValueError(f'Unknown consecutive exposure policy: {policy}')
    thresholds = tuple(thresholds)
    if (not thresholds or len(set(thresholds)) != len(thresholds)
            or any(int(n) != n or n <= 0 for n in thresholds)):
        raise ValueError('thresholds must contain unique positive integers')
    totals = {int(n): 0 for n in thresholds}
    for (entry_event, _), entry in state.items():
        if entry_event != event_id:
            continue
        lengths = (*entry['completed_runs'], entry['current_run'])
        for threshold in totals:
            if policy == 'qualifying_runs_only':
                count = sum(length for length in lengths if length >= threshold)
            else:
                count = sum(lengths) if max(lengths) >= threshold else 0
            totals[threshold] += count*entry['urban_count']
    return totals

def calculate_sensitivity_trends(city_dataset):
    tracks = load_exposure_tracks()
    rows = []
    annual_rows = []
    for distance_key, _, _ in EXPOSURE_DISTANCE_CASES:
        points = identify_nh_cltc_events(tracks, distance_key)
        annual = {
            threshold: np.zeros(len(GISA_YEARS), dtype=float)
            for threshold in REPEAT_THRESHOLDS
        }
        grouped = points.groupby('EVENT_ID', sort=False)
        total = grouped.ngroups
        print(f'[{distance_key}] {total:,} CLTC events')
        city_year = None
        city_fraction = None

        for completed, (event_id, event) in enumerate(grouped, start=1):
            year = int(event.EVENT_YEAR.iloc[0])
            if city_year != year:
                city_fraction = load_gisa_year(city_dataset, year)
                city_year = year
            run_state = {}

            for point in event.itertuples(index=False):
                path = precipitation_file_for_time(point.ISO_TIME)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with open_source_netcdf(path, decode_times=True,
                                     mask_and_scale=True) as dataset:
                    rain = subset_tc_box(
                        select_exact_precipitation(
                            dataset, point.ISO_TIME, path
                        ),
                        point.LAT, point.LON,
                    )
                    rain_values = np.asarray(rain.values, dtype=float)
                    rain_lat = np.asarray(rain.lat.values, dtype=float)
                    rain_lon = normalize_lon_180(rain.lon.values)
                    lon_grid, lat_grid = np.meshgrid(rain_lon, rain_lat)
                    city_counts = city_counts_on_rain_grid(
                        city_fraction, rain
                    )
                    exposed = (
                        (haversine_km(point.LAT, point.LON,
                                      lat_grid, lon_grid) <= TC_RADIUS_KM)
                        & (city_counts > 0)
                        & np.isfinite(rain_values)
                        & (rain_values >= HEAVY_RAIN_MM_3H)
                    )
                    for i, j in zip(*np.where(exposed)):
                        key = grid_key(rain_lat[i], rain_lon[j])
                        record_consecutive_exposure(
                            run_state, event_id, key, point.ISO_TIME,
                            int(city_counts[i, j]), step_hours=TIME_STEP_HOURS,
                            max_gap_hours=MAX_EXPOSURE_GAP_HOURS,
                        )

            # Strictly consecutive runs never span different events or grids.
            # Count exposures only in runs reaching N; exclude all shorter runs.
            event_exposure = summarize_consecutive_exposure(
                run_state, event_id, REPEAT_THRESHOLDS,
                policy=CONSECUTIVE_COUNTING_POLICY,
            )
            year_index = year - GISA_YEAR_MIN
            for threshold in REPEAT_THRESHOLDS:
                annual[threshold][year_index] += event_exposure[threshold]
            if completed % 20 == 0 or completed == total:
                print(f'[{distance_key}] processed {completed:,}/{total:,}')

        for threshold in REPEAT_THRESHOLDS:
            trend = annual_trend(GISA_YEARS, annual[threshold])
            for index, year in enumerate(GISA_YEARS):
                annual_rows.append({
                    'record': 'M4A_consecutive_annual_exposure',
                    'region': 'NH (>20°N)',
                    'distance_case': distance_key,
                    'repeat_threshold': threshold,
                    'exposure_rule': EXPOSURE_RULE,
                    'counting_policy': CONSECUTIVE_COUNTING_POLICY,
                    'time_step_hours': TIME_STEP_HOURS,
                    'max_gap_hours': MAX_EXPOSURE_GAP_HOURS,
                    'year': int(year),
                    'exposure': float(annual[threshold][index]),
                    'fitted': float(trend['fitted'][index]),
                    'exposure_smooth5': float(trend['smoothed'][index]),
                    **trend_smoothing_metadata(),
                    'trend_method': TREND_METHOD,
                    'significance_method': SIGNIFICANCE_METHOD,
                })
            rows.append({
                'record': 'S10_sensitivity',
                'panel': 'C',
                'region': 'NH (>20°N)',
                'distance_case': distance_key,
                'repeat_threshold': threshold,
                'exposure_rule': EXPOSURE_RULE,
                'counting_policy': CONSECUTIVE_COUNTING_POLICY,
                'time_step_hours': TIME_STEP_HOURS,
                'max_gap_hours': MAX_EXPOSURE_GAP_HOURS,
                **trend_smoothing_metadata(),
                'slope_decade': trend['slope_decade'],
                'slope_ci95_lower': trend['slope_ci95_lower'],
                'slope_ci95_upper': trend['slope_ci95_upper'],
                'trend_method': TREND_METHOD,
                'significance_method': SIGNIFICANCE_METHOD,
                'trend_p': trend['trend_p'],
            })
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)

def validate_series(table):
    missing = set(SERIES_COLUMNS)-set(table.columns)
    if missing:
        raise ValueError(f'M4A series missing columns: {sorted(missing)}')
    table = table[list(SERIES_COLUMNS)].copy()
    for column in SERIES_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    table = table.sort_values('year').reset_index(drop=True)
    if not np.array_equal(table.year.values, YEARS):
        raise ValueError('M4A requires each year from 1985 to 2021 exactly once')
    for column in SERIES_COLUMNS[1:5]:
        if not np.isfinite(table[column]).all():
            raise ValueError(f'M4A has missing/nonfinite raw values: {column}')
    for column in SERIES_COLUMNS[5:]:
        if not np.isfinite(table[column].iloc[2:-2]).all():
            raise ValueError(f'M4A has missing/nonfinite complete five-year means: {column}')
        if not table[column].iloc[[0, 1, -2, -1]].isna().all():
            raise ValueError('Five-year endpoints must remain NaN; do not pad or smooth again')
    if table.exposure_raw.lt(0).any() or table.exposure_smooth5.dropna().lt(0).any():
        raise ValueError('Exposure counts must be nonnegative')
    for prefix in ('raw', 'smooth5'):
        lower, fit, upper = (table[f'{prefix}_ols_ci95_lower'],
                             table[f'{prefix}_ols_fitted'], table[f'{prefix}_ols_ci95_upper'])
        if ((lower > fit) | (fit > upper)).any():
            raise ValueError(f'Invalid mean-fit confidence band: {prefix}')
    return table

def validate_statistics(table):
    missing = set(STAT_COLUMNS)-set(table.columns)
    if missing:
        raise ValueError(f'M4 statistics missing columns: {sorted(missing)}')
    table = table[list(STAT_COLUMNS)].copy()
    for name in ('panel', 'series', 'distance_case'):
        table[name] = table[name].fillna('').astype(str).str.strip()
    for name in STAT_COLUMNS[3:]:
        table[name] = pd.to_numeric(table[name], errors='raise')
    if len(table) != 13 or table.groupby('panel').size().to_dict() != {'A': 2, 'B': 9, 'C': 1, 'D': 1}:
        raise ValueError('Expected A:2, B:9, C:1 and D:1 trend records')
    a, b = table.loc[table.panel.eq('A')], table.loc[table.panel.eq('B')]
    if set(a.series) != {'raw', 'smooth5'}:
        raise ValueError('M4A requires separate raw and smooth5 trend statistics')
    expected = {(distance, threshold) for distance in DISTANCE_KEYS for threshold in (1, 2, 3)}
    if set(zip(b.distance_case, b.repeat_threshold)) != expected:
        raise ValueError('M4B must contain three distance bands and thresholds 1-3')
    if table[['slope_decade', 'trend_p']].isna().any().any():
        raise ValueError('Every trend requires a finite slope and P value')
    if not np.isfinite(table[['slope_decade', 'trend_p']].to_numpy(float)).all():
        raise ValueError('Nonfinite slope or P value')
    if not table.trend_p.between(0, 1).all():
        raise ValueError('P values must be within [0, 1]')
    ci = table.loc[~table.panel.eq('A'), ['slope_ci95_lower', 'slope_decade', 'slope_ci95_upper']]
    if not np.isfinite(ci.to_numpy(float)).all() or (np.diff(ci.to_numpy(float), axis=1) < 0).any():
        raise ValueError('Invalid trend confidence intervals')
    return table

def write_preprocessed(output_dir, series, statistics, maps):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in [('FigM4A_timeseries.csv', series), ('FigM4_plot_statistics.csv', statistics)]:
        path = output_dir/filename
        temporary = path.with_suffix('.csv.tmp')
        try:
            table.to_csv(temporary, index=False, encoding='utf-8', float_format='%.17g')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f'Saved: {path}')
    for panel, dataset in maps.items():
        path = output_dir/f'FigM4{panel}_trend.nc'
        temporary = path.with_suffix('.nc.tmp')
        try:
            dataset.to_netcdf(temporary, engine='h5netcdf', format='NETCDF4',
                             encoding={name: {'zlib': True, 'complevel': 4, 'shuffle': True}
                                       for name in ('trend_of_exposure', 'trend_p')})
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f'Saved: {path}')



def configure_inputs(raw_data_dir, mswep_dir=None):
    """Resolve source paths without reading or changing any source files."""
    global TRACK_FILE, GISA_NC_FILE, PRECIP_ROOT
    raw_data_dir = Path(raw_data_dir).expanduser().resolve()
    TRACK_FILE = raw_data_dir / TRACK_RELATIVE
    GISA_NC_FILE = raw_data_dir / GISA_RELATIVE
    PRECIP_ROOT = (Path(mswep_dir).expanduser().resolve() if mswep_dir is not None
                   else raw_data_dir / 'MSWEP')


def check_inputs():
    missing = [str(path) for path in (TRACK_FILE, GISA_NC_FILE) if not path.is_file()]
    if not PRECIP_ROOT.is_dir():
        missing.append(str(PRECIP_ROOT) + ' (MSWEP directory)')
    elif next(PRECIP_ROOT.glob('*/*.nc'), None) is None:
        missing.append(str(PRECIP_ROOT / 'YYYY/YYYYDDD.HH.nc') + ' (MSWEP files)')
    if missing:
        raise FileNotFoundError('Missing source inputs:\n  ' + '\n  '.join(missing)
            + '\nProvide the original MSWEP files with --mswep-dir; '
              'existing result files cannot replace precipitation inputs.')


def calculate_regional_outputs(city_dataset):
    """Recompute regional annual exposures and trends without result caches."""
    tracks = load_tracks()
    events = {region: identify_cltc_events(tracks, region)
              for region in ('East Asia', 'North America')}
    annual, templates = calculate_exposure(events, city_dataset)
    spatial, regional_fits = {}, {}
    for region in ('East Asia', 'North America'):
        template = templates[region]
        trend, p, significant = spatial_trend(template['annual'])
        spatial[region] = (trend, p, significant, template['lat'], template['lon'])
        regional_fits[region] = annual_trend(annual.YEAR, annual[region])
    return spatial, regional_fits


def prepare_plot_outputs(spatial, regional_fits, sensitivity, annual_exposure):
    """Build the existing plotting schema from newly calculated analysis outputs."""
    coastal = prepare_coastal_min1_series(annual_exposure)
    series = validate_series(pd.DataFrame({
        'year': coastal['years'],
        'exposure_raw': coastal['raw'],
        'raw_ols_fitted': coastal['raw_fit']['fitted'],
        'raw_ols_ci95_lower': coastal['raw_fit']['fitted_ci95_lower'],
        'raw_ols_ci95_upper': coastal['raw_fit']['fitted_ci95_upper'],
        'exposure_smooth5': coastal['smooth_fit']['smoothed'],
        'smooth5_ols_fitted': coastal['smooth_fit']['fitted'],
        'smooth5_ols_ci95_lower': coastal['smooth_fit']['fitted_ci95_lower'],
        'smooth5_ols_ci95_upper': coastal['smooth_fit']['fitted_ci95_upper'],
    }))
    statistics = formal_trend_summary(regional_fits, sensitivity, coastal)
    statistics = statistics[list(STAT_COLUMNS)].copy()
    statistics.loc[statistics.panel.eq('A'),
                   ['slope_ci95_lower', 'slope_ci95_upper']] = np.nan
    statistics.loc[~statistics.panel.eq('B'),
                   ['distance_case', 'repeat_threshold']] = np.nan
    statistics = validate_statistics(statistics)
    maps = {}
    for panel, region in REGIONS.items():
        trend, p, _, lat, lon = spatial[region]
        maps[panel] = xr.Dataset(
            data_vars={
                'trend_of_exposure': (('lat', 'lon'), trend.astype(np.float32), {
                    'units': 'urban-grid 3 h decade-1',
                    'description': 'ols trend; only mk P < 0.05 cells retained'}),
                'trend_p': (('lat', 'lon'), p.astype(np.float32), {
                    'long_name': 'Mann-Kendall P value for the exposure trend',
                    'significance_method': 'mk',
                    'description': 'Original floating-point P values; not significance flags'}),
            },
            coords={'lat': lat, 'lon': lon},
            attrs={
                'region': region, 'panel': panel, 'years': '1985-2021',
                'trend_method': TREND_METHOD, 'significance_method': SIGNIFICANCE_METHOD,
                **trend_smoothing_metadata(),
                'significance_threshold': .05, 'source_trend_already_masked': 1,
                'source_trend_mask': 'Only source slopes with MK P < 0.05 are available',
                'spatial_smoothing': 'none', 'source_grid': 'native MSWEP 0.1 degree',
            })
    return series, statistics, maps


EXPOSURE_CSV_COLUMNS = {
    'M4_event_points.csv': ('SELECTOR', 'EVENT_ID', 'SID', 'EVENT_YEAR', 'ISO_TIME',
                            'TC_LAT', 'TC_LON', 'EXPOSED_GRID_COUNT', 'URBAN_GRID_EXPOSURE'),
    'M4_exposure_occurrences.csv': ('SELECTOR', 'EVENT_ID', 'ISO_TIME',
                                    'LAT_KEY', 'LON_KEY', 'URBAN_COUNT'),
    'M4_grid_coordinates.csv': ('REGION', 'AXIS', 'INDEX', 'AXIS_LENGTH', 'VALUE'),
}


def selected_exposure_events():
    """Yield the original five selections in deterministic event/point order."""
    tracks = load_exposure_tracks()
    for distance_key, _, _ in EXPOSURE_DISTANCE_CASES:
        points = identify_nh_cltc_events(tracks, distance_key)
        for event_id, event in points.groupby('EVENT_ID', sort=False):
            yield f'nh:{distance_key}', event_id, event
    regional_tracks = load_tracks()
    for region in ('East Asia', 'North America'):
        points = identify_cltc_events(regional_tracks, region)
        for event_id, event in points.groupby('EVENT_ID', sort=False):
            yield f'region:{region}', event_id, event


def exposed_grid_occurrences(city_fraction, point):
    """Read one MSWEP time and return the original exposed-grid observations."""
    path = precipitation_file_for_time(point.ISO_TIME)
    require_file(path, 'MSWEP time file')
    with open_source_netcdf(path, decode_times=True, mask_and_scale=True) as dataset:
        rain = subset_tc_box(select_exact_precipitation(dataset, point.ISO_TIME, path),
                             point.LAT, point.LON)
        values = np.asarray(rain.values, dtype=float)
        lat = np.asarray(rain.lat.values, dtype=float)
        lon = normalize_lon_180(rain.lon.values)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        city_counts = city_counts_on_rain_grid(city_fraction, rain)
        exposed = ((haversine_km(point.LAT, point.LON, lat_grid, lon_grid) <= TC_RADIUS_KM)
                   & (city_counts > 0) & np.isfinite(values) & (values >= HEAVY_RAIN_MM_3H))
        return [(grid_key(lat[i], lon[j]), int(city_counts[i, j]))
                for i, j in zip(*np.where(exposed))]


def export_exposure_csv(output_dir, *, overwrite=False):
    """Stream time-by-time urban-grid exposures and their grid-level detail.

    URBAN_GRID_EXPOSURE is the sum of URBAN_COUNT at each event observation:
    the number of exposed 0.01-degree urban cells, not the number of MSWEP
    cells. EXPOSED_GRID_COUNT is the number of accompanying occurrence rows.
    Both counts are zero for an observation with no exposed urban grid.

    Call configure_inputs() first. All three temporary CSVs are complete before
    any target is replaced. Existing M4 CSVs require overwrite=True explicitly.
    Other files in output_dir, including S1D archives, are left untouched.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    targets = [output_dir / name for name in EXPOSURE_CSV_COLUMNS]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError('M4 CSV output already exists; pass overwrite=True explicitly.')
    check_inputs()
    validate_gisa_netcdf()
    mswep_lat, mswep_lon = mswep_grid_coordinates()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {'event_points': 0, 'exposure_occurrences': 0, 'events': 0}
    with tempfile.TemporaryDirectory(prefix='.m4_export_', dir=output_dir) as directory:
        staging = Path(directory)
        with (staging / 'M4_grid_coordinates.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(EXPOSURE_CSV_COLUMNS['M4_grid_coordinates.csv'])
            for region in ('East Asia', 'North America'):
                west, east, south, north = MAP_EXTENTS[region]
                lat = np.sort(mswep_lat[(mswep_lat >= south) & (mswep_lat <= north)])
                lon180 = normalize_lon_180(mswep_lon)
                lon = np.sort(lon180[(lon180 >= west) & (lon180 <= east)])
                for axis, values in (('lat', lat), ('lon', lon)):
                    for index, value in enumerate(values):
                        writer.writerow((region, axis, index, len(values), float(value)))
        with (staging / 'M4_event_points.csv').open('w', newline='', encoding='utf-8') as points_handle, \
             (staging / 'M4_exposure_occurrences.csv').open('w', newline='', encoding='utf-8') as occurrences_handle, \
             open_source_netcdf(GISA_NC_FILE, decode_times=False, mask_and_scale=True) as city:
            point_writer, occurrence_writer = csv.writer(points_handle), csv.writer(occurrences_handle)
            point_writer.writerow(EXPOSURE_CSV_COLUMNS['M4_event_points.csv'])
            occurrence_writer.writerow(EXPOSURE_CSV_COLUMNS['M4_exposure_occurrences.csv'])
            city_year, city_fraction = None, None
            for selector, event_id, event in selected_exposure_events():
                year = int(event.EVENT_YEAR.iloc[0])
                if city_year != year:
                    city_fraction, city_year = load_gisa_year(city, year), year
                urban_counts = {}
                for point in event.itertuples(index=False):
                    occurrences = exposed_grid_occurrences(city_fraction, point)
                    timestamp = pd.Timestamp(point.ISO_TIME).isoformat()
                    point_writer.writerow((selector, event_id, point.SID, year, timestamp,
                                           float(point.LAT), float(point.LON), len(occurrences),
                                           sum(number for _, number in occurrences)))
                    seen = set()
                    for key, urban_count in occurrences:
                        if key in seen:
                            raise ValueError(f'Duplicate exposed grid at {selector}/{event_id}/{timestamp}: {key}')
                        seen.add(key)
                        if key in urban_counts and urban_counts[key] != urban_count:
                            raise ValueError(f'Inconsistent urban count within event: {selector}/{event_id}/{key}')
                        urban_counts[key] = urban_count
                        occurrence_writer.writerow((selector, event_id, timestamp, *key, urban_count))
                    counts['event_points'] += 1
                    counts['exposure_occurrences'] += len(occurrences)
                counts['events'] += 1
                if counts['events'] % PROGRESS_EVERY == 0:
                    print(f"Exported {counts['events']:,} events, {counts['event_points']:,} points.", flush=True)
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError('M4 CSV targets appeared during export; no target was replaced.')
        for target in targets:
            (staging / target.name).replace(target)
    print(f"Saved M4 CSVs: {counts['event_points']:,} points and "
          f"{counts['exposure_occurrences']:,} grid occurrences in {output_dir}.", flush=True)
    return counts


def exposure_csv_reader(handle, filename):
    reader = csv.DictReader(handle)
    if reader.fieldnames != list(EXPOSURE_CSV_COLUMNS[filename]):
        raise ValueError(f'{filename}: unexpected CSV columns: {reader.fieldnames}')
    def checked_rows():
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{filename}: malformed CSV row at line {reader.line_num}')
            yield row
    return checked_rows()


def csv_integer(value, label, minimum=None):
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{label}: expected an integer, got {value!r}') from error
    if minimum is not None and number < minimum:
        raise ValueError(f'{label}: expected at least {minimum}, got {number}')
    return number


def read_exposure_grid_coordinates(path):
    axes = {(region, axis): [] for region in MAP_EXTENTS for axis in ('lat', 'lon')}
    axis_lengths = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for row in exposure_csv_reader(handle, path.name):
            key = row['REGION'], row['AXIS']
            if key not in axes:
                raise ValueError(f'Unknown grid coordinate group: {key}')
            axis_length = csv_integer(row['AXIS_LENGTH'], 'coordinate AXIS_LENGTH', 1)
            if key in axis_lengths and axis_lengths[key] != axis_length:
                raise ValueError(f'Inconsistent coordinate AXIS_LENGTH: {key}')
            axis_lengths[key] = axis_length
            index = csv_integer(row['INDEX'], 'coordinate INDEX', 0)
            if index != len(axes[key]) or index >= axis_length:
                raise ValueError(f'Duplicate, missing or out-of-order coordinate INDEX: {key}/{index}')
            value = float(row['VALUE'])
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite grid coordinate: {key}/{index}')
            axes[key].append(value)
    for key, values in axes.items():
        if key not in axis_lengths or len(values) != axis_lengths[key]:
            raise ValueError(f'Incomplete regional grid axis: {key}; '
                             f'read {len(values)}, expected {axis_lengths.get(key, "missing AXIS_LENGTH")}')
    templates = {}
    for region, (west, east, south, north) in MAP_EXTENTS.items():
        lat, lon = (np.asarray(axes[(region, axis)], dtype=float) for axis in ('lat', 'lon'))
        if not len(lat) or not len(lon):
            raise ValueError(f'Missing regional grid axis: {region}')
        validate_regular_grid(lat, lon)
        if (np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0)
                or lat.min() < south or lat.max() > north or lon.min() < west or lon.max() > east):
            raise ValueError(f'Invalid or out-of-bounds regional grid coordinates: {region}')
        lat_keys = [grid_key(v, 0)[0] for v in lat]
        lon_keys = [grid_key(0, v)[1] for v in lon]
        for axis, values, keys in (('lat', lat, lat_keys), ('lon', lon, lon_keys)):
            if (len(set(keys)) != len(values)
                    or not np.allclose(np.asarray(keys) / 100.0, values, atol=0.0001, rtol=0)):
                raise ValueError(f'Grid coordinates do not map uniquely to native grid keys: {region}/{axis}')
        templates[region] = {
            'lat': lat, 'lon': lon,
            'lat_lookup': {key: i for i, key in enumerate(lat_keys)},
            'lon_lookup': {key: i for i, key in enumerate(lon_keys)},
            'lat_key_range': (lat_keys[0], lat_keys[-1]),
            'lon_key_range': (lon_keys[0], lon_keys[-1]),
            'annual': np.zeros((len(GISA_YEARS), len(lat), len(lon)), dtype=np.float32),
        }
    return templates


def sensitivity_from_annual_counts(annual_by_case):
    """Apply the original annual statistics to reconstructed observation counts."""
    rows, annual_rows = [], []
    for distance_key, _, _ in EXPOSURE_DISTANCE_CASES:
        for threshold in REPEAT_THRESHOLDS:
            values = annual_by_case[distance_key][threshold]
            trend = annual_trend(GISA_YEARS, values)
            metadata = {'region': 'NH (>20°N)', 'distance_case': distance_key,
                        'repeat_threshold': threshold, 'exposure_rule': EXPOSURE_RULE,
                        'counting_policy': CONSECUTIVE_COUNTING_POLICY,
                        'time_step_hours': TIME_STEP_HOURS, 'max_gap_hours': MAX_EXPOSURE_GAP_HOURS,
                        **trend_smoothing_metadata(), 'trend_method': TREND_METHOD,
                        'significance_method': SIGNIFICANCE_METHOD}
            for index, year in enumerate(GISA_YEARS):
                annual_rows.append({'record': 'M4A_consecutive_annual_exposure', **metadata,
                                    'year': int(year), 'exposure': float(values[index]),
                                    'fitted': float(trend['fitted'][index]),
                                    'exposure_smooth5': float(trend['smoothed'][index])})
            rows.append({'record': 'S10_sensitivity', 'panel': 'C', **metadata,
                         **{name: trend[name] for name in ('slope_decade', 'slope_ci95_lower',
                                                          'slope_ci95_upper', 'trend_p')}})
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def calculate_from_exposure_csv(csv_dir):
    """Validate time-by-time urban-grid exposure, then recompute M4 statistics.

    Points and occurrences must retain the deterministic exporter order. This
    allows bounded event-level memory and detects duplicate/missing/extra rows.
    Every selected track point is required, including explicitly zero exposure.
    URBAN_GRID_EXPOSURE must equal the summed urban-cell counts of that point's
    occurrence rows; EXPOSED_GRID_COUNT must equal the number of those rows.
    """
    csv_dir = Path(csv_dir).expanduser().resolve()
    for filename in EXPOSURE_CSV_COLUMNS:
        require_file(csv_dir / filename, 'M4 exposure CSV')
    require_file(TRACK_FILE, 'processed IBTrACS track file for CSV selection validation')
    templates = read_exposure_grid_coordinates(csv_dir / 'M4_grid_coordinates.csv')
    annual_by_case = {key: {threshold: np.zeros(len(GISA_YEARS), dtype=float)
                            for threshold in REPEAT_THRESHOLDS} for key in DISTANCE_KEYS}
    regional_annual = {region: np.zeros(len(GISA_YEARS), dtype=float) for region in MAP_EXTENTS}
    with (csv_dir / 'M4_event_points.csv').open(newline='', encoding='utf-8-sig') as points_handle, \
         (csv_dir / 'M4_exposure_occurrences.csv').open(newline='', encoding='utf-8-sig') as occurrences_handle:
        points = exposure_csv_reader(points_handle, 'M4_event_points.csv')
        occurrences = exposure_csv_reader(occurrences_handle, 'M4_exposure_occurrences.csv')
        for selector, event_id, event in selected_exposure_events():
            year = int(event.EVENT_YEAR.iloc[0])
            run_state, repeat_counts, urban_counts = {}, {}, {}
            for expected in event.itertuples(index=False):
                row = next(points, None)
                timestamp = pd.Timestamp(expected.ISO_TIME)
                identity = selector, event_id, timestamp
                if row is None:
                    raise ValueError(f'Missing event point: {identity}')
                actual = row['SELECTOR'], row['EVENT_ID'], pd.Timestamp(row['ISO_TIME'])
                if (actual != identity or row['SID'] != expected.SID
                        or csv_integer(row['EVENT_YEAR'], 'EVENT_YEAR') != year
                        or float(row['TC_LAT']) != float(expected.LAT)
                        or float(row['TC_LON']) != float(expected.LON)):
                    raise ValueError(f'Event point differs from source-track selection or order: {identity}')
                count = csv_integer(row['EXPOSED_GRID_COUNT'], 'EXPOSED_GRID_COUNT', 0)
                reported_urban = csv_integer(row['URBAN_GRID_EXPOSURE'], 'URBAN_GRID_EXPOSURE', 0)
                urban_exposure = 0
                seen = set()
                for _ in range(count):
                    occurrence = next(occurrences, None)
                    if occurrence is None:
                        raise ValueError(f'Missing exposure occurrence at {identity}; cannot treat it as zero.')
                    occurrence_identity = (occurrence['SELECTOR'], occurrence['EVENT_ID'],
                                           pd.Timestamp(occurrence['ISO_TIME']))
                    if occurrence_identity != identity:
                        raise ValueError(f'Occurrence count/order does not match event point: {identity}')
                    key = (csv_integer(occurrence['LAT_KEY'], 'LAT_KEY'),
                           csv_integer(occurrence['LON_KEY'], 'LON_KEY'))
                    number = csv_integer(occurrence['URBAN_COUNT'], 'URBAN_COUNT', 1)
                    urban_exposure += number
                    if key in seen:
                        raise ValueError(f'Duplicate exposure occurrence: {identity}/{key}')
                    seen.add(key)
                    if not (-9000 <= key[0] <= 9000 and -18000 <= key[1] < 18000):
                        raise ValueError(f'Invalid exposed-grid key: {key}')
                    if key in urban_counts and urban_counts[key] != number:
                        raise ValueError(f'Inconsistent URBAN_COUNT within event: {selector}/{event_id}/{key}')
                    urban_counts[key] = number
                    if selector.startswith('nh:'):
                        record_consecutive_exposure(run_state, event_id, key, timestamp, number,
                                                    step_hours=TIME_STEP_HOURS,
                                                    max_gap_hours=MAX_EXPOSURE_GAP_HOURS)
                    else:
                        repeat_counts[key] = repeat_counts.get(key, 0) + 1
                if urban_exposure != reported_urban:
                    raise ValueError(f'URBAN_GRID_EXPOSURE does not match occurrence sums: {identity}')
            year_index = year - GISA_YEAR_MIN
            if selector.startswith('nh:'):
                totals = summarize_consecutive_exposure(run_state, event_id, REPEAT_THRESHOLDS,
                                                        policy=CONSECUTIVE_COUNTING_POLICY)
                for threshold in REPEAT_THRESHOLDS:
                    annual_by_case[selector[3:]][threshold][year_index] += totals[threshold]
            else:
                region = selector[len('region:'):]
                retained = {key: repetitions * urban_counts[key]
                            for key, repetitions in repeat_counts.items()
                            if repetitions >= MIN_EXPOSURE_FREQUENCY}
                regional_annual[region][year_index] += float(sum(retained.values()))
                template = templates[region]
                for (lat_key, lon_key), value in retained.items():
                    i, j = template['lat_lookup'].get(lat_key), template['lon_lookup'].get(lon_key)
                    if i is not None and j is not None:
                        template['annual'][year_index, i, j] += float(value)
                    elif (template['lat_key_range'][0] <= lat_key <= template['lat_key_range'][1]
                          and template['lon_key_range'][0] <= lon_key <= template['lon_key_range'][1]):
                        raise ValueError(f'Exposure occurrence has no native regional grid cell: '
                                         f'{region}/{(lat_key, lon_key)}')
        if next(points, None) is not None:
            raise ValueError('Extra or duplicate event points not present in the source-track selection.')
        if next(occurrences, None) is not None:
            raise ValueError('Extra or duplicate exposure occurrences not accounted for by event-point counts.')
    sensitivity, annual_exposure = sensitivity_from_annual_counts(annual_by_case)
    spatial, regional_fits = {}, {}
    for region, template in templates.items():
        trend, p, significant = spatial_trend(template['annual'])
        spatial[region] = (trend, p, significant, template['lat'], template['lon'])
        regional_fits[region] = annual_trend(GISA_YEARS, regional_annual[region])
    return prepare_plot_outputs(spatial, regional_fits, sensitivity, annual_exposure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Root containing IBTrACS, GISA and (by default) MSWEP.')
    parser.add_argument('--mswep-dir', type=Path,
                        help='Override Raw_data/MSWEP; e.g. /mnt/data/precipitation/mswep.')
    parser.add_argument('--exposure-csv-dir', type=Path,
                        help='Use all three M4 event/grid CSVs explicitly; missing files are errors.')
    parser.add_argument('--use-netcdf', action='store_true',
                        help='Force the original GISA/MSWEP calculation even when CSVs are available.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Destination for the four Database/EXPOSURE plotting inputs.')
    args = parser.parse_args(argv)
    if args.exposure_csv_dir is not None and args.use_netcdf:
        parser.error('--exposure-csv-dir and --use-netcdf cannot be combined.')
    configure_inputs(args.raw_data_dir, args.mswep_dir)
    csv_dir = (args.exposure_csv_dir.expanduser().resolve() if args.exposure_csv_dir is not None
               else args.raw_data_dir.expanduser().resolve() / 'EXPOSURE_CSV')
    use_csv = (not args.use_netcdf and (args.exposure_csv_dir is not None
               or all((csv_dir / name).is_file() for name in EXPOSURE_CSV_COLUMNS)))
    if use_csv:
        print(f'Recomputing M4 runs and statistics from event/grid CSVs: {csv_dir}', flush=True)
        series, statistics, maps = calculate_from_exposure_csv(csv_dir)
    else:
        check_inputs()
        validate_gisa_netcdf()
        print('Recalculating M4 exposures from tracks, GISA and MSWEP; no result caches.', flush=True)
        with open_source_netcdf(GISA_NC_FILE, decode_times=False, mask_and_scale=True) as city:
            sensitivity, annual_exposure = calculate_sensitivity_trends(city)
            spatial, regional_fits = calculate_regional_outputs(city)
        series, statistics, maps = prepare_plot_outputs(
            spatial, regional_fits, sensitivity, annual_exposure)
    write_preprocessed(args.output_dir.expanduser().resolve(), series, statistics, maps)


if __name__ == '__main__':
    main()
