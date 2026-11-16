#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile
import warnings

import dask
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.path import Path as MplPath

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/VWS_ATTRIBUTION'


EXPERIMENTS = {
    "hist-aer": {"label": "AER", "color": "#DDAA9E", "marker": "o"},
    "hist-GHG": {"label": "GHG", "color": "#456D9F", "marker": "s"},
    "hist-nat": {"label": "NAT", "color": "#88ADC1", "marker": "^"},
    "historical": {"label": "HIST", "color": "#9B5F5B", "marker": None},
}

NH_MONTHS = (7, 8, 9, 10, 11)

S6_REGION = {
    "name": "WNP_20-40N",
    "label": "WNP (20–40°N)",
    "lat_min": 20.0,
    "lat_max": 40.0,
    "months": NH_MONTHS,
    "season": "July–November",
}

EAST_ASIA_EXTENT = (100.0, 145.0, 10.0, 45.0)

MODELS = (
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
    "CNRM-CM6-1", "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
    "IPSL-CM6A-LR", "MIROC6", "MRI-ESM2-0", "NorESM2-LM",
)

MIN_VALID_YEARS = 20

COORDINATE_CANDIDATES = {
    "time": ("time", "valid_time", "date"),
    "level": ("level", "plev", "pressure_level", "pressure", "isobaricInhPa"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude"),
}

AOD_CANDIDATES = (
    "TOTEXTTAU", "totexttau", "AODANA", "aodana", "AOD", "aod",
    "AOD_550", "aod550", "aerosol_optical_depth",
)


PROFILE_COLUMNS = ('Series', 'Distance_Center_km', 'VWS_Trend_m_s_per_decade', 'MK_P')
SERIES_ORDER = ('AER', 'GHG', 'NAT', 'HIST', 'ERA5')
DISTANCE_CENTERS = tuple(range(-350, 400, 100))
MAP_VARIABLES = ('aod_trend', 'MK_P', 'distance_to_coast')


def validate_profiles(table):
    """Validate 32 ensemble medians and eight ERA5 trends with genuine MK P."""
    if table.columns.duplicated().any() or set(table.columns) != set(PROFILE_COLUMNS):
        raise ValueError(f'S6 profiles require exactly these columns: {PROFILE_COLUMNS}')
    table = table.loc[:, list(PROFILE_COLUMNS)].copy()
    keys = ['Series', 'Distance_Center_km']
    for name in PROFILE_COLUMNS[1:]:
        table[name] = pd.to_numeric(table[name], errors='raise')
    expected = [(series, distance) for series in SERIES_ORDER for distance in DISTANCE_CENTERS]
    if len(table) != 40 or table.duplicated(keys).any():
        raise ValueError('S6 requires 40 unique profile points: five series and eight distance bins')
    if set(table[keys].itertuples(index=False, name=None)) != set(expected):
        raise ValueError('S6 profile series/distance combinations are incomplete or unexpected')
    if not np.isfinite(table['VWS_Trend_m_s_per_decade']).all():
        raise ValueError('S6 profile trends must be finite')
    observed = table.Series.eq('ERA5')
    p = table.loc[observed, 'MK_P']
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError('ERA5 profiles require genuine finite MK P values within [0, 1]')
    if not table.loc[~observed, 'MK_P'].isna().all():
        raise ValueError('Model medians have no MK test and must leave MK_P blank')
    order = pd.MultiIndex.from_tuples(expected, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(PROFILE_COLUMNS)]


def validate_aod_map(dataset):
    """Protect native East Asia coordinates and retain real P values, not flags."""
    if set(dataset.data_vars) != set(MAP_VARIABLES):
        raise ValueError(f'S6 AOD map requires exactly these variables: {MAP_VARIABLES}')
    for name in MAP_VARIABLES:
        if dataset[name].dims != ('lat', 'lon'):
            raise ValueError(f'S6 {name} must have dimensions (lat, lon)')
    lat, lon = dataset.lat.values, dataset.lon.values
    expected_lat = np.arange(10.0, 45.0 + .001, .5)
    expected_lon = np.arange(100.0, 145.0 + .001, .625)
    if (lat.shape != expected_lat.shape or lon.shape != expected_lon.shape
            or not np.allclose(lat, expected_lat, rtol=0, atol=1e-8)
            or not np.allclose(lon, expected_lon, rtol=0, atol=1e-8)):
        raise ValueError('S6 must retain the 71 x 73 native MERRA-2 East Asia grid')
    for key, expected in {'period': '1980-2024', 'season': 'July-November',
                          'trend_method': 'OLS', 'significance_method': 'original Mann-Kendall'}.items():
        if dataset.attrs.get(key) != expected:
            raise ValueError(f'S6 AOD map has incompatible {key} metadata')
    trend = np.asarray(dataset.aod_trend.values, dtype=float)
    p = np.asarray(dataset.MK_P.values, dtype=float)
    distance = np.asarray(dataset.distance_to_coast.values, dtype=float)
    finite = np.isfinite(trend)
    if np.isinf(trend).any() or not finite.any() or not np.isfinite(distance).all():
        raise ValueError('S6 AOD trends/distances contain invalid values')
    if np.isinf(p).any() or not np.isfinite(p[finite]).all() or not ((p[finite] >= 0) & (p[finite] <= 1)).all():
        raise ValueError('Every finite AOD trend requires a genuine MK P value within [0, 1]')
    if not np.isnan(p[~finite]).all():
        raise ValueError('AOD cells without a valid trend must leave MK_P missing')
    return dataset


def find_name(names, candidates, description):
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Cannot find {description}; available names: {sorted(names)}")

def normalize_longitude(array):
    lon = ((array.lon + 180.0) % 360.0) - 180.0
    array = array.assign_coords(lon=lon).sortby("lon")
    _, unique_indices = np.unique(array.lon.values, return_index=True)
    if len(unique_indices) != array.sizes["lon"]:
        array = array.isel(lon=np.sort(unique_indices))
    return array

def canonicalize(array, require_level=False, require_time=True):
    names = set(array.dims) | set(array.coords)
    rename = {}
    coordinate_names = ("time", "lat", "lon") if require_time else ("lat", "lon")
    for canonical in coordinate_names:
        old = find_name(names, COORDINATE_CANDIDATES[canonical], canonical)
        if old != canonical:
            rename[old] = canonical
    if require_level:
        old = find_name(names, COORDINATE_CANDIDATES["level"], "pressure level")
        if old != "level":
            rename[old] = "level"
    array = array.rename(rename)
    array = normalize_longitude(array)
    if array.lat.values[0] > array.lat.values[-1]:
        array = array.sortby("lat")
    if require_level:
        values = np.asarray(array.level.values, dtype=float)
        units = str(array.level.attrs.get("units", "")).strip().lower()
        if units in {"pa", "pascal", "pascals"} or (
            not units and np.nanmax(np.abs(values)) > 2000.0
        ):
            values = values / 100.0
        array = array.assign_coords(level=values)
        array.level.attrs["units"] = "hPa"
    return array

def choose_data_variable(dataset, candidates, description):
    for candidate in candidates:
        if candidate in dataset.data_vars:
            print(f"{description} variable: {candidate}")
            return candidate
    numeric = [
        name for name, value in dataset.data_vars.items()
        if np.issubdtype(value.dtype, np.number)
    ]
    if len(numeric) == 1:
        warnings.warn(
            f"None of {candidates} found; using the only numeric variable {numeric[0]!r}"
        )
        return numeric[0]
    raise KeyError(
        f"Cannot identify {description}; data variables={list(dataset.data_vars)}"
    )

def require_same_grid(field, distance, context):
    same = (
        field.sizes["lat"] == distance.sizes["lat"]
        and field.sizes["lon"] == distance.sizes["lon"]
        and np.allclose(field.lat.values, distance.lat.values, atol=1e-5, rtol=0)
        and np.allclose(field.lon.values, distance.lon.values, atol=1e-5, rtol=0)
    )
    if not same:
        raise ValueError(
            f"{context}: field and distance grids differ; regenerate the distance file "
            "from this model/data grid"
        )
    return distance.assign_coords(lat=field.lat, lon=field.lon)

def complete_season(field, months, start_year, end_year, context):
    selected = field.where(field.time.dt.month.isin(months), drop=True)
    counts = pd.Series(selected.time.dt.year.values).value_counts()
    incomplete = {
        year: int(counts.get(year, 0))
        for year in range(start_year, end_year + 1)
        if int(counts.get(year, 0)) != len(months)
    }
    if incomplete:
        raise ValueError(f"{context}: incomplete seasonal months: {incomplete}")
    return selected.groupby("time.year").mean("time", skipna=True)

def mann_kendall_p(values):
    """Two-sided Mann-Kendall P value with correction for tied values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    s_stat = sum(
        np.sign(values[j] - values[i])
        for i in range(n - 1) for j in range(i + 1, n)
    )
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if s_stat > 0:
        z_stat = (s_stat - 1) / np.sqrt(var_s)
    elif s_stat < 0:
        z_stat = (s_stat + 1) / np.sqrt(var_s)
    else:
        z_stat = 0.0
    return 2 * stats.norm.sf(abs(z_stat))

def calculate_grid_trend_and_p(annual):
    """Calculate per-grid-cell OLS trend and Mann-Kendall significance."""
    years = np.asarray(annual.year.values, dtype=float)
    def per_cell(values):
        valid = np.isfinite(values)
        valid_count = int(valid.sum())
        if valid_count < MIN_VALID_YEARS:
            return np.nan, np.nan, valid_count
        observed = np.asarray(values[valid], dtype=float)
        x = years[valid]
        slope = float(stats.linregress(x, observed).slope * 10.0)
        pvalue = mann_kendall_p(observed)
        return slope, pvalue, valid_count
    slope, pvalue, count = xr.apply_ufunc(
        per_cell,
        annual,
        input_core_dims=[["year"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="forbidden",
        output_dtypes=[float, float, np.int16],
    )
    slope = slope.where(count >= MIN_VALID_YEARS).astype(np.float32)
    pvalue = pvalue.where(count >= MIN_VALID_YEARS).astype(np.float64)
    return slope, pvalue, count.astype(np.int16)



def open_netcdf(path, **kwargs):
    """Read NetCDF3/HDF5 using backends compatible with Unicode file paths."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('rb') as handle:
        signature = handle.read(8)
    if signature.startswith(b'CDF'):
        engine = 'scipy'
    elif signature == b'\x89HDF\r\n\x1a\n':
        engine = 'h5netcdf'
    else:
        raise ValueError(f'Unsupported NetCDF signature: {path}')
    return xr.open_dataset(path, engine=engine, **kwargs)





def open_signed_distance(path):
    with open_netcdf(path) as dataset:
        name = next((item for item in ('distance_to_coast', 'signed_distance_to_coast')
                     if item in dataset.data_vars), None)
        if name is None:
            raise KeyError(f'No signed distance variable in {path}')
        distance = canonicalize(dataset[name].squeeze(drop=True), require_time=False)
        sign = str(distance.attrs.get('sign_convention', '')).lower()
        if any(text in sign for text in ('positive over land', 'land positive', 'land > 0')):
            distance = -distance
        elif not any(text in sign for text in ('negative over land', 'land negative', 'land < 0')):
            raise ValueError(f'Undocumented distance sign in {path}')
        distance.attrs['sign_convention'] = 'negative over land; positive over ocean'
        return distance.load()


def calculate_aod_map(database_root):
    """Apply the original analysis only to the small native East Asia subset."""
    database_root = Path(database_root)
    if (database_root / 'ENVIRONMENT/aod_map.nc').is_file():
        from environment_inputs import aod_inputs
        annual, distance, variable = aod_inputs(database_root)
    else:
        path = database_root / 'ENVIRONMENT/MERRA2/MERRA2_gas_1980-2024.nc'
        with open_netcdf(path, chunks={'time': 12}) as dataset:
            variable = choose_data_variable(dataset, AOD_CANDIDATES, 'MERRA-2 AOD')
            aod = canonicalize(dataset[variable])
            for dim in set(aod.dims) - {'time', 'lat', 'lon'}:
                if aod.sizes[dim] != 1:
                    raise ValueError(f'Unsupported MERRA-2 AOD dimension: {dim}')
                aod = aod.isel({dim: 0}, drop=True)
            aod = aod.sel(lat=slice(10, 45), lon=slice(100, 145))
            aod = aod.where(((aod.time.dt.year >= 1980) & (aod.time.dt.year <= 2024)), drop=True)
            distance = open_signed_distance(database_root / 'DIST2LAND/MERRA2_distance_to_coast.nc')
            distance = distance.sel(lat=slice(10, 45), lon=slice(100, 145))
            distance = require_same_grid(aod, distance, 'MERRA-2 East Asia AOD')
            annual = complete_season(aod, NH_MONTHS, 1980, 2024, 'MERRA-2 East Asia July-November')
            annual = annual.astype(np.float32).load()
    trend, p_value, count = calculate_grid_trend_and_p(annual)
    trend.name = 'aod_trend'
    trend.attrs = {'long_name': 'OLS trend in July-November AOD', 'units': 'decade-1'}
    p_value.name = 'MK_P'
    p_value.attrs = {'long_name': 'Two-sided original Mann-Kendall P value', 'units': '1'}
    result = xr.Dataset({'aod_trend': trend, 'MK_P': p_value,
                         'distance_to_coast': distance.astype(np.float32)})
    result.attrs = {'period': '1980-2024', 'season': 'July-November', 'source_variable': variable,
                    'region': 'East Asia (100-145E, 10-45N)', 'trend_method': 'OLS',
                    'significance_method': 'original Mann-Kendall', 'minimum_valid_years': MIN_VALID_YEARS,
                    'distance_sign': 'land negative; ocean positive', 'spatial_smoothing': 'none'}
    return validate_aod_map(result)





def write_profiles(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False, float_format='%.17g', lineterminator='\n')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_map(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        encoding = {name: {'zlib': True, 'complevel': 4, 'shuffle': True} for name in MAP_VARIABLES}
        dataset.to_netcdf(temporary, engine='h5netcdf', format='NETCDF4', encoding=encoding)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


CMIP6_START_YEAR, CMIP6_END_YEAR = 1980, 2014
ERA5_START_YEAR, ERA5_END_YEAR = 1980, 2014
LOW_LEVEL_HPA, UPPER_LEVEL_HPA = 850.0, 200.0
DISTANCE_EDGES_KM = np.arange(-400.0, 400.0 + 100.0, 100.0)
FIG3_REGION = S6_REGION
WNP_POLYGON = np.asarray([
    (102.0, 15.0), (110.0, 8.0), (113.0, 5.0), (148.0, 5.0),
    (148.0, 55.0), (102.0, 55.0), (102.0, 15.0),
], dtype=float)


def configure_paths(raw_data_dir):
    global CMIP6_DIR, DISTANCE_DIR, ERA5_U_FILE, ERA5_V_FILE, ERA5_DISTANCE_FILE
    raw_data_dir = Path(raw_data_dir).expanduser().resolve()
    CMIP6_DIR = raw_data_dir / 'CMIP6'
    DISTANCE_DIR = raw_data_dir / 'DIST2LAND'
    ERA5_U_FILE = raw_data_dir / 'ENVIRONMENT/ERA5/u_component_of_wind_1980-2024.nc'
    ERA5_V_FILE = raw_data_dir / 'ENVIRONMENT/ERA5/v_component_of_wind_1980-2024.nc'
    ERA5_DISTANCE_FILE = DISTANCE_DIR / 'ERA5_distance_to_coast.nc'


configure_paths(DEFAULT_RAW_DATA_DIR)


def cmip6_path(model, experiment, variable):
    return CMIP6_DIR / (
        f"{model}_{experiment}_{variable}_{CMIP6_START_YEAR}-{CMIP6_END_YEAR}.nc"
    )


def cmip6_distance_path(model):
    return DISTANCE_DIR / f"CMIP6_{model}_distance_to_coast.nc"


def open_cmip6_field(path, variable):
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = open_netcdf(path, chunks={"time": 60})
    if variable not in dataset.data_vars:
        raise KeyError(f"{variable!r} not in {path}; variables={list(dataset.data_vars)}")
    array = canonicalize(dataset[variable], require_level=True)
    extra = set(array.dims) - {"time", "level", "lat", "lon"}
    for dim in list(extra):
        if array.sizes[dim] != 1:
            raise ValueError(f"Unsupported non-singleton dimension {dim} in {path}")
        array = array.isel({dim: 0}, drop=True)
    years = array.time.dt.year
    array = array.where(
        (years >= CMIP6_START_YEAR) & (years <= CMIP6_END_YEAR), drop=True
    )
    return array


def select_level(array, target_hpa):
    levels = np.asarray(array.level.values, dtype=float)
    index = int(np.argmin(np.abs(levels - target_hpa)))
    if abs(levels[index] - target_hpa) > 1.0:
        raise ValueError(f"Cannot find {target_hpa:g} hPa; available={levels.tolist()}")
    return array.isel(level=index, drop=True)


def region_mask(distance, region):
    """Select the exact 001_CAL_CLTC WNP polygon, restricted to 20–40°N."""
    latitude_ok = (
        (distance.lat >= region["lat_min"])
        & (distance.lat <= region["lat_max"])
    ).broadcast_like(distance)
    lon2d, lat2d = np.meshgrid(
        np.asarray(distance.lon.values, dtype=float),
        np.asarray(distance.lat.values, dtype=float),
    )
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside_wnp = MplPath(WNP_POLYGON, closed=True).contains_points(
        points, radius=1.0e-10
    ).reshape(lat2d.shape)
    wnp_mask = xr.DataArray(
        inside_wnp,
        coords={"lat": distance.lat, "lon": distance.lon},
        dims=("lat", "lon"),
    )
    return latitude_ok & wnp_mask


def spherical_cell_area(lat, lon, earth_radius_m=6_371_000.0):
    """Exact spherical areas inferred from 1-D rectilinear cell centres."""
    lat_values = np.asarray(lat.values, dtype=np.float64)
    lon_values = np.asarray(lon.values, dtype=np.float64)
    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError(
            "CMIP6 cell-area calculation requires one-dimensional lat/lon coordinates"
        )
    if lat_values.size < 2 or lon_values.size < 2:
        raise ValueError("CMIP6 cell-area calculation needs at least two lat/lon points")
    if (
        not np.all(np.isfinite(lat_values))
        or not np.all(np.isfinite(lon_values))
        or not np.all(np.diff(lat_values) > 0)
        or not np.all(np.diff(lon_values) > 0)
    ):
        raise ValueError("CMIP6 latitude/longitude coordinates must be finite and increasing")
    lat_edges = np.empty(lat_values.size + 1, dtype=np.float64)
    lat_edges[1:-1] = 0.5 * (lat_values[:-1] + lat_values[1:])
    lat_edges[0] = lat_values[0] - 0.5 * (lat_values[1] - lat_values[0])
    lat_edges[-1] = lat_values[-1] + 0.5 * (lat_values[-1] - lat_values[-2])
    lat_edges = np.clip(lat_edges, -90.0, 90.0)
    lon_edges = np.empty(lon_values.size + 1, dtype=np.float64)
    lon_edges[1:-1] = 0.5 * (lon_values[:-1] + lon_values[1:])
    lon_edges[0] = lon_values[0] - 0.5 * (lon_values[1] - lon_values[0])
    lon_edges[-1] = lon_values[-1] + 0.5 * (lon_values[-1] - lon_values[-2])
    latitude_factor = np.abs(np.diff(np.sin(np.deg2rad(lat_edges))))
    longitude_width = np.abs(np.diff(np.deg2rad(lon_edges)))
    area = earth_radius_m ** 2 * latitude_factor[:, None] * longitude_width[None, :]
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("Calculated CMIP6 grid-cell areas are invalid")
    return xr.DataArray(
        area,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="calculated_cell_area",
        attrs={
            "long_name": "spherical grid-cell area inferred from coordinate centres",
            "units": "m2",
            "earth_radius_m": float(earth_radius_m),
        },
    )


def raw_cmip6_vws_bins(model, experiment):
    """Extract July–November native-grid distance-bin mean VWS by year."""
    print(f"Processing CMIP6: {model} / {experiment}")
    u = open_cmip6_field(cmip6_path(model, experiment, "ua"), "ua")
    v = open_cmip6_field(cmip6_path(model, experiment, "va"), "va")
    u, v = xr.align(u, v, join="exact")
    vws = np.hypot(
        select_level(u, UPPER_LEVEL_HPA) - select_level(u, LOW_LEVEL_HPA),
        select_level(v, UPPER_LEVEL_HPA) - select_level(v, LOW_LEVEL_HPA),
    ).sel(lat=slice(FIG3_REGION["lat_min"], FIG3_REGION["lat_max"]))
    distance = open_signed_distance(cmip6_distance_path(model)).sel(
        lat=slice(FIG3_REGION["lat_min"], FIG3_REGION["lat_max"])
    )
    distance = require_same_grid(vws, distance, f"{model}/{experiment}")
    region = FIG3_REGION
    region_cells = region_mask(distance, region)
    selected_domain_cells = int(region_cells.sum().item())
    print(
        f"  WNP 20-40N polygon contains "
        f"{selected_domain_cells:,} native-grid cells"
    )
    if selected_domain_cells == 0:
        raise RuntimeError(
            f"{model}/{experiment}: WNP 20-40N polygon contains no grid cells; "
            "check longitude normalization and model grid"
        )
    annual = complete_season(
        vws, region["months"], CMIP6_START_YEAR, CMIP6_END_YEAR,
        f"{model}/{experiment}/{region['months']}",
    )
    print(f"  loading seasonal annual VWS for months {region['months']}")
    with ProgressBar():
        annual = annual.astype(np.float32).load()
    base = region_cells & np.isfinite(distance)
    years = np.asarray(annual.year.values, dtype=float)
    cell_area = spherical_cell_area(distance.lat, distance.lon)
    rows = []
    for index, (left, right) in enumerate(
        zip(DISTANCE_EDGES_KM[:-1], DISTANCE_EDGES_KM[1:])
    ):
        if index == len(DISTANCE_EDGES_KM) - 2:
            in_bin = (distance >= left) & (distance <= right)
        else:
            in_bin = (distance >= left) & (distance < right)
        selected = base & in_bin
        # One annual, cell-area-weighted spatial mean is calculated first.
        # The local archive has no areacella, so all 12 models consistently use
        # spherical areas inferred from their own rectilinear grid bounds.
        valid_area = cell_area.where(selected)
        numerator = (annual * valid_area).sum(("lat", "lon"), skipna=True)
        denominator = valid_area.where(annual.notnull()).sum(
            ("lat", "lon"), skipna=True
        )
        series = numerator / denominator
        values = np.asarray(series.values, dtype=float)
        rows.append(pd.DataFrame({
            'Year': years.astype(int), 'VWS': values,
            'Distance_Left_km': left, 'Distance_Right_km': right,
            'Distance_Center_km': (left+right)/2,
            'Grid_Cell_Count': int(selected.sum().item()),
            'Domain_Grid_Cell_Count': selected_domain_cells,
        }))
    u.close()
    v.close()
    return pd.concat(rows, ignore_index=True)


def model_experiment_vws_trends(model, experiment, annual_bins=None):
    """Fit each model's trend to its annual distance-bin mean VWS series."""
    if annual_bins is None:
        annual_bins = raw_cmip6_vws_bins(model, experiment)
    region = FIG3_REGION
    rows = []
    for (left, right), group in annual_bins.groupby(['Distance_Left_km', 'Distance_Right_km']):
        years = group.Year.to_numpy(dtype=float)
        values = group.VWS.to_numpy(dtype=float)
        valid = np.isfinite(years) & np.isfinite(values)
        n_years = int(valid.sum())
        if n_years < MIN_VALID_YEARS:
            raise ValueError(
                f"{model}/{experiment}/{region['name']}/{left:g}:{right:g} km "
                f"has only {n_years} valid annual bin means"
            )
        fit = stats.linregress(years[valid], values[valid])
        rows.append({
            "Record_Type": "Model",
            "Model": model,
            "Experiment": experiment,
            "Forcing": EXPERIMENTS[experiment]["label"],
            "Region": region["name"],
            "Region_Label": region["label"],
            "Season": region["season"],
            "Distance_Left_km": float(left),
            "Distance_Right_km": float(right),
            "Distance_Center_km": float((left + right) / 2.0),
            "Grid_Cell_Count": int(group.Grid_Cell_Count.iloc[0]),
            "Domain_Grid_Cell_Count": int(group.Domain_Grid_Cell_Count.iloc[0]),
            "Exclusion_Region": "none; WNP inclusion polygon only",
            "Start_Year": CMIP6_START_YEAR,
            "End_Year": CMIP6_END_YEAR,
            "N_Years": n_years,
            "VWS_Trend_m_s_per_decade": float(fit.slope * 10.0),
            "OLS_P_Value": float(fit.pvalue),
            "Annual_Bin_Mean_First": float(values[valid][0]),
            "Annual_Bin_Mean_Last": float(values[valid][-1]),
            "Trend_Method": (
                "spatial mean VWS in each signed-distance bin for every "
                "July-November season, then OLS trend"
            ),
            "Spatial_Statistic": (
                "cell-area-weighted mean on each model's native grid"
            ),
            "Area_Weight_Method": (
                "spherical cell area from 1-D lat/lon cell bounds; R=6371000 m"
            ),
            "Distance_Sign": "land negative; ocean positive",
        })
    return rows


def calculate_era5_vws_trends():
    """ERA5 reference: same season, region, signed bins and OLS method as CMIP6."""
    if (CMIP6_DIR.parent / 'ENVIRONMENT/era5_vws.csv').is_file():
        from environment_inputs import era5_vws
        return era5_vws(CMIP6_DIR.parent, mann_kendall_p)
    return era5_vws_trends_from_monthly()


def era5_vws_trends_from_monthly():
    for path in (ERA5_U_FILE, ERA5_V_FILE, ERA5_DISTANCE_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    with open_netcdf(ERA5_U_FILE, chunks='auto') as uds, \
            open_netcdf(ERA5_V_FILE, chunks='auto') as vds:
        uname = choose_data_variable(uds, ('u', 'ua', 'uwnd', 'u_component_of_wind'), 'ERA5 zonal wind')
        vname = choose_data_variable(vds, ('v', 'va', 'vwnd', 'v_component_of_wind'), 'ERA5 meridional wind')
        u = canonicalize(uds[uname], require_level=True)
        v = canonicalize(vds[vname], require_level=True)
        fields = []
        for field in (u, v):
            for dim in set(field.dims) - {'time', 'level', 'lat', 'lon'}:
                if field.sizes[dim] != 1:
                    raise ValueError(f'ERA5 has unsupported dimension {dim}: {field.sizes[dim]}')
                field = field.isel({dim: 0}, drop=True)
            field = field.where(
                (field.time.dt.year >= ERA5_START_YEAR)
                & (field.time.dt.year <= ERA5_END_YEAR), drop=True,
            ).sel(lat=slice(FIG3_REGION['lat_min'], FIG3_REGION['lat_max']))
            # Keep spatial/temporal blocks small before vector arithmetic and seasonal slicing.
            field = field.chunk({'time': 12, 'lat': 80, 'lon': 180})
            fields.append(field)
        u, v = xr.align(*fields, join='exact')
        vws = np.hypot(
            select_level(u, UPPER_LEVEL_HPA) - select_level(u, LOW_LEVEL_HPA),
            select_level(v, UPPER_LEVEL_HPA) - select_level(v, LOW_LEVEL_HPA),
        )
        distance = open_signed_distance(ERA5_DISTANCE_FILE).sel(
            lat=slice(FIG3_REGION['lat_min'], FIG3_REGION['lat_max'])
        )
        distance = require_same_grid(vws, distance, 'ERA5 VWS')
        base = region_mask(distance, FIG3_REGION) & np.isfinite(distance)
        area = spherical_cell_area(distance.lat, distance.lon)
        annual = complete_season(
            vws, FIG3_REGION['months'], ERA5_START_YEAR, ERA5_END_YEAR, 'ERA5 VWS'
        ).astype(np.float32).load()
    years = np.asarray(annual.year.values, dtype=float)
    rows, annual_rows = [], []
    for index, (left, right) in enumerate(zip(DISTANCE_EDGES_KM[:-1], DISTANCE_EDGES_KM[1:])):
        in_bin = ((distance >= left) &
                  ((distance <= right) if index == len(DISTANCE_EDGES_KM)-2 else (distance < right)))
        selected = base & in_bin
        if not bool(selected.any().item()):
            raise ValueError(f'ERA5 has no cells in distance bin {left:g}..{right:g} km')
        weights = area.where(selected)
        series = ((annual * weights).sum(('lat', 'lon'), skipna=True)
                  / weights.where(annual.notnull()).sum(('lat', 'lon'), skipna=True))
        values = np.asarray(series.values, dtype=float)
        valid = np.isfinite(years) & np.isfinite(values)
        if valid.sum() != ERA5_END_YEAR - ERA5_START_YEAR + 1:
            raise ValueError(f'ERA5 {left:g}..{right:g} km: incomplete annual means')
        fit = stats.linregress(years[valid], values[valid])
        # Same trend-significance convention as the native-grid AOD calculation.
        observed = values[valid]
        mk_p = mann_kendall_p(observed)
        rows.append({
            'Panel': 'a', 'Record_Type': 'Reanalysis distance-bin trend', 'Dataset': 'ERA5',
            'Region': FIG3_REGION['name'], 'Region_Label': FIG3_REGION['label'],
            'Season': FIG3_REGION['season'], 'Start_Year': ERA5_START_YEAR,
            'End_Year': ERA5_END_YEAR, 'N_Years': int(valid.sum()),
            'Distance_Left_km': float(left), 'Distance_Right_km': float(right),
            'Distance_Center_km': float((left+right)/2),
            'Grid_Cell_Count': int(selected.sum().item()),
            'VWS_Trend_m_s_per_decade': float(fit.slope*10),
            'OLS_P_Value': float(fit.pvalue),
            'MK_P': mk_p, 'Significant_95': bool(mk_p < 0.05),
            'Spatial_Statistic': 'spherical grid-cell area weighted mean on native ERA5 grid',
        })
        annual_rows.extend({
            'Dataset': 'ERA5', 'Year': int(year), 'Region': FIG3_REGION['name'],
            'Distance_Center_km': float((left+right)/2), 'Mean_VWS_m_s': float(value),
        } for year, value in zip(years, values))
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def calculate_cmip6_summary():
    compact_path = CMIP6_DIR / 'vws_annual_bins.csv.gz'
    compact = None
    if compact_path.is_file():
        from cmip6_inputs import read_annual_bins
        compact = read_annual_bins(
            compact_path, MODELS, EXPERIMENTS, DISTANCE_EDGES_KM,
            CMIP6_START_YEAR, CMIP6_END_YEAR, 'VWS')
    rows = []
    for model in MODELS:
        for experiment in EXPERIMENTS:
            bins = compact.get_group((model, experiment)) if compact is not None else None
            rows.extend(model_experiment_vws_trends(model, experiment, bins))
    model_table = pd.DataFrame(rows)
    group_columns = [
        "Experiment", "Forcing", "Region", "Region_Label", "Season",
        "Distance_Left_km", "Distance_Right_km", "Distance_Center_km",
        "Exclusion_Region", "Start_Year", "End_Year", "Distance_Sign",
        "Area_Weight_Method",
    ]
    ensemble = (
        model_table.groupby(group_columns, as_index=False)
        .agg(
            Model_Count=("Model", "nunique"),
            Median_VWS_Trend=("VWS_Trend_m_s_per_decade", "median"),
            Q25_VWS_Trend=("VWS_Trend_m_s_per_decade", lambda x: x.quantile(0.25)),
            Q75_VWS_Trend=("VWS_Trend_m_s_per_decade", lambda x: x.quantile(0.75)),
            Min_VWS_Trend=("VWS_Trend_m_s_per_decade", "min"),
            Max_VWS_Trend=("VWS_Trend_m_s_per_decade", "max"),
        )
    )
    ensemble.insert(0, "Record_Type", "Ensemble")
    ensemble.insert(1, "Model", "12-model median")
    if not (ensemble["Model_Count"] == len(MODELS)).all():
        bad = ensemble[ensemble["Model_Count"] != len(MODELS)]
        raise RuntimeError(f"Some ensemble bins do not contain all 12 models:\n{bad}")
    return model_table, ensemble


def calculate_profiles():
    """Compute all 40 output rows from monthly wind fields."""
    models, ensemble = calculate_cmip6_summary()
    if len(models) != 384 or len(ensemble) != 32:
        raise ValueError('Expected 384 model slopes and 32 ensemble medians')
    era5, _annual = calculate_era5_vws_trends()
    rows = []
    for experiment, settings in EXPERIMENTS.items():
        for center in DISTANCE_CENTERS:
            selected = ensemble.loc[ensemble.Experiment.eq(experiment)
                                    & ensemble.Distance_Center_km.eq(center)]
            if len(selected) != 1 or not selected.Model_Count.eq(len(MODELS)).all():
                raise ValueError(f'Incomplete ensemble: {experiment}, {center} km')
            rows.append({'Series': settings['label'], 'Distance_Center_km': center,
                         'VWS_Trend_m_s_per_decade': float(selected.Median_VWS_Trend.iloc[0]),
                         'MK_P': np.nan})
    for center in DISTANCE_CENTERS:
        selected = era5.loc[era5.Distance_Center_km.eq(center)]
        if len(selected) != 1:
            raise ValueError(f'Missing ERA5 result for {center} km')
        rows.append({'Series': 'ERA5', 'Distance_Center_km': center,
                     'VWS_Trend_m_s_per_decade': float(selected.VWS_Trend_m_s_per_decade.iloc[0]),
                     'MK_P': float(selected.MK_P.iloc[0])})
    return validate_profiles(pd.DataFrame(rows, columns=PROFILE_COLUMNS))


def preprocess(raw_data_dir, output_dir, panels=('A', 'B')):
    raw_data_dir = Path(raw_data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if raw_data_dir == output_dir:
        raise ValueError('Raw inputs and calculated outputs must use separate directories')
    configure_paths(raw_data_dir)
    if 'A' in panels:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            profiles = calculate_profiles()
        path = output_dir / 'FigS6A_VWS_profiles.csv'
        write_profiles(path, profiles)
        print(f'Saved {len(profiles)} profiles: {path}')
    if 'B' in panels:
        with dask.config.set(scheduler='synchronous', **{'array.slicing.split_large_chunks': True}):
            dataset = calculate_aod_map(raw_data_dir)
        path = output_dir / 'FigS6B_MERRA2_AOD.nc'
        write_map(path, dataset)
        print(f'Saved native AOD trend, original MK P and signed distance: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir',
                        type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--panels', nargs='+', choices=('A', 'B'), default=['A', 'B'])
    args = parser.parse_args(argv)
    preprocess(args.raw_data_dir, args.output_dir, args.panels)


if __name__ == '__main__':
    main()
