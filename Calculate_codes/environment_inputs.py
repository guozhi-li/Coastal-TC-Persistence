"""Calculate observational/reanalysis figure statistics from pre-trend annual data."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr


def read_series(raw, name):
    return pd.read_csv(Path(raw) / 'ENVIRONMENT' / name, float_precision='round_trip')


def checked_series(table, year_col, value_col, start, end):
    ordered = table.sort_values(year_col)
    years = ordered[year_col].to_numpy()
    values = ordered[value_col].to_numpy(dtype=float)
    if not np.array_equal(years, np.arange(start, end + 1)) or not np.isfinite(values).all():
        raise ValueError(f'Expected one finite annual value per calendar year from {start} through {end}')
    return years, values


def fit_profiles(annual, mk_p):
    rows = []
    keys = ['Source', 'Region', 'Distance_Center_km']
    for (source, region, center), group in annual.groupby(keys, sort=False):
        years, values = checked_series(group, 'Year', 'Value', 1982 if source == 'OISST' else 1980, 2024)
        fit = stats.linregress(years.astype(float), values)
        rows.append({'Region': region, 'Source': source, 'Distance_Center_km': float(center),
                     'OLS_Trend_per_Decade': fit.slope * 10,
                     'OLS_CI95_per_Decade': stats.t.ppf(.975, len(years) - 2) * fit.stderr * 10,
                     'MK_P': mk_p(values)})
    return pd.DataFrame(rows)


def grid_inputs(raw, name, first):
    path = Path(raw) / 'ENVIRONMENT' / name
    with xr.open_dataset(path, engine='h5netcdf') as ds:
        if ds.annual.dims != ('year', 'lat', 'lon') or ds.distance_to_coast.dims != ('lat', 'lon'):
            raise ValueError(f'Unexpected grid dimensions in {path}')
        if not np.array_equal(ds.year, np.arange(first, 2025)):
            raise ValueError(f'Unexpected year coordinate in {path}')
        return ds.annual.load(), ds.distance_to_coast.load(), ds.attrs


def map_statistics(annual, trend_and_p):
    smooth = annual.rolling(year=5, center=True, min_periods=5).mean().dropna('year', how='all')
    return trend_and_p(smooth)


def atmos_outputs(raw, variable, trend_and_p, mk_p):
    prefix = variable.lower()
    annual, distance, _ = grid_inputs(raw, f'{prefix}_map.nc', 1980)
    slope, pvalue = map_statistics(annual, trend_and_p)
    profiles = fit_profiles(read_series(raw, f'{prefix}_profiles.csv'), mk_p)
    return profiles, slope, pvalue, distance


def sst_map(raw, trend_and_p):
    annual, distance, _ = grid_inputs(raw, 'oisst_map.nc', 1982)
    slope, pvalue = map_statistics(annual, trend_and_p)
    trend = slope.where(distance >= 0).astype('float32')
    return trend, pvalue.where(np.isfinite(trend)), distance


def sst_profiles(raw, source, mk_p):
    table = read_series(raw, 'sst_profiles.csv')
    return fit_profiles(table.loc[table.Source.eq(source)], mk_p).rename(columns={
        'OLS_Trend_per_Decade': 'OLS_Trend_K_per_Decade',
        'OLS_CI95_per_Decade': 'OLS_CI95_K_per_Decade',
    })


def sst_index(raw, source, start_year):
    table = read_series(raw, 'sst_index.csv')
    table = table.loc[table.Source.eq(source) & table.YEAR.ge(start_year), ['YEAR', 'SST_K']]
    checked_series(table, 'YEAR', 'SST_K', start_year, 2024)
    return table.sort_values('YEAR').reset_index(drop=True)


def era5_vws(raw, mk_p):
    annual = read_series(raw, 'era5_vws.csv')
    rows = []
    for center, group in annual.groupby('Distance_Center_km', sort=False):
        years, values = checked_series(group, 'Year', 'Mean_VWS_m_s', 1980, 2014)
        rows.append({'Distance_Center_km': center,
                     'VWS_Trend_m_s_per_decade': stats.linregress(years, values).slope * 10,
                     'MK_P': mk_p(values)})
    return pd.DataFrame(rows), annual


def aod_inputs(raw):
    annual, distance, attrs = grid_inputs(raw, 'aod_map.nc', 1980)
    return annual, distance, attrs['source_variable']
