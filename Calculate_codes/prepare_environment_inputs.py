#!/usr/bin/env python3
"""Build the reusable seasonal grids and annual series directly from original monthly inputs.

Use --raw-data-dir for the original Raw_data and --output-dir for the target Raw_data/ENVIRONMENT directory.
The signed coastal distances, regional masks, valid-data weights and same-calendar-year
seasons follow the figure calculations. Output NetCDFs retain native grid precision and
CSV uses 17 significant digits so annual area means round trip to binary64.
"""
import argparse
import importlib
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr


def write_grid(out, annual, distance, source, units, **attrs):
    ds = xr.Dataset({'annual': annual, 'distance_to_coast': distance})
    ds.annual.attrs = {'units': units, 'long_name': 'Peak-season mean for the calendar year'}
    ds.distance_to_coast.attrs = {'units': 'km', 'sign_convention': 'negative over land; positive over ocean'}
    ds.attrs.update(source=source, **attrs)
    temporary = out.with_suffix('.nc.tmp')
    ds.to_netcdf(temporary, engine='h5netcdf', encoding={
        name: {'zlib': True, 'complevel': 4, 'shuffle': True} for name in ds.data_vars})
    temporary.replace(out)


def write_table(out, frames):
    temporary = out.with_suffix('.csv.tmp')
    pd.concat(frames, ignore_index=True).to_csv(temporary, index=False, float_format='%.17g')
    temporary.replace(out)


def prepare_atmos(raw, out, variable):
    module = importlib.import_module('002_S4_RH' if variable == 'RH' else '002_S5_VWS')
    series = []
    for product in ('ERA5', 'MERRA2'):
        cache = module.load_product(raw, product)
        north = cache[module.NH_MONTHS]
        south = cache[module.SH_MONTHS]
        dist = module.open_distance(raw / 'DIST2LAND' / f'{product}_distance_to_coast.nc',
                                    north.isel(year=0, drop=True))
        series.append(module.annual_profiles(product, cache, dist))
        if product == 'ERA5':
            field = xr.concat([south.isel(lat=np.flatnonzero(south.lat < 0)),
                               north.isel(lat=np.flatnonzero(north.lat >= 0))], dim='lat').sortby('lat')
            write_grid(out / f'{variable.lower()}_map.nc', field, dist, product,
                       '%' if variable == 'RH' else 'm s-1')
            del field
        del cache, north, south
    write_table(out / f'{variable.lower()}_profiles.csv', series)

def prepare_sst_map(m2, out):
    field_path, distance_path = m2.SST_PRODUCTS['OISST'][:2]
    m2.require_file(field_path)
    with m2.open_netcdf(field_path, chunks='auto', decode_times=True) as ds:
        name = next((key for key in ('sst','SST','sea_surface_temperature')
                     if key in ds.data_vars), None)
        if name is None: raise KeyError(f'Missing SST variable: {field_path}')
        field = m2.canonicalize_sst_field(ds[name], True)
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
        distance=m2.open_sst_distance(distance_path,field)
        annual=annual.where(distance>=0)
        print('Loading annual OISST fields for the five-year spatial trend ...')
        annual=annual.load()
    write_grid(out, annual, distance, "OISST", "degree_C")


def prepare_sst_profiles(m2, source, field_path, distance_path):
    years_for_source = m2.SST_PROFILE_YEARS[source]
    m2.require_file(field_path)
    with m2.open_netcdf(field_path, chunks='auto', decode_times=True) as ds:
        variable = next((name for name in ('sst', 'sea_surface_temperature')
                         if name in ds.data_vars), None)
        if variable is None:
            candidates = [name for name, item in ds.data_vars.items() if item.ndim >= 3]
            if len(candidates) != 1:
                raise KeyError(f'Cannot identify SST in {field_path}')
            variable = candidates[0]
        sst = m2.canonicalize_sst_field(ds[variable], True)
        sst = sst.where(sst.time.dt.year.isin(years_for_source), drop=True)
        sst = sst.sel(lat=slice(-40.0, 40.0))
        sample = float(np.nanmedian(sst.isel(time=0,
                    lat=slice(None, None, 20), lon=slice(None, None, 20)).compute()))
        unit = str(sst.attrs.get('units', '')).strip().lower()
        is_kelvin = unit in {'k', 'kelvin', 'degrees_kelvin', 'degree_kelvin'} or sample > 100
        sst = sst.where((sst > 250) & (sst < 330)) if is_kelvin else \
              sst.where((sst > -5) & (sst < 50)) + 273.15
        distance = m2.open_sst_distance(distance_path, sst)
        exclude = m2.sst_exclusion_mask(distance)
        area = m2.spherical_cell_area(distance.lat, distance.lon)
        rows = []
        for months in ((7, 8, 9, 10, 11), (12, 1, 2, 3, 4)):
            seasonal = sst.where(sst.time.dt.month.isin(months), drop=True)
            counts = pd.Series(seasonal.time.dt.year.values).value_counts()
            bad = {int(y): int(counts.get(y, 0)) for y in years_for_source
                   if int(counts.get(y, 0)) != len(months)}
            if bad:
                raise ValueError(f'{source} missing seasonal months: {bad}')
            annual = seasonal.groupby('time.year').mean('time', skipna=True).astype('float32').load()
            for panel, region_name, region_label, south, north, region_months, season in m2.SST_PROFILE_REGIONS:
                if months != region_months:
                    continue
                region_mask = ((distance.lat >= south) & (distance.lat <= north)).broadcast_like(distance)
                base = region_mask & ~exclude & np.isfinite(distance)
                years = np.asarray(annual.year.values, dtype=float)
                for index, (left, right) in enumerate(zip(m2.SST_PROFILE_EDGES_KM[:-1], m2.SST_PROFILE_EDGES_KM[1:])):
                    in_bin = ((distance >= left) &
                              ((distance <= right) if index == len(m2.SST_PROFILE_EDGES_KM)-2 else (distance < right)))
                    selected = base & in_bin
                    weights = area.where(selected)
                    yearly_mean = ((annual * weights).sum(('lat', 'lon'), skipna=True) /
                                   weights.where(annual.notnull()).sum(('lat', 'lon'), skipna=True))
                    values = np.asarray(yearly_mean.values, dtype=float)
                    valid = np.isfinite(years) & np.isfinite(values)
                    if valid.sum() != len(years_for_source):
                        raise ValueError(f'{source} {region_name} {left:g}–{right:g} km: '
                                         f'only {valid.sum()} valid years')
                    rows.extend({'Source': source, 'Region': region_name, 'Distance_Center_km': (left+right)/2,
                                 'Year': int(y), 'Value': v} for y, v in zip(years, values))
        return pd.DataFrame(rows)


def prepare_aod(s6, database_root, out):
    """Apply the original analysis only to the small native East Asia subset."""
    database_root = Path(database_root)
    path = database_root / 'ENVIRONMENT/MERRA2/MERRA2_gas_1980-2024.nc'
    with s6.open_netcdf(path, chunks={'time': 12}) as dataset:
        variable = s6.choose_data_variable(dataset, s6.AOD_CANDIDATES, 'MERRA-2 AOD')
        aod = s6.canonicalize(dataset[variable])
        for dim in set(aod.dims) - {'time', 'lat', 'lon'}:
            if aod.sizes[dim] != 1:
                raise ValueError(f'Unsupported MERRA-2 AOD dimension: {dim}')
            aod = aod.isel({dim: 0}, drop=True)
        aod = aod.sel(lat=slice(10, 45), lon=slice(100, 145))
        aod = aod.where(((aod.time.dt.year >= 1980) & (aod.time.dt.year <= 2024)), drop=True)
        distance = s6.open_signed_distance(database_root / 'DIST2LAND/MERRA2_distance_to_coast.nc')
        distance = distance.sel(lat=slice(10, 45), lon=slice(100, 145))
        distance = s6.require_same_grid(aod, distance, 'MERRA-2 East Asia AOD')
        annual = s6.complete_season(aod, s6.NH_MONTHS, 1980, 2024, 'MERRA-2 East Asia July-November')
        annual = annual.astype(np.float32).load()
    write_grid(out, annual, distance, "MERRA2", "1", source_variable=variable)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', choices=('RH','VWS','SST','AOD','WNP'),
                        default=['RH','VWS','SST','AOD','WNP'])
    args = parser.parse_args()
    raw, out = args.raw_data_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    m2, s6 = (importlib.import_module(name) for name in ('002_M2_SST','002_S6_VWS_CMIP6_MERRA2'))
    m2.configure_paths(raw)
    s6.configure_paths(raw)
    with dask.config.set(scheduler='synchronous', **{'array.slicing.split_large_chunks': True}):
        for part in args.parts:
            print(f'Preparing {part} ...', flush=True)
            if part in ('RH', 'VWS'):
                prepare_atmos(raw, out, part)
            elif part == 'SST':
                prepare_sst_map(m2, out / 'oisst_map.nc')
                profiles, indices = [], []
                for source, paths in m2.SST_PRODUCTS.items():
                    profiles.append(prepare_sst_profiles(m2, source, *paths))
                    indices.append(m2.coastal_sst_index_from_monthly(source, int(m2.SST_PROFILE_YEARS[source][0])).assign(Source=source))
                write_table(out / 'sst_profiles.csv', profiles)
                write_table(out / 'sst_index.csv', indices)
            elif part == 'WNP':
                _, annual = s6.era5_vws_trends_from_monthly()
                write_table(out / 'era5_vws.csv', [annual])
            else:
                prepare_aod(s6, raw, out / 'aod_map.nc')


if __name__ == '__main__':
    main()
