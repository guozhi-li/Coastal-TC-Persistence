#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/URBAN'
GISA_RELATIVE = Path('GISA/GISA_impervious_fraction_1985_2021_0p01deg.nc')
GISA_REGIONAL = {
    'East Asia': Path('GISA/GISA_impervious_fraction_2021_East_Asia_0p01deg.nc'),
    'North America': Path('GISA/GISA_impervious_fraction_2021_North_America_0p01deg.nc'),
}
COAST_RELATIVE = Path('GSHHG/GSHHS_f_L1_1400.shp')
COAST_FILE = DEFAULT_RAW_DATA_DIR / COAST_RELATIVE


REGION_EXTENTS = {
    'East Asia': (100., 148., 15., 45.),
    'North America': (-130., -65., 15., 45.),
}

RATIO_LAT_BOUNDS = (20., 40.)

BIN_ORDER = ('-100-0 km', '-300--100 km')

REGIONS = {'A': ('East Asia', (100., 148., 15., 45.)),
           'B': ('North America', (-130., -65., 15., 45.))}

CONTOUR_STRIDE = 10

RATIO_COLUMNS = ['region', 'distance_band', 'average_impervious_area_ratio']

def open_source_netcdf(path, **kwargs):
    """Read NetCDF3/HDF5 paths safely, including non-ASCII Windows paths."""
    path = Path(path)
    with path.open('rb') as handle:
        signature = handle.read(8)
    kwargs.setdefault('engine', 'scipy' if signature.startswith(b'CDF') else 'h5netcdf')
    return xr.open_dataset(path, **kwargs)

def unit_sphere_xyz(lat, lon):
    lat = np.deg2rad(np.asarray(lat, float))
    lon = np.deg2rad(np.asarray(lon, float))
    return np.column_stack((np.cos(lat)*np.cos(lon),
                            np.cos(lat)*np.sin(lon), np.sin(lat)))

def coordinate_edges(values):
    values = np.asarray(values, float)
    mid = (values[:-1]+values[1:])/2
    return np.r_[values[0]-(mid[0]-values[0]), mid,
                 values[-1]+(values[-1]-mid[-1])]

def read_gisa_box(dataset, box):
    west,east,south,north = box
    # Four-degree margin lets nearby coastlines set distance at the map edge.
    data = dataset.impervious_fraction
    if data.lat.values[0] > data.lat.values[-1]:
        data = data.sortby('lat')
    if data.lon.values[0] > data.lon.values[-1]:
        data = data.sortby('lon')
    fraction = data.sel(
        year=2021,lat=slice(south-4,north+4),
        lon=slice(west-5,east+5)).squeeze(drop=True)
    lat = np.asarray(fraction.lat.values, float)
    lon = np.asarray(fraction.lon.values, float)
    if lat[0] > lat[-1]:
        fraction = fraction.sortby('lat')
        lat = lat[::-1]
    if lon[0] > lon[-1]:
        fraction = fraction.sortby('lon')
        lon = lon[::-1]
    if fraction.size == 0:
        raise ValueError(f'GISA subset empty: {box}')
    return fraction.load()

def coastal_geometries(bounds):
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        gpd = None
    if gpd is not None:
        coast = gpd.read_file(COAST_FILE, bbox=bounds)
        if coast.crs is not None and coast.crs.to_epsg() != 4326:
            coast = coast.to_crs(4326)
        if coast.empty:
            raise ValueError('No GSHHG polygons in selected map')
        yield from (geometry for geometry in coast.geometry if geometry is not None)
        return

    import shapefile
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds, transform_geom
    wgs84 = CRS.from_epsg(4326)
    source_crs = CRS.from_wkt(COAST_FILE.with_suffix('.prj').read_text())
    query_bounds = bounds if source_crs == wgs84 else transform_bounds(wgs84, source_crs, *bounds)
    with shapefile.Reader(str(COAST_FILE)) as coast:
        for shape in coast.iterShapes(bbox=query_bounds):
            if shape is not None and shape.shapeType != shapefile.NULL:
                geometry = shape.__geo_interface__
                yield geometry if source_crs == wgs84 else transform_geom(source_crs, wgs84, geometry)

def land_mask_from_gshhg(lat, lon):
    import rasterio.features
    from rasterio.transform import from_bounds
    west,east = coordinate_edges(lon)[[0,-1]]
    south,north = coordinate_edges(lat)[[0,-1]]
    bounds = (west,south,east,north)
    transform = from_bounds(*bounds,len(lon),len(lat))
    top_down = rasterio.features.rasterize(
        ((geometry,1) for geometry in coastal_geometries(bounds)),
        out_shape=(len(lat),len(lon)),transform=transform,
        fill=0,dtype='uint8',all_touched=False)
    return np.flipud(top_down).astype(bool)

def signed_land_distance_km(land, lat, lon):
    # Conservative screen: midpoint latitude can omit real 300-km cells near the north edge.
    poleward_lat = np.max(np.abs(lat))
    approx = distance_transform_edt(
        land,sampling=(111.195*abs(lat[1]-lat[0]),
                       111.195*np.cos(np.deg2rad(poleward_lat))*abs(lon[1]-lon[0])))
    boundary = binary_dilation(land)^binary_erosion(land)
    by,bx = np.where(boundary)
    tree = cKDTree(unit_sphere_xyz(lat[by],lon[bx]))
    result = np.full(land.shape,np.nan,np.float32)
    rows,cols = np.where(land & (approx<=400))
    for start in range(0,len(rows),200_000):
        end = min(start+200_000,len(rows))
        xyz = unit_sphere_xyz(lat[rows[start:end]],lon[cols[start:end]])
        chord = tree.query(xyz,workers=-1)[0]
        result[rows[start:end],cols[start:end]] = (
            -6371.0088*2*np.arcsin(np.clip(chord/2,0,1)))
    return result

def weighted_fraction(fraction, lat, selected):
    # Exact latitude factor for equal-angle cells on a sphere.
    half = .005
    row_area = np.sin(np.deg2rad(lat+half))-np.sin(np.deg2rad(lat-half))
    valid = selected & np.isfinite(fraction)
    weighted_sum = np.sum(np.where(valid,fraction,0)*row_area[:,None])
    total_area = np.sum(valid*row_area[:,None])
    return float(weighted_sum/total_area) if total_area>0 else np.nan

def prepare_region(dataset, region):
    """Map 15°–45°N; retain 20°–40°N for the reported averages."""
    box = REGION_EXTENTS[region]
    field = read_gisa_box(dataset, box)
    lat = np.asarray(field.lat.values, float)
    lon = np.asarray(field.lon.values, float)
    fraction = np.asarray(field.values, float)
    land = land_mask_from_gshhg(lat, lon)
    distance = signed_land_distance_km(land, lat, lon)
    y = (lat >= box[2]) & (lat <= box[3])
    x = (lon >= box[0]) & (lon <= box[1])
    lat, lon = lat[y], lon[x]
    fraction = fraction[np.ix_(y, x)]
    land = land[np.ix_(y, x)]
    distance = distance[np.ix_(y, x)]

    bands = [
        land & (distance <= 0) & (distance >= -100),
        land & (distance < -100) & (distance >= -300),
    ]
    category = np.full(fraction.shape, np.nan, np.float32)
    eligible = bands[0] | bands[1]
    category[eligible & (fraction <= .2)] = 0
    category[eligible & (fraction > .2)] = 1

    ratio_lat = ((lat >= RATIO_LAT_BOUNDS[0])
                 & (lat <= RATIO_LAT_BOUNDS[1]))
    ratio_bands = [band & ratio_lat[:, None] for band in bands]
    summary = [
        {
            'panel': 'A' if region == 'East Asia' else 'B',
            'record': 'impervious_ratio',
            'region': region,
            'distance_band': name,
            'average_impervious_area_ratio':
                weighted_fraction(fraction, lat, mask),
            'n_0p01_cells': int(mask.sum()),
        }
        for name, mask in zip(BIN_ORDER, ratio_bands)
    ]
    return lat, lon, category, distance, summary

def validate_regular_grid(lat, lon, spacing=0.1):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) > 1 and (not np.allclose(abs(np.diff(lat)), spacing, atol=0.0001, rtol=0)):
        raise ValueError(f'Latitude spacing is not {spacing} degrees')
    if len(lon) > 1 and (not np.allclose(abs(np.diff(np.rad2deg(np.unwrap(np.deg2rad(lon))))), spacing, atol=0.0001, rtol=0)):
        raise ValueError(f'Longitude spacing is not {spacing} degrees')

def validate_coordinates(lat, lon, region):
    west, east, south, north = dict(REGIONS.values())[region]
    for name, values, lower, upper in (('lat', lat, south, north),
                                        ('lon', lon, west, east)):
        values = np.asarray(values, float)
        if (values.ndim != 1 or values.size < 2 or not np.isfinite(values).all()
                or not np.allclose(np.diff(values), .01, atol=1e-4, rtol=0)
                or values[0] < lower - 1e-6 or values[-1] > upper + 1e-6):
            raise ValueError(f'{region}: invalid native 0.01-degree {name} coordinates')

def validate_map(dataset, panel):
    region = REGIONS[panel][0]
    if set(dataset.data_vars) != {'display_class', 'signed_distance_km'}:
        raise ValueError('Urban NC requires only display_class and signed_distance_km')
    if dataset.display_class.dims != ('lat', 'lon'):
        raise ValueError('display_class must retain the native (lat, lon) grid')
    if dataset.signed_distance_km.dims != ('contour_lat', 'contour_lon'):
        raise ValueError('signed_distance_km must use (contour_lat, contour_lon)')
    if (dataset.attrs.get('region') != region or dataset.attrs.get('year') != 2021
            or dataset.attrs.get('urban_threshold') != .2
            or dataset.attrs.get('contour_stride') != CONTOUR_STRIDE):
        raise ValueError(f'{panel}: incompatible region/year/threshold/contour metadata')
    validate_coordinates(dataset.lat.values, dataset.lon.values, region)
    if (not np.array_equal(dataset.contour_lat.values, dataset.lat.values[::CONTOUR_STRIDE])
            or not np.array_equal(dataset.contour_lon.values, dataset.lon.values[::CONTOUR_STRIDE])):
        raise ValueError('Contour coordinates must exactly match every tenth native cell')
    values = dataset.display_class.values
    valid = np.isfinite(values)
    if not valid.any() or np.isinf(values).any() or not np.isin(values[valid], [0., 1.]).all():
        raise ValueError('display_class must contain only 0, 1 and missing values')
    distance = dataset.signed_distance_km.values
    if np.isinf(distance).any() or not np.isfinite(distance).any() or np.nanmax(distance) > 1e-6:
        raise ValueError('Expected original nonpositive inland distances and missing values')
    return dataset

def validate_ratios(table):
    if list(table.columns) != RATIO_COLUMNS:
        raise ValueError(f'Ratio CSV requires exactly {RATIO_COLUMNS}')
    expected = [(region, band) for region, _ in REGIONS.values() for band in BIN_ORDER]
    if list(table[RATIO_COLUMNS[:2]].itertuples(index=False, name=None)) != expected:
        raise ValueError('Ratio CSV must contain the four original region/band rows in order')
    values = table.average_impervious_area_ratio.to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Impervious area ratios must lie within [0, 1]')
    return table

def write_map(path, dataset, panel):
    # Round-trip comparison checks every native class and every saved contour value.
    with tempfile.TemporaryDirectory(prefix='.s10_', dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        dataset.to_netcdf(temporary, engine='h5netcdf', encoding={
            'display_class': {'dtype': 'uint8', '_FillValue': 255, 'zlib': True,
                              'complevel': 4, 'shuffle': True, 'chunksizes': (128, 512)},
            'signed_distance_km': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        })
        with open_source_netcdf(temporary, engine='h5netcdf') as reopened:
            checked = validate_map(reopened.load(), panel)
            xr.testing.assert_equal(dataset, checked)
        temporary.replace(path)
    print(f'Saved {path} ({path.stat().st_size:,} bytes)', flush=True)



def prepare_plot_map(region, lat, lon, category, distance):
    """Retain native classes and sample only the original contour coordinates."""
    dataset = xr.Dataset({
        'display_class': (('lat', 'lon'), category),
        'signed_distance_km': (('contour_lat', 'contour_lon'),
                               distance[::CONTOUR_STRIDE, ::CONTOUR_STRIDE].astype(np.float32)),
    }, coords={'lat': lat, 'lon': lon,
               'contour_lat': lat[::CONTOUR_STRIDE], 'contour_lon': lon[::CONTOUR_STRIDE]})
    dataset.attrs = {
        'region': region, 'year': 2021, 'source_grid': 'native GISA 0.01 degree',
        'urban_threshold': .2, 'contour_stride': CONTOUR_STRIDE,
        'classification': '0: impervious fraction <= 0.2; 1: > 0.2; '
                          'missing: outside the original inland display bands',
        'distance_sampling': 'Original signed distance sampled every tenth native cell; '
                             'identical to the original contour plotting input',
    }
    dataset.display_class.attrs = {'description': dataset.attrs['classification']}
    dataset.signed_distance_km.attrs = {'units': 'km', 'sign_convention': 'negative over land'}
    return dataset


def main(argv=None):
    global COAST_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Root containing the GISA and GSHHG source directories.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Destination for the three Database/URBAN plotting inputs.')
    args = parser.parse_args(argv)
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    COAST_FILE = raw_data_dir / COAST_RELATIVE
    gisa_sources = {region: raw_data_dir / path for region, path in GISA_REGIONAL.items()}
    global_source = raw_data_dir / GISA_RELATIVE
    if global_source.is_file():
        for region, path in gisa_sources.items():
            if not path.is_file():
                gisa_sources[region] = global_source
    required = list(gisa_sources.values()) + [COAST_FILE.with_suffix(suffix)
                             for suffix in ('.shp', '.shx', '.dbf', '.prj')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing source inputs:\n  ' + '\n  '.join(missing))
    ratio_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for panel, (region, _) in REGIONS.items():
        path = gisa_sources[region]
        with open_source_netcdf(path, decode_times=False, mask_and_scale=True) as dataset:
            if 'impervious_fraction' not in dataset:
                raise KeyError(f'Missing impervious_fraction: {path}')
            if 2021 not in np.asarray(dataset.year.values, int):
                raise ValueError(f'GISA missing year 2021: {path}')
            validate_regular_grid(dataset.lat.values, dataset.lon.values, spacing=.01)
            print(f'Calculating S10 {region} from GISA and GSHHG.', flush=True)
            lat, lon, category, distance, rows = prepare_region(dataset, region)
            result = prepare_plot_map(region, lat, lon, category, distance)
            validate_map(result, panel)
            write_map(output_dir / f'FigS10{panel}_urban.nc', result, panel)
            ratio_rows.extend(rows)
    ratios = validate_ratios(pd.DataFrame(ratio_rows)[RATIO_COLUMNS])
    path = output_dir / 'FigS10_urban_ratios.csv'
    with tempfile.TemporaryDirectory(prefix='.s10_', dir=output_dir) as directory:
        temporary = Path(directory) / path.name
        ratios.to_csv(temporary, index=False, float_format='%.17g')
        checked = pd.read_csv(temporary, float_precision='round_trip')
        pd.testing.assert_frame_equal(ratios, validate_ratios(checked), check_exact=True)
        temporary.replace(path)
    print(f'Saved {path} ({path.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    main()
