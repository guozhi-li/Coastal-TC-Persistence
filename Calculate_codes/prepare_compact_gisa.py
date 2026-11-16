#!/usr/bin/env python3
"""Extract the native 2021 GISA windows required by Fig. S10.

Example:
    python Calculate_codes/prepare_compact_gisa.py --raw-data-dir Full_raw_data --output-dir Raw_data

Both arguments refer to roots containing a GISA subdirectory. The two windows
include the four-degree latitude and five-degree longitude coastal-distance
margins used in 004_S10_GISA_EXPOSURE.py. Each original float32 cell, fill value
and coordinate is copied without interpolation, thresholding or quantization.
"""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr


DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / 'Raw_data'
SOURCE_NAME = 'GISA_impervious_fraction_1985_2021_0p01deg.nc'
WINDOWS = {
    'East_Asia': (95, 153, 11, 49),
    'North_America': (-135, -60, 11, 49),
}


def coordinate_slice(coordinate, low, high):
    return slice(low, high) if coordinate[0] < coordinate[-1] else slice(high, low)


def verify(crop, path):
    """Compare on-disk data to the unmodified source, including fill-value bits."""
    with xr.open_dataset(path, engine='h5netcdf', decode_cf=False) as stored:
        for name in crop.variables:
            original, copied = crop[name], stored[name]
            if original.dtype != copied.dtype or original.dims != copied.dims or original.shape != copied.shape:
                raise ValueError(f'{path.name}: incompatible representation of {name}')
            np.testing.assert_equal(original.attrs, copied.attrs,
                                    err_msg=f'{path.name}: attributes differ for {name}')
            for first in range(0, original.sizes.get('lat', 1), 128):
                selection = {'lat': slice(first, first + 128)} if 'lat' in original.dims else {}
                if original.isel(selection).values.tobytes() != copied.isel(selection).values.tobytes():
                    raise ValueError(f'{path.name}: {name} differs at latitude block {first}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=DEFAULT_RAW_DIR,
                        help='Input root containing GISA/' + SOURCE_NAME)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RAW_DIR,
                        help='Output root for the two GISA regional NetCDF files.')
    args = parser.parse_args()
    source = args.raw_data_dir.expanduser().resolve() / 'GISA' / SOURCE_NAME
    output = args.output_dir.expanduser().resolve() / 'GISA'
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(source, engine='h5netcdf', decode_cf=False) as full, \
         tempfile.TemporaryDirectory(dir=output) as staging:
        fraction = full[['impervious_fraction']].sel(year=[2021])
        for region, (west, east, south, north) in WINDOWS.items():
            crop = fraction.sel(lat=coordinate_slice(full.lat.values, south, north),
                                lon=coordinate_slice(full.lon.values, west, east)).drop_encoding()
            if crop.sizes['lat'] != (north - south) * 100 or crop.sizes['lon'] != (east - west) * 100:
                raise ValueError(f'The source does not contain the complete native {region} window.')
            crop.attrs.update(geospatial_lon_min=west, geospatial_lon_max=east,
                              geospatial_lat_min=south, geospatial_lat_max=north,
                              temporal_coverage='2021',
                              subset='Figure S10 native-grid source window, including the coastal-distance margin')
            name = f'GISA_impervious_fraction_2021_{region}_0p01deg.nc'
            path = Path(staging) / name
            encoding = {'impervious_fraction': {'zlib': True, 'shuffle': True, 'complevel': 7,
                                                 'chunksizes': (1, 128, 512)}}
            for coordinate in crop.coords:
                if '_FillValue' not in crop[coordinate].attrs:
                    encoding[coordinate] = {'_FillValue': None}
            crop.to_netcdf(path, engine='h5netcdf', encoding=encoding)
            verify(crop, path)
            path.replace(output / name)
            print(f'Verified {output / name} ({(output / name).stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    main()
