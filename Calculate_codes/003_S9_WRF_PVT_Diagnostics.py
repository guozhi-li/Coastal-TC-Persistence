#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/WRF'
PVT_ROOT = DEFAULT_RAW_DATA_DIR / 'WRF/PVT'
MAP_FILENAME = 'pvt_tendency_maps_1km_HA_VA_DH_XYZ.nc'
SOURCE_SSTS = ('302K-2K', '302K')
WINDS_MS = tuple(range(34, 41))
DISTANCES_KM = (-50, 0, 50)
HOURS = np.array((3, 6, 9, 12, 15, 18, 21), dtype=int)
HEIGHT_M = 1000.0
AXIS_KM = np.arange(-120.0, 121.0, 5.0)
BASE_VECTOR_KM_PER_MS = 30.0
MAX_DH_CLOSURE_ERROR_MS = 0.1
DH_VECTOR_STEMS = ('dh_x', 'dh_y', 'dh_z', 'dh')
TERMS = (('DH-X', 'dh_x'), ('DH-Y', 'dh_y'), ('DH-Z', 'dh_z'), ('DH', 'dh_reconstructed'))
VECTOR_COLUMNS = ['SST_K', 'Term', 'CX_m_s', 'CY_m_s']


def resolve_pvt_root(raw_data_dir, override=None):
    """Select one PVT tree; never combine cases from different roots."""
    if override is not None:
        root = override.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"PVT directory does not exist: {root}")
        return root
    wrf_dir = raw_data_dir / "WRF"
    candidates = (wrf_dir / "PVT", wrf_dir)
    found = [root for root in candidates
             if any((root / sst).is_dir() for sst in SOURCE_SSTS)]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(
            "PVT scenario directories exist in both WRF/PVT and WRF; "
            "use --pvt-dir to select the intended root."
        )
    raise FileNotFoundError(
        "No PVT scenario directories (302K-2K or 302K) found in: "
        + ", ".join(str(root) for root in candidates)
    )


def mean_valid(values):
    stack = np.asarray(values, dtype=float)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    out = np.full(total.shape, np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    if not valid.any():
        raise ValueError("Composite contains no finite values")
    return out


def case_folder(sst, wind, distance):
    """Resolve current and older PVT layouts, rejecting ambiguous duplicates."""
    if distance == "CTRL":
        options = (
            PVT_ROOT / sst / str(wind) / "CTRL",
            PVT_ROOT / sst / f"{wind}ms" / "CTRL",
            PVT_ROOT / sst / f"CTRL_{wind}ms",
            PVT_ROOT / sst / f"CTRL_{wind}",
        )
    else:
        options = (
            PVT_ROOT / sst / str(wind) / str(distance),
            PVT_ROOT / sst / f"{wind}ms" / str(distance),
        )
    complete = [
        folder for folder in options
        if (folder / "pvt_vectors_1km.csv").is_file()
        and (folder / MAP_FILENAME).is_file()
    ]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Expected one complete PVT folder for {sst}/{wind}/{distance}, "
            f"found {len(complete)}. Checked: {[str(p) for p in options]}"
        )
    return complete[0]


def read_vectors(path):
    columns = [
        "time", "height_m", "center_x_grid", "center_y_grid",
        *[f"{stem}_{component}_ms"
          for stem in DH_VECTOR_STEMS for component in ("cx", "cy")],
    ]
    table = pd.read_csv(path, usecols=columns, low_memory=False)
    if len(table) != len(HOURS):
        raise ValueError(f"{path}: expected {len(HOURS)} rows")
    table["time"] = pd.to_datetime(table["time"], utc=True, errors="raise")
    table = table.sort_values("time").reset_index(drop=True)
    if table["time"].duplicated().any():
        raise ValueError(f"{path}: duplicate diagnostic times")
    if not table["time"].diff().iloc[1:].eq(pd.Timedelta(hours=3)).all():
        raise ValueError(f"{path}: diagnostic times are not 3 h apart")
    if not np.allclose(pd.to_numeric(table["height_m"], errors="raise"), HEIGHT_M):
        raise ValueError(f"{path}: expected 1-km PVT vectors")
    vectors = {}
    for stem in DH_VECTOR_STEMS:
        values = table[[f"{stem}_cx_ms", f"{stem}_cy_ms"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite {stem} vector")
        vectors[stem] = values
    centers = table[["center_x_grid", "center_y_grid"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    if not np.isfinite(centers).all():
        raise ValueError(f"{path}: non-finite 1-km centers")
    return table["time"], vectors, centers


def read_maps(path, sst, wind, distance, expected_times, expected_centers):
    """Read the original time-resolved DH field and validate its CSV alignment."""
    # h5netcdf also supports paths containing non-ASCII characters on Windows.
    with xr.open_dataset(path, engine='h5netcdf') as nc:
        for attr, expected in (
            ('sst_condition', sst),
            ('initial_wind_ms', wind),
            ('initial_distance_km', distance),
        ):
            actual = nc.attrs.get(attr)
            if actual is None or str(actual) != str(expected):
                raise ValueError(f'{path}: {attr}={actual!r}; expected {expected!r}')
        if not np.isclose(float(nc.attrs.get('height_m', np.nan)), HEIGHT_M):
            raise ValueError(f'{path}: expected a 1-km field')
        for axis in ('x', 'y'):
            name = f'{axis}_km'
            if name not in nc or not np.array_equal(nc[name].values, AXIS_KM):
                raise ValueError(f'{path}: unexpected {name} coordinate')
        if 'forecast_hour' not in nc or not np.array_equal(nc.forecast_hour.values, HOURS):
            raise ValueError(f'{path}: expected forecast hours {HOURS.tolist()}')
        if 'time_utc' not in nc:
            raise KeyError(f'{path}: missing time_utc')
        nc_times = pd.to_datetime([str(value) for value in nc.time_utc.values], utc=True)
        if not np.array_equal(nc_times.to_numpy(), expected_times.to_numpy()):
            raise ValueError(f'{path}: NetCDF and CSV diagnostic times differ')
        centers = []
        for axis in ('x', 'y'):
            name = f'center_{axis}_grid'
            if name not in nc:
                raise KeyError(f'{path}: missing {name}')
            centers.append(np.asarray(nc[name].values, dtype=float))
        centers = np.column_stack(centers)
        if centers.shape != expected_centers.shape or not np.allclose(
            centers, expected_centers, atol=0.1, rtol=0
        ):
            raise ValueError(f'{path}: NetCDF and PVT CSV use different 1-km centers')
        if 'dh1' not in nc:
            raise KeyError(f'{path}: missing time-resolved dh1')
        data = np.asarray(nc.dh1.values, dtype=float)
        shape = (len(HOURS), len(AXIS_KM), len(AXIS_KM))
        if data.shape != shape:
            raise ValueError(f'{path}: dh1 shape {data.shape}; expected {shape}')
        if not np.isfinite(data).any():
            raise ValueError(f'{path}: dh1 contains no finite values')
        return {'dh1': data}


def read_case(sst, wind, distance):
    folder = case_folder(sst, wind, distance)
    times, vectors, centers = read_vectors(folder / "pvt_vectors_1km.csv")
    maps = read_maps(
        folder / MAP_FILENAME, sst, wind, distance, times, centers
    )
    return {"times": times, "vectors": vectors, **maps}


def paired_difference(exp, ctrl, label):
    if not exp["times"].equals(ctrl["times"]):
        raise ValueError(f"{label}: EXP and CTRL diagnostic times differ")
    result = {
        "dh1": mean_valid(exp["dh1"] - ctrl["dh1"]),
    }
    for stem in DH_VECTOR_STEMS:
        result[stem] = mean_valid(
            exp["vectors"][stem] - ctrl["vectors"][stem]
        )
    # Decompose each run first, subtract Cartesian vector components, then
    # reconstruct DH. Never subtract speeds or directions.
    result["dh_reconstructed"] = (
        result["dh_x"] + result["dh_y"] + result["dh_z"]
    )
    result["closure_error"] = result["dh_reconstructed"] - result["dh"]
    return result


def composite_sst(sst):
    cases = []
    for wind in WINDS_MS:
        ctrl = read_case(sst, wind, "CTRL")
        for distance in DISTANCES_KM:
            print(f"Pairing {sst}/{wind}ms/{distance} with CTRL", flush=True)
            exp = read_case(sst, wind, distance)
            cases.append(paired_difference(exp, ctrl, f"{sst}/{wind}/{distance}"))
    if len(cases) != len(WINDS_MS) * len(DISTANCES_KM):
        raise ValueError(f"{sst}: wrong number of paired cases")
    result = {
        name: mean_valid([case[name] for case in cases])
        for name in ("dh1", "dh_x", "dh_y", "dh_z", "dh")
    }
    result["dh_reconstructed"] = (
        result["dh_x"] + result["dh_y"] + result["dh_z"]
    )
    error = result["dh_reconstructed"] - result["dh"]
    result["closure_error"] = error
    print(f"{sst}: DH component closure error (x,y) = "
          f"({error[0]:+.4f}, {error[1]:+.4f}) m/s", flush=True)
    if np.max(np.abs(error)) > MAX_DH_CLOSURE_ERROR_MS:
        raise ValueError(
            f"{sst}: DH-X/Y/Z do not close to direct DH within "
            f"{MAX_DH_CLOSURE_ERROR_MS:g} m/s; inspect paired PVT CSVs"
        )
    return result


def arrow_scale(results):
    largest = max(
        float(np.hypot(*results[sst][stem]))
        for sst in SOURCE_SSTS
        for stem in ("dh_x", "dh_y", "dh_z", "dh_reconstructed")
    )
    return min(BASE_VECTOR_KM_PER_MS, 105.0 / largest) if largest else BASE_VECTOR_KM_PER_MS


def build_plot_data(results):
    """Apply the original export precision after recomputing all composites."""
    scale = arrow_scale(results)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid common arrow scale')
    fields, rows = [], []
    for temperature, source_sst in ((300, '302K-2K'), (302, '302K')):
        result = results[source_sst]
        fields.append(np.asarray(result['dh1'], dtype=np.float32))
        for term, stem in TERMS:
            values = np.asarray(result[stem], dtype=np.float32)
            rows.append({'SST_K': temperature, 'Term': term,
                         'CX_m_s': float(values[0]), 'CY_m_s': float(values[1])})
    axis = AXIS_KM.astype(np.float32)
    data = xr.Dataset(
        {'dh1': (('sst', 'y_km', 'x_km'), np.stack(fields))},
        coords={'sst': np.array([300, 302], dtype=np.int32), 'y_km': axis, 'x_km': axis},
        attrs={
            'title': 'Figure S9 paired EXP-minus-CTRL DH plotting data',
            'source_sst_300K': '302K-2K', 'source_sst_302K': '302K',
            'height_m': 1000., 'case_count': 21, 'forecast_hours': '3, 6, 9, 12, 15, 18, 21',
            'arrow_km_per_ms': scale,
            'aggregation': 'At each diagnostic hour, subtract paired CTRL from EXP; average 3-21 h within case, then average 21 cases',
            'vector_definition': 'DH is the stored reconstructed sum of paired DH-X, DH-Y and DH-Z Cartesian motion vectors',
        },
    )
    data.sst.attrs['units'] = 'K'
    for name in ('x_km', 'y_km'):
        data[name].attrs['units'] = 'km'
    data.dh1.attrs.update({'long_name': 'Total-DH wavenumber-1 PVT contribution',
                          'units': '1e-6 K m2 kg-1 s-1 h-1'})
    return data, pd.DataFrame(rows, columns=VECTOR_COLUMNS)



def write_outputs(data, vectors, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_nc = output_dir / 'FigS9_DH_spatial.nc'
    target_csv = output_dir / 'FigS9_DH_vectors.csv'
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.s9_', suffix='.nc', delete=False) as handle:
        temporary_nc = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.s9_', suffix='.csv', delete=False) as handle:
        temporary_csv = Path(handle.name)
    try:
        data.to_netcdf(temporary_nc, engine='h5netcdf', encoding={'dh1': {'zlib': True, 'complevel': 4}})
        vectors.to_csv(temporary_csv, index=False, float_format='%.17g')
        with xr.open_dataset(temporary_nc, engine='h5netcdf') as saved:
            xr.testing.assert_equal(saved.load(), data)
        saved_vectors = pd.read_csv(temporary_csv, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved_vectors, vectors, check_exact=True)
        temporary_nc.replace(target_nc)
        temporary_csv.replace(target_csv)
    finally:
        temporary_nc.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
    print(f'Saved: {target_nc}')
    print(f'Saved: {target_csv}')


def main(argv=None):
    global PVT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir',
                        type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Root containing the original WRF calculation inputs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--pvt-dir', type=Path,
                        help='Optional directory containing the 302K-2K and 302K PVT folders.')
    args = parser.parse_args(argv)
    PVT_ROOT = resolve_pvt_root(args.raw_data_dir.expanduser().resolve(), args.pvt_dir)
    print(f'PVT inputs: {PVT_ROOT}', flush=True)
    results = {sst: composite_sst(sst) for sst in SOURCE_SSTS}
    data, vectors = build_plot_data(results)
    write_outputs(data, vectors, args.output_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
