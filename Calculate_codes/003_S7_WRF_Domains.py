#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/WRF'
INTENSITY_COLUMNS = ['sst', 'initial_wind_ms', 'distance', 'forecast_hour', 'mws_ms', 'mcp_hpa']


SSTS = ("302K-2K", "302K")

DISPLAY_SST = {"302K-2K": "300 K", "302K": "302 K"}

WINDS_MS = tuple(range(34, 41))

INITIAL_DISTANCES_KM = (-50, 0, 50)

DISTANCES = tuple(str(x) for x in INITIAL_DISTANCES_KM) + ("CTRL",)

HOURS = np.arange(0, 25, 1, dtype=int)


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def normalize_distance(series):
    return (series.astype(str).str.strip()
            .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True))

def read_case_table(path, value_columns):
    keys = ["sst", "initial_wind_ms", "distance", "forecast_hour"]
    columns = keys + list(value_columns)
    table = pd.read_csv(require_file(path), usecols=columns, low_memory=False, float_precision="round_trip")
    table["sst"] = table["sst"].astype(str).str.strip()
    table["distance"] = normalize_distance(table["distance"])
    for name in ["initial_wind_ms", "forecast_hour", *value_columns]:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table = table.loc[
        table["sst"].isin(SSTS)
        & table["initial_wind_ms"].isin(WINDS_MS)
        & table["distance"].isin(DISTANCES)
        & table["forecast_hour"].isin(HOURS)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [SSTS, WINDS_MS, DISTANCES, HOURS], names=keys
    )
    actual = pd.MultiIndex.from_frame(table[keys])
    if actual.has_duplicates:
        raise ValueError(f"{path}: duplicate case/time rows")
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(f"{path}: missing case/time rows; first five: {list(missing[:5])}")
    if len(table) != len(expected):
        raise ValueError(f"{path}: expected {len(expected)} selected rows, found {len(table)}")
    return table.sort_values(keys).reset_index(drop=True)


def prepare_intensity(path):
    table = read_case_table(path, ('mws_ms', 'mcp_hpa'))[INTENSITY_COLUMNS]
    if not np.isfinite(table[['mws_ms', 'mcp_hpa']].to_numpy(float)).all():
        raise ValueError('Nonfinite maximum wind or minimum pressure')
    return table


def write_outputs(intensity, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.S7_', dir=output_dir) as temp_dir:
        temporary = Path(temp_dir) / 'FigS7_intensity.csv'
        intensity.to_csv(temporary, index=False, float_format='%.17g')
        saved = pd.read_csv(temporary, float_precision='round_trip')
        pd.testing.assert_frame_equal(intensity, saved, check_exact=True)
        destination = output_dir / temporary.name
        temporary.replace(destination)
        print(f'Saved: {destination} ({destination.stat().st_size:,} bytes)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir',
                        type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Root containing the original WRF calculation inputs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    source_file = raw_data_dir / 'WRF/WRF_ideal_experience_TC_intensity.csv'
    output_dir = args.output_dir.expanduser().resolve()
    if source_file == (output_dir / 'FigS7_intensity.csv').resolve():
        parser.error('Output intensity file must not overwrite the source file')
    intensity = prepare_intensity(source_file)
    write_outputs(intensity, output_dir)


if __name__ == '__main__':
    main()
