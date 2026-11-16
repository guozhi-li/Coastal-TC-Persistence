#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / 'Raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/WRF'


SSTS = ("302K-2K", "302K")

DISPLAY_SST = {"302K-2K": "300 K", "302K": "302 K"}

WINDS_MS = tuple(range(34, 41))

INITIAL_DISTANCES_KM = (-50, 0, 50)

DISTANCES = tuple(str(x) for x in INITIAL_DISTANCES_KM) + ("CTRL",)

HOURS = np.arange(0, 25, 3, dtype=int)

def normalize_distance(series):
    return (series.astype(str).str.strip()
            .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True))


PANEL_CASES = {
    'A': ('302K-2K', 'EXP'), 'B': ('302K-2K', 'CTRL'),
    'C': ('302K', 'EXP'), 'D': ('302K', 'CTRL'),
}
TRACK_COLUMNS = ('panel', 'initial_distance_km', 'initial_wind_ms',
                 'time_h', 'series_type', 'x_km')


def validate_tracks(table):
    """Validate all 504 member and 72 stored-mean coordinates without refitting."""
    if table.columns.duplicated().any() or set(table.columns) != set(TRACK_COLUMNS):
        raise ValueError(f'S8 requires exactly these plotting columns: {TRACK_COLUMNS}')
    table = table.loc[:, list(TRACK_COLUMNS)].copy()
    table['initial_distance_km'] = normalize_distance(table['initial_distance_km'])
    for name in ('initial_wind_ms', 'time_h', 'x_km'):
        table[name] = pd.to_numeric(table[name], errors='raise')
    if len(table) != 576 or not table['series_type'].isin(('member', 'mean')).all():
        raise ValueError('S8 requires 576 coordinates: 504 members and 72 means')
    if not np.isfinite(table[['time_h', 'x_km']].to_numpy(float)).all():
        raise ValueError('S8 time and zonal coordinates must be finite')
    members = table.loc[table.series_type.eq('member')]
    means = table.loc[table.series_type.eq('mean')]
    member_keys = ['panel', 'initial_distance_km', 'initial_wind_ms', 'time_h']
    mean_keys = ['panel', 'initial_distance_km', 'time_h']
    expected_members, expected_means = [], []
    for panel, (_, kind) in PANEL_CASES.items():
        distances = tuple(str(value) for value in INITIAL_DISTANCES_KM) if kind == 'EXP' else ('CTRL',)
        expected_members.extend((panel, distance, wind, hour)
            for distance in distances for wind in WINDS_MS for hour in HOURS)
        expected_means.extend((panel, distance, hour)
            for distance in distances for hour in HOURS)
    if (len(members) != 504 or members.duplicated(member_keys).any()
            or set(members[member_keys].itertuples(index=False, name=None)) != set(expected_members)):
        raise ValueError('S8 requires every panel/distance/wind/hour member exactly once')
    if (len(means) != 72 or means.duplicated(mean_keys).any()
            or set(means[mean_keys].itertuples(index=False, name=None)) != set(expected_means)
            or not means.initial_wind_ms.isna().all()):
        raise ValueError('S8 requires each stored mean once, with initial_wind_ms blank')
    table['_series_order'] = table.series_type.map({'member': 0, 'mean': 1})
    return (table.sort_values(['panel', 'initial_distance_km', '_series_order', 'initial_wind_ms', 'time_h'],
                              kind='stable', na_position='last')
            .drop(columns='_series_order').reset_index(drop=True))


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def read_case_table(path, value_columns):
    keys = ["sst", "initial_wind_ms", "distance", "forecast_hour"]
    columns = keys + list(value_columns)
    table = pd.read_csv(require_file(path), usecols=columns, low_memory=False)
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

def read_tracks(track_file):
    table = read_case_table(
        track_file, ("x_coast_km", "x_from_initial_km")
    )
    exp = table["distance"].ne("CTRL")
    if not np.isfinite(table.loc[exp, "x_coast_km"].to_numpy(float)).all():
        raise ValueError("EXP x_coast_km is incomplete")
    if not np.isfinite(
        table.loc[~exp, "x_from_initial_km"].to_numpy(float)
    ).all():
        raise ValueError("CTRL x_from_initial_km is incomplete")
    return table

def case_series(table, sst, wind, distance, column):
    sub = table.loc[
        table["sst"].eq(sst)
        & table["initial_wind_ms"].eq(wind)
        & table["distance"].eq(str(distance))
    ].sort_values("forecast_hour")
    if len(sub) != len(HOURS) or not np.array_equal(
        sub["forecast_hour"].to_numpy(int), HOURS
    ):
        raise ValueError(f"Incomplete series: {sst}/{wind}/{distance}")
    return sub[column].to_numpy(float)



def prepare_plot_data(track_file):
    tracks = read_tracks(Path(track_file))
    rows = []
    for panel, (sst, kind) in PANEL_CASES.items():
        distances = INITIAL_DISTANCES_KM if kind == 'EXP' else ('CTRL',)
        column = 'x_coast_km' if kind == 'EXP' else 'x_from_initial_km'
        for distance in distances:
            members = []
            for wind in WINDS_MS:
                values = case_series(tracks, sst, wind, distance, column)
                members.append(values)
                rows.extend({'panel': panel, 'initial_distance_km': str(distance),
                             'initial_wind_ms': wind, 'time_h': int(hour),
                             'series_type': 'member', 'x_km': float(value)}
                            for hour, value in zip(HOURS, values))
            rows.extend({'panel': panel, 'initial_distance_km': str(distance),
                         'initial_wind_ms': np.nan, 'time_h': int(hour),
                         'series_type': 'mean', 'x_km': float(value)}
                        for hour, value in zip(HOURS, np.mean(members, axis=0)))
    return validate_tracks(pd.DataFrame(rows, columns=TRACK_COLUMNS))





def write_plot_data(path, table):
    path = Path(path)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', '--database-root', dest='raw_data_dir',
                        type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help='Root containing the original WRF calculation inputs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    track_file = raw_data_dir / 'WRF/WRF_ideal_experience_TC_trajectory.csv'
    output_file = args.output_dir.expanduser().resolve() / 'FigS8_track_coordinates.csv'
    if track_file == output_file:
        raise ValueError('Output must not overwrite the original input')
    table = prepare_plot_data(track_file)
    write_plot_data(output_file, table)
    print(f'Saved 576 plotting coordinates: {output_file}')


if __name__ == '__main__':
    main()
