#!/usr/bin/env python3

import argparse

import math

from pathlib import Path

import tempfile

import numpy as np

import pandas as pd

import pymannkendall as mk

from scipy import stats
from matplotlib.path import Path as MplPath

PROJECT_DIR = Path(__file__).resolve().parent.parent

GROUP_LABELS = ('>20°N', '0–20°N', 'SH')

DISTANCE_LABELS = ('−300–−100', '±100', '100–300')

BASIN_LABELS = ('WNP', 'EP', 'WNA')

PLOT_COLUMNS = (
    'Panel', 'X_Label', 'Metric', 'Trend_per_Decade', 'CI95_per_Decade',
    'MK_P', 'Unit', 'Mean_point_exposure_persons', 'CI95_low', 'CI95_high',
)

def validate_plot_data(table):
    """Validate the ten plotting columns and return A/B/C/D category/metric order.

    MK P values are finite for A-C and blank for D. Numerical values
    retain their original units: trend CIs are half-widths, while panel D CIs
    are absolute lower and upper bounds. Panel D uses bootstrap intervals rather than a trend test.
    """
    if table.columns.duplicated().any():
        raise ValueError('S1 plotting data contain duplicate column names')
    missing = set(PLOT_COLUMNS) - set(table.columns)
    extra = set(table.columns) - set(PLOT_COLUMNS)
    if missing or extra:
        raise ValueError(
            f'S1 requires exactly ten plotting columns; '
            f'missing={sorted(missing)}, unexpected={sorted(extra)}'
        )
    table = table.loc[:, list(PLOT_COLUMNS)].copy()
    # Accept the former 12-row label spelling without changing its statistics.
    is_latitude_panel = table['Panel'].eq('A')
    table.loc[is_latitude_panel, 'X_Label'] = table.loc[
        is_latitude_panel, 'X_Label'
    ].replace({'NH (> 20°)': '>20°N', 'NH (≤ 20°)': '0–20°N'})
    counts = table['Panel'].value_counts(dropna=False).to_dict()
    if len(table) != 12 or counts != {'A': 3, 'B': 3, 'C': 3, 'D': 3}:
        raise ValueError('S1 requires exactly 12 rows: A=3, B=3, C=3, D=3; regenerate the former 18-row table with the S1 calculation script')

    keys = ['Panel', 'X_Label', 'Metric']
    expected = (
        [('A', label, 'CLTC count') for label in GROUP_LABELS]
        + [('B', label, 'CLTC count') for label in BASIN_LABELS]
        + [('C', label, 'CLTC count') for label in DISTANCE_LABELS]
        + [('D', label, 'Mean point exposure') for label in DISTANCE_LABELS]
    )
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError('S1 panel/category/metric keys must be present and unique')
    observed = set(table[keys].itertuples(index=False, name=None))
    if observed != set(expected):
        raise ValueError('S1 has unexpected or missing panel/category/metric combinations')

    expected_units = {
        'CLTC count': 'decade^-1',
        'Mean point exposure': 'persons per 3-h TC point',
    }
    if not table['Unit'].eq(table['Metric'].map(expected_units)).all():
        raise ValueError('S1 units do not match the original plotted metrics')

    is_exposure = table['Panel'].eq('D')
    table['MK_P'] = pd.to_numeric(table['MK_P'], errors='raise')
    p_values = table.loc[~is_exposure, 'MK_P']
    if not np.isfinite(p_values.to_numpy(dtype=float)).all() or not p_values.between(0, 1).all():
        raise ValueError('Panels A-C require genuine finite MK P values within [0, 1]')
    if not table.loc[is_exposure, 'MK_P'].isna().all():
        raise ValueError('Panel D uses bootstrap intervals and must leave MK_P blank')

    trend_columns = ['Trend_per_Decade', 'CI95_per_Decade']
    exposure_columns = ['Mean_point_exposure_persons', 'CI95_low', 'CI95_high']
    for column in trend_columns + exposure_columns:
        table[column] = pd.to_numeric(table[column], errors='raise')
    trends = table.loc[~is_exposure, trend_columns]
    exposure = table.loc[is_exposure, exposure_columns]
    if not np.isfinite(trends.to_numpy(dtype=float)).all():
        raise ValueError('Panels A-C require finite trends and CI half-widths')
    if trends['CI95_per_Decade'].lt(0).any():
        raise ValueError('Trend CI half-widths must be nonnegative')
    if not np.isfinite(exposure.to_numpy(dtype=float)).all():
        raise ValueError('Panel D requires finite means and both confidence bounds')
    if (exposure['CI95_low'].lt(0).any()
            or exposure['CI95_low'].gt(exposure['Mean_point_exposure_persons']).any()
            or exposure['Mean_point_exposure_persons'].gt(exposure['CI95_high']).any()):
        raise ValueError('Panel D requires 0 <= lower CI <= mean <= upper CI')
    if (not table.loc[is_exposure, trend_columns].isna().all().all()
            or not table.loc[~is_exposure, exposure_columns].isna().all().all()):
        raise ValueError('Numerical fields not used by a panel must remain blank')

    order = pd.MultiIndex.from_tuples(expected, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(PLOT_COLUMNS)]

YEAR_MIN, YEAR_MAX = 1980, 2024

WIND_LIMIT_KT = 34.0

LINGER_HOURS = 24

TIME_STEP_HOURS = 3

LINGER_STEPS = math.ceil(LINGER_HOURS / TIME_STEP_HOURS) + 1

REGION_CASES = (
    {
        "key": "NH20",
        "label": ">20°N",
        "selector": lambda x: x["LAT"] > 20.0,
    },
    {
        "key": "NH0_20",
        "label": "0–20°N",
        "selector": lambda x: x["LAT"].between(0.0, 20.0, inclusive="both"),
    },
    {
        "key": "SH",
        "label": "SH",
        "selector": lambda x: x["LAT"] < 0.0,
    },
)

DISTANCE_CASES = (
    {
        "key": "LAND_m300_m100",
        "label": "−300–−100",
        "selector": lambda x: x["CAL_DIST"].between(-300.0, -100.0, inclusive="both"),
    },
    {
        "key": "COAST_pm100",
        "label": "\N{PLUS-MINUS SIGN}100",
        "selector": lambda x: x["CAL_DIST"].abs() < 100.0,
    },
    {
        "key": "OCEAN_100_300",
        "label": "100–300",
        "selector": lambda x: x["CAL_DIST"].between(100.0, 300.0, inclusive="both"),
    },
)

def read_ibtracs(input_file):
    if not input_file.is_file():
        raise FileNotFoundError(f"Missing input: {input_file}")
    required = ["SID", "ISO_TIME", "LAT", "LON", "USA_WIND", "CAL_DIST"]
    frame = pd.read_csv(
        input_file,
        usecols=required,
        keep_default_na=False,
        low_memory=False,
    )
    for column in ["LAT", "LON", "USA_WIND", "CAL_DIST"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    frame["SID"] = frame["SID"].astype(str).str.strip()
    frame = frame[
        frame["SID"].ne("")
        & frame["ISO_TIME"].notna()
        & frame["ISO_TIME"].dt.year.between(
            YEAR_MIN, YEAR_MAX, inclusive="both"
        )
    ].copy()
    frame = frame.dropna(subset=["LAT", "LON", "CAL_DIST"])
    frame = frame.sort_values(["SID", "ISO_TIME"])
    frame = frame.drop_duplicates(["SID", "ISO_TIME"], keep="first")
    # Missing winds do not qualify a SID; the lifetime maximum must be >=34 kt.
    # Once a SID qualifies, its missing-wind track points remain in the LMI analysis.
    sid_lmi = frame.groupby("SID")["USA_WIND"].max()
    eligible_sids = sid_lmi[sid_lmi >= WIND_LIMIT_KT].index
    frame = frame[frame["SID"].isin(eligible_sids)].copy()
    frame["LMI_WIND"] = frame["SID"].map(sid_lmi)
    if not (frame["CAL_DIST"].lt(0).any() and frame["CAL_DIST"].gt(0).any()):
        raise ValueError(
            "CAL_DIST must contain both negative land and positive ocean values."
        )
    print(f"Input: {input_file}")
    print(f"Eligible SIDs (lifetime maximum USA_WIND >= 34 kt): {len(eligible_sids):,}")
    print(f"Retained track points: {len(frame):,}")
    print(f"Retained points with missing USA_WIND: {frame['USA_WIND'].isna().sum():,}")
    return frame

def annual_event_counts(frame, mask, case_key, panel_key):
    selected = frame.loc[mask(frame)].copy()
    event_rows = []
    for sid, storm in selected.groupby("SID", sort=False):
        storm = storm.sort_values("ISO_TIME")
        block_ids = (
            storm["ISO_TIME"].diff()
            .ne(pd.Timedelta(hours=TIME_STEP_HOURS))
            .cumsum()
        )
        event_number = 1
        for _, block in storm.groupby(block_ids, sort=False):
            block = block.sort_values("ISO_TIME")
            if len(block) < LINGER_STEPS:
                continue
            start = block["ISO_TIME"].iloc[0]
            end = block["ISO_TIME"].iloc[-1]
            event_rows.append({
                "Panel": panel_key,
                "Case": case_key,
                "SID": sid,
                "Event_ID": f"{panel_key}_{case_key}_{sid}_E{event_number:03d}",
                "ISO_YEAR": int(start.year),
                "Start_Time": start,
                "End_Time": end,
                "Duration_Hours": (end - start).total_seconds() / 3600.0,
                "Obs_Count": int(len(block)),
            })
            event_number += 1
    events = pd.DataFrame(event_rows)
    years = np.arange(YEAR_MIN, YEAR_MAX + 1, dtype=int)
    if events.empty:
        counts = pd.Series(0, index=years, dtype=int)
    else:
        counts = (
            events.groupby("ISO_YEAR").size()
            .reindex(years, fill_value=0)
            .astype(int)
        )
    annual = pd.DataFrame({
        "Panel": panel_key,
        "Case": case_key,
        "ISO_YEAR": years,
        "Count": counts.to_numpy(),
    })
    return events, annual

def trend_statistics(annual):
    years = annual["ISO_YEAR"].to_numpy(dtype=float)
    counts = annual["Count"].to_numpy(dtype=float)
    ols = stats.linregress(years, counts)
    tcrit = stats.t.ppf(0.975, len(years) - 2)
    if np.unique(counts).size <= 1:
        mk_z, mk_p, mk_trend = 0.0, 1.0, "no trend"
    else:
        result = mk.original_test(counts)
        mk_z = float(result.z)
        mk_p = float(result.p)
        mk_trend = result.trend
    return {
        "Total_Events": int(np.rint(counts.sum())),
        "Trend_events_decade": float(ols.slope * 10.0),
        "CI95_events_decade": float(tcrit * ols.stderr * 10.0),
        "OLS_P": float(ols.pvalue),
        "MK_Z": mk_z,
        "MK_P": mk_p,
        "MK_Trend": mk_trend,
    }

def summarize_cases(frame, cases, panel_key, common_mask=None):
    summaries = []
    annual_tables = []
    event_tables = []
    for case in cases:
        def combined_mask(x, case_selector=case["selector"]):
            chosen = case_selector(x)
            if common_mask is not None:
                chosen = chosen & common_mask(x)
            return chosen
        events, annual = annual_event_counts(
            frame, combined_mask, case["key"], panel_key
        )
        statistics = trend_statistics(annual)
        summaries.append({
            "Panel": panel_key,
            "Case": case["key"],
            "Label": case["label"].replace("\n", " "),
            **statistics,
        })
        annual_tables.append(annual)
        if not events.empty:
            event_tables.append(events)
    summary = pd.DataFrame(summaries)
    annual_all = pd.concat(annual_tables, ignore_index=True)
    if event_tables:
        events_all = pd.concat(event_tables, ignore_index=True)
    else:
        events_all = pd.DataFrame()
    return summary, annual_all, events_all











BASIN_DISTANCE_KM = 100.0
BASIN_WNA_VERTS = [
    (-105, 30), (-98, 18), (-92, 16.6), (-90, 15), (-85, 13),
    (-83, 9), (-81, 8), (-79.3, 9.3), (-78, 8.8), (-76, 6),
    (-48, 6), (-48, 60), (-105, 60),
]
BASIN_EP_VERTS = [
    (-105.0, 30.0), (-98.0, 18.0), (-92.0, 16.6), (-90.0, 15.0),
    (-85.0, 13.0), (-83.0, 9.0), (-81.0, 8.0), (-79.3, 9.3),
    (-78.0, 8.8), (-76.0, 6.0), (-130.0, 6.0), (-130.0, 37.0),
    (-105.0, 37.0),
]
BASIN_WNP_VERTS = [
    (102.0, 15.0), (110.0, 8.0), (113.0, 5.0), (148.0, 5.0),
    (148.0, 55.0), (102.0, 55.0), (102.0, 15.0),
]

BASIN_PATH_WNA = MplPath(BASIN_WNA_VERTS)
BASIN_PATH_EP = MplPath(BASIN_EP_VERTS)
BASIN_PATH_WNP = MplPath(BASIN_WNP_VERTS)


def basin_first_present(columns, aliases):
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None

def basin_point_region(lon, lat):
    point = (lon, lat)
    if BASIN_PATH_WNA.contains_point(point):
        return "WNA"
    if BASIN_PATH_EP.contains_point(point):
        return "EP"
    if BASIN_PATH_WNP.contains_point(point):
        return "WNP"
    return "OTHER"

def read_basin_tracks(input_file):
    """Load processed tracks and apply the whole-SID >=34-kt eligibility rule."""
    input_file = Path(input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Missing input: {input_file}")

    raw = pd.read_csv(input_file, low_memory=False, keep_default_na=False)
    aliases = {
        "SID": ["SID", "sid"],
        "LON": ["LON", "lon"],
        "LAT": ["LAT", "lat"],
        "USA_WIND": ["USA_WIND", "WMO_WIND", "usa_wind", "wind"],
        "CAL_DIST": ["CAL_DIST", "cal_dist"],
        "ISO_TIME": ["ISO_TIME", "time", "iso_time"],
    }
    mapped = {target: basin_first_present(raw.columns, choices) for target, choices in aliases.items()}
    missing = [target for target, source in mapped.items() if source is None]
    if missing:
        raise KeyError(
            "Required columns missing: " + ", ".join(missing)
            + f"\nAvailable columns preview: {list(raw.columns)[:25]} ..."
        )

    data = raw[[mapped[name] for name in aliases]].copy()
    data.columns = list(aliases)
    data["SID"] = data["SID"].astype(str).str.strip()
    data = data[data["SID"].ne("")].copy()
    for column in ["LON", "LAT", "USA_WIND", "CAL_DIST"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], errors="coerce")

    # Missing winds do not qualify a storm, but remain in a qualifying LMI track.
    sid_max_wind = (
        data.assign(_WIND_FOR_FILTER=data["USA_WIND"].fillna(0.0))
        .groupby("SID")["_WIND_FOR_FILTER"]
        .max()
    )
    eligible_sids = sid_max_wind[sid_max_wind >= WIND_LIMIT_KT].index
    data = data[data["SID"].isin(eligible_sids)].copy()
    data = data[
        data["ISO_TIME"].notna()
        & data["ISO_TIME"].dt.year.between(YEAR_MIN, YEAR_MAX, inclusive="both")
    ].copy()
    data = data.dropna(subset=["SID", "LON", "LAT", "CAL_DIST", "ISO_TIME"]).copy()
    data["LMI_WIND"] = data["SID"].map(sid_max_wind)
    data["point_region"] = [
        basin_point_region(lon, lat) for lon, lat in zip(data["LON"], data["LAT"])
    ]
    return data

def basin_region_subset(data, region):
    """Use the same point-level regional cuts as 001_CAL_CLTC."""
    if region == "Global":
        return data.copy()
    if region == "NH":
        return data[data["LAT"] > 0].copy()
    if region == "NH20":
        return data[data["LAT"] > 20].copy()
    if region.endswith("20"):
        basin = region[:-2]
        return data[(data["point_region"] == basin) & (data["LAT"] > 20)].copy()
    raise ValueError(f"Unknown region: {region}")

def annual_basin_counts(data, region):
    """Count 24-h LMI CLTC blocks within |CAL_DIST| < 100 km by start year."""
    selected = basin_region_subset(data, region)
    selected = selected[
        (selected["LMI_WIND"] >= WIND_LIMIT_KT)
        & (selected["CAL_DIST"].abs() < BASIN_DISTANCE_KM)
    ].copy()

    event_years = []
    for _, group in selected.groupby("SID", sort=False):
        ordered = group.sort_values("ISO_TIME")
        block_ids = ordered["ISO_TIME"].diff().ne(
            pd.Timedelta(hours=TIME_STEP_HOURS)
        ).cumsum()
        for _, block in ordered.groupby(block_ids, sort=False):
            if len(block) >= LINGER_STEPS:
                event_years.append(int(block["ISO_TIME"].iloc[0].year))

    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    counts = (
        pd.Series(event_years, dtype="int64")
        .value_counts()
        .reindex(years, fill_value=0)
        .sort_index()
        .to_numpy(dtype=float)
    )
    return years, counts, len(event_years)

def basin_mann_kendall_p(values):
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

def basin_trend(years, counts):
    result = stats.linregress(years, counts)
    tcrit = stats.t.ppf(0.975, len(years) - 2)
    error = tcrit * result.stderr * 10
    return result.slope * 10, error, basin_mann_kendall_p(counts)


def summarize_basins(input_file):
    tracks = read_basin_tracks(input_file)
    rows = []
    for key, label in (("WNP20", "WNP"), ("EP20", "EP"), ("WNA20", "WNA")):
        years, counts, event_count = annual_basin_counts(tracks, key)
        slope, error, mk_p = basin_trend(years, counts)
        rows.append({"Panel": "S1b", "Case": key, "Label": label,
                     "Total_Events": event_count, "Trend_events_decade": slope,
                     "CI95_events_decade": error, "MK_P": mk_p})
    return pd.DataFrame(rows)



def calculate_trend_statistics(database_root):
    """Calculate latitude, basin and distance-band trends independently."""
    input_file = Path(database_root) / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv'
    frame = read_ibtracs(input_file)
    summary_a, _, _ = summarize_cases(
        frame, REGION_CASES, panel_key='S1a',
        common_mask=lambda x: x['CAL_DIST'].abs() < 100.0,
    )
    summary_b = summarize_basins(input_file)
    summary_c, _, _ = summarize_cases(
        frame, DISTANCE_CASES, panel_key='S1c',
        common_mask=lambda x: x['LAT'] > 20.0,
    )
    rows = []
    for panel, summary in (('A', summary_a), ('B', summary_b), ('C', summary_c)):
        for _, row in summary.iterrows():
            rows.append({
                'Panel': panel, 'X_Label': row['Label'], 'Metric': 'CLTC count',
                'Trend_per_Decade': row['Trend_events_decade'],
                'CI95_per_Decade': row['CI95_events_decade'], 'MK_P': row['MK_P'],
            })
    return pd.DataFrame(rows)

def write_plot_data(output_file, table):
    """Atomically replace the plotting CSV after writing a sibling temporary file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', dir=output_file.parent,
            prefix=f'.{output_file.name}.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False, float_format='%.17g', lineterminator='\n')
        temporary.replace(output_file)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


import re
import xarray as xr


def open_netcdf(path, **kwargs):
    """Select a backend that also accepts Unicode filesystem paths."""
    with Path(path).open('rb') as handle:
        signature = handle.read(8)
    engine = 'h5netcdf' if signature.startswith(b'\x89HDF') else 'scipy'
    return xr.open_dataset(path, engine=engine, **kwargs)

PRECIP_ROOT = PROJECT_DIR / 'Raw_data/MSWEP'
POP_ROOT = PROJECT_DIR / 'Raw_data/LandScan'
EXPOSURE_YEAR_MIN, EXPOSURE_YEAR_MAX = 2000, 2024
TC_RADIUS_KM = 500.0
HEAVY_RAIN_MM_3H = 30.0
BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED = 5000, 20261008
PROGRESS_EVERY = 100
EXPOSURE_DISTANCE_BINS = (
    ('-300 to -100', '−300–−100'),
    ('-100 to 100', '±100'),
    ('100 to 300', '100–300'),
)
EXPOSURE_CSV_FILENAME = 'S1D_point_population_exposure.csv'
EXPOSURE_CSV_COLUMNS = (
    'SID', 'ISO_TIME', 'LAT', 'LON', 'CAL_DIST', 'DISTANCE_BIN',
    'POPULATION_YEAR', 'EXPOSURE_PERSON_3H', 'STATUS',
)


def normalize_lon_180(values):
    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0


def exposure_distance_bin(value):
    value = float(value)
    if -300.0 <= value < -100.0:
        return "-300 to -100"
    if -100.0 <= value <= 100.0:
        return "-100 to 100"
    if 100.0 < value <= 300.0:
        return "100 to 300"
    return None


def precipitation_file_for_time(timestamp, mswep_dir=None):
    timestamp = pd.Timestamp(timestamp)
    return (
        (Path(mswep_dir) if mswep_dir is not None else PRECIP_ROOT)
        / f"{timestamp.year:04d}"
        / f"{timestamp.year:04d}{timestamp.dayofyear:03d}.{timestamp.hour:02d}.nc"
    )


def select_exact_precipitation(dataset, timestamp, path):
    if "precipitation" not in dataset or "time" not in dataset.coords:
        raise KeyError(f"Missing precipitation/time: {path}")
    target = np.datetime64(pd.Timestamp(timestamp).to_datetime64(), "ns")
    times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    matches = np.flatnonzero(times == target)
    if len(matches) != 1:
        raise KeyError(f"MSWEP time mismatch: {timestamp}, {path}")
    return dataset.precipitation.isel(time=int(matches[0]), drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    phi1 = np.radians(float(lat1))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlambda = np.radians(
        (np.asarray(lon2, dtype=float) - float(lon1) + 180.0) % 360.0 - 180.0
    )
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def subset_tc_box(data, tc_lat, tc_lon):
    lat = np.asarray(data.lat.values, dtype=float)
    lon = normalize_lon_180(data.lon.values)
    lat_half = TC_RADIUS_KM / 111.0 + 0.2
    lon_half = min(
        180.0,
        TC_RADIUS_KM
        / (111.0 * max(abs(math.cos(math.radians(float(tc_lat)))), 0.05))
        + 0.2,
    )
    rows = np.flatnonzero(np.abs(lat - float(tc_lat)) <= lat_half)
    cols = np.flatnonzero(
        np.abs((lon - float(tc_lon) + 180.0) % 360.0 - 180.0) <= lon_half
    )
    if not len(rows) or not len(cols):
        raise ValueError(f"Empty MSWEP subset at {tc_lat}, {tc_lon}")
    return data.isel(lat=rows, lon=cols)


def discover_population_files(population_dir=None):
    root = Path(population_dir) if population_dir is not None else POP_ROOT
    files = {}
    pattern = re.compile(r"^(\d{4})_population\.nc$")
    for path in sorted(root.glob("*_population.nc")):
        match = pattern.match(path.name)
        if match:
            files[int(match.group(1))] = path
    if not files:
        raise FileNotFoundError(f"No LandScan files: {root}")
    return files


def nearest_population_year(year, available):
    return min(available, key=lambda item: (abs(item - int(year)), item))


def population_on_precip_grid(population, rain):
    lat = np.asarray(rain.lat.values, dtype=float)
    lon = np.asarray(rain.lon.values, dtype=float)
    aligned = population.sel(lat=lat, lon=lon).squeeze(drop=True)
    if aligned.shape != (len(lat), len(lon)):
        raise ValueError("LandScan/MSWEP grid mismatch")
    return np.asarray(aligned.values, dtype=float)


def select_panel_d_points(frame):
    """Select exactly the original 2000-2024, north-of-20N, within-300-km points."""
    required = ['SID', 'ISO_TIME', 'LAT', 'LON', 'CAL_DIST']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing track columns for S1D: {sorted(missing)}')
    source = frame.copy()
    source['SID'] = source['SID'].astype(str).str.strip()
    source['ISO_TIME'] = pd.to_datetime(source['ISO_TIME'], errors='raise', utc=True).dt.tz_convert(None)
    years = source["ISO_TIME"].dt.year
    selected = source[
        years.between(EXPOSURE_YEAR_MIN, EXPOSURE_YEAR_MAX)
        & source["LAT"].gt(20.0)
        & source["CAL_DIST"].abs().le(300.0)
    ].copy()
    selected["DISTANCE_BIN"] = selected["CAL_DIST"].map(exposure_distance_bin)
    selected = selected[selected["DISTANCE_BIN"].notna()].copy()
    if (selected['SID'].eq('').any() or selected['ISO_TIME'].isna().any()
            or selected.duplicated(['SID', 'ISO_TIME']).any()):
        raise ValueError('Expected S1D track-point keys must be present and unique')
    if not np.isfinite(selected[['LAT', 'LON', 'CAL_DIST']].to_numpy(float)).all():
        raise ValueError('Expected S1D coordinates and coastal distances must be finite')
    return selected.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)


def calculate_panel_d_points(frame, population_files, mswep_dir=None):
    """Calculate every selected point, retaining zero exposure and its provenance."""
    selected = select_panel_d_points(frame)

    available = sorted(population_files)
    cache = {}
    records = []

    def population_for_year(year):
        mapped = nearest_population_year(year, available)
        if mapped not in cache:
            dataset = open_netcdf(
                population_files[mapped], mask_and_scale=True
            )
            if "population" not in dataset:
                dataset.close()
                raise KeyError(f"Missing population: {population_files[mapped]}")
            cache[mapped] = dataset
        return cache[mapped].population.squeeze(drop=True)

    try:
        for number, row in enumerate(selected.itertuples(index=False), start=1):
            exposure = np.nan
            status = "OK"
            try:
                path = precipitation_file_for_time(row.ISO_TIME, mswep_dir)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with open_netcdf(
                    path, decode_times=True, mask_and_scale=True
                ) as dataset:
                    rain = subset_tc_box(
                        select_exact_precipitation(dataset, row.ISO_TIME, path),
                        row.LAT,
                        row.LON,
                    )
                    rain_values = np.asarray(rain.values, dtype=float)
                    lon_grid, lat_grid = np.meshgrid(
                        normalize_lon_180(rain.lon.values), rain.lat.values
                    )
                    in_radius = (
                        haversine_km(
                            row.LAT, row.LON, lat_grid, lon_grid
                        )
                        <= TC_RADIUS_KM
                    )
                    population = population_on_precip_grid(
                        population_for_year(row.ISO_TIME.year), rain
                    )
                    heavy = (
                        in_radius
                        & np.isfinite(rain_values)
                        & (rain_values >= HEAVY_RAIN_MM_3H)
                        & np.isfinite(population)
                        & (population > 0)
                    )
                    exposure = float(population[heavy].sum())
            except Exception as exc:
                raise RuntimeError(f"S1D input failed for SID={row.SID}, time={row.ISO_TIME}") from exc
            records.append(
                {
                    "SID": row.SID,
                    "ISO_TIME": row.ISO_TIME,
                    "LAT": float(row.LAT),
                    "LON": float(row.LON),
                    "CAL_DIST": float(row.CAL_DIST),
                    "DISTANCE_BIN": row.DISTANCE_BIN,
                    "POPULATION_YEAR": nearest_population_year(row.ISO_TIME.year, available),
                    "EXPOSURE_PERSON_3H": exposure,
                    "STATUS": status,
                }
            )
            if number % PROGRESS_EVERY == 0 or number == len(selected):
                print(f"[Panel D, >20°N] {number:,}/{len(selected):,}")
    finally:
        for dataset in cache.values():
            dataset.close()
    return pd.DataFrame(records, columns=EXPOSURE_CSV_COLUMNS)


def validate_exposure_rows(table, frame):
    """Require complete point keys, track metadata and valid population exposures."""
    if table.columns.duplicated().any() or set(table.columns) != set(EXPOSURE_CSV_COLUMNS):
        raise ValueError(f'S1D exposure CSV requires exactly these columns: {EXPOSURE_CSV_COLUMNS}')
    table = table.loc[:, list(EXPOSURE_CSV_COLUMNS)].copy()
    table['SID'] = table['SID'].astype(str).str.strip()
    table['ISO_TIME'] = pd.to_datetime(table['ISO_TIME'], errors='raise', utc=True).dt.tz_convert(None)
    keys = ['SID', 'ISO_TIME']
    if (table['SID'].eq('').any() or table['ISO_TIME'].isna().any()
            or table.duplicated(keys).any()):
        raise ValueError('S1D exposure CSV point keys must be present and unique')
    if not table['STATUS'].eq('OK').all():
        raise ValueError('S1D exposure CSV must contain successful results for every point; STATUS must be OK')
    numeric = ['LAT', 'LON', 'CAL_DIST', 'POPULATION_YEAR', 'EXPOSURE_PERSON_3H']
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError('S1D exposure CSV coordinates, population years and exposures must be finite')
    table[['LAT', 'LON', 'CAL_DIST']] = table[['LAT', 'LON', 'CAL_DIST']].astype(float)
    if table['EXPOSURE_PERSON_3H'].lt(0).any():
        raise ValueError('S1D population exposure must be nonnegative; zero-exposure points must be retained')
    table['EXPOSURE_PERSON_3H'] = table['EXPOSURE_PERSON_3H'].astype(float)
    population_year = table['POPULATION_YEAR'].to_numpy(float)
    if ((population_year != np.floor(population_year)).any()
            or ((population_year < 1) | (population_year > 9999)).any()):
        raise ValueError('POPULATION_YEAR must contain valid integer years')
    table['POPULATION_YEAR'] = table['POPULATION_YEAR'].astype(int)

    expected = select_panel_d_points(frame)
    wanted_keys = pd.MultiIndex.from_frame(expected[keys])
    actual_keys = pd.MultiIndex.from_frame(table[keys])
    missing, extra = wanted_keys.difference(actual_keys), actual_keys.difference(wanted_keys)
    if len(missing) or len(extra):
        raise ValueError(f'S1D exposure CSV has incomplete or unexpected point coverage: '
                         f'missing={len(missing)} {list(missing[:3])}; extra={len(extra)} {list(extra[:3])}')
    table = table.set_index(keys).reindex(wanted_keys).reset_index()
    for column in ('LAT', 'LON', 'CAL_DIST'):
        if not np.array_equal(table[column].to_numpy(float), expected[column].to_numpy(float)):
            raise ValueError(f'S1D exposure CSV {column} differs from the original track points')
    if not np.array_equal(table['DISTANCE_BIN'].to_numpy(), expected['DISTANCE_BIN'].to_numpy()):
        raise ValueError('S1D exposure CSV distance bins differ from the original boundary rules')
    # The available raster years are recorded through the selected years. A
    # complete export must have a consistent nearest-year choice (earlier on ties).
    available = sorted(table['POPULATION_YEAR'].unique())
    for year, rows in table.groupby(table['ISO_TIME'].dt.year, sort=False):
        mapped = nearest_population_year(year, available)
        if not rows['POPULATION_YEAR'].eq(mapped).all():
            raise ValueError(f'S1D exposure CSV has inconsistent nearest population years for {year}')
    return table.loc[:, list(EXPOSURE_CSV_COLUMNS)]


def read_exposure_csv(csv_dir, frame):
    """Read the complete point table without opening precipitation/population data."""
    path = Path(csv_dir) / EXPOSURE_CSV_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing complete S1D point exposure CSV: {path}')
    table = pd.read_csv(path, dtype={'SID': str, 'DISTANCE_BIN': str, 'STATUS': str},
                        keep_default_na=False, float_precision='round_trip')
    return validate_exposure_rows(table, frame)


def export_exposure_csv(output_dir, raw_data_dir, mswep_dir=None,
                        population_dir=None, overwrite=False):
    """Compute and atomically export all S1D points; return the resulting CSV path."""
    raw = Path(raw_data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    target = output / EXPOSURE_CSV_FILENAME
    if target.exists() and not overwrite:
        raise FileExistsError(f'Exposure CSV already exists: {target}; set overwrite=True to replace it')
    rain_root = Path(mswep_dir).expanduser().resolve() if mswep_dir is not None else raw / 'MSWEP'
    pop_root = Path(population_dir).expanduser().resolve() if population_dir is not None else raw / 'LandScan'
    if not rain_root.is_dir():
        raise FileNotFoundError(f'MSWEP directory not found: {rain_root}')
    frame = read_ibtracs(raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv')
    table = calculate_panel_d_points(frame, discover_population_files(pop_root), rain_root)
    table = validate_exposure_rows(table, frame)
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                dir=output, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False, float_format='%.17g', lineterminator='\n')
        saved = pd.read_csv(temporary, dtype={'SID': str, 'DISTANCE_BIN': str, 'STATUS': str},
                            keep_default_na=False, float_precision='round_trip')
        pd.testing.assert_frame_equal(validate_exposure_rows(saved, frame), table, check_exact=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f'Exposure CSV appeared during calculation: {target}')
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Saved {len(table):,} complete S1D point exposures: {target}')
    return target


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("No TC samples available for bootstrap")
    sample_indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values))
    )
    means = values[sample_indices].mean(axis=1)
    return float(values.mean()), np.percentile(means, [2.5, 97.5])


def panel_d_statistics(point_rows):
    """Return TC-equal means and TC-level bootstrap intervals for each bin."""
    values = pd.to_numeric(point_rows['EXPOSURE_PERSON_3H'], errors='raise').to_numpy(float)
    if (not point_rows['STATUS'].eq('OK').all() or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError('S1D statistics require all point exposures to be successful, finite and nonnegative')
    valid = point_rows
    tc = (
        valid.groupby(["SID", "DISTANCE_BIN"], as_index=False)
        ["EXPOSURE_PERSON_3H"]
        .mean()
        .rename(columns={"EXPOSURE_PERSON_3H": "TC_POINT_MEAN"})
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for key, label in EXPOSURE_DISTANCE_BINS:
        values = tc.loc[
            tc["DISTANCE_BIN"].eq(key), "TC_POINT_MEAN"
        ].to_numpy(float)
        mean, ci = bootstrap_mean(values, rng)
        rows.append(
            {
                "distance_key": key,
                "label": label,
                "mean": mean,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "n_tc": len(values),
            }
        )
    return pd.DataFrame(rows)


def combine_panel_data(trends, point_rows):
    """Combine the A-C trends with the original equal-TC D statistics."""
    trends = trends.copy()
    trends['Unit'] = trends['Metric'].map({
        'CLTC count': 'decade^-1',
    })
    population = panel_d_statistics(point_rows)
    exposure = pd.DataFrame([{
        'Panel': 'D', 'X_Label': row.label, 'Metric': 'Mean point exposure',
        'Unit': 'persons per 3-h TC point', 'Mean_point_exposure_persons': row.mean,
        'CI95_low': row.ci_low, 'CI95_high': row.ci_high,
    } for row in population.itertuples(index=False)])
    return validate_plot_data(pd.concat([trends, exposure], ignore_index=True).reindex(columns=PLOT_COLUMNS))


def calculate_from_exposure_csv(raw_data_dir, csv_dir):
    """Return the 12 plotting rows from tracks and a complete S1D point CSV."""
    raw = Path(raw_data_dir).expanduser().resolve()
    trends = calculate_trend_statistics(raw)
    frame = read_ibtracs(raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv')
    return combine_panel_data(trends, read_exposure_csv(csv_dir, frame))


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=PROJECT_DIR / 'Raw_data')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'Database/CLTC')
    parser.add_argument('--mswep-dir', type=Path)
    parser.add_argument('--population-dir', type=Path)
    parser.add_argument('--exposure-csv-dir', type=Path,
                        help='Complete point CSV directory; default: Raw_data/EXPOSURE_CSV.')
    parser.add_argument('--use-netcdf', action='store_true',
                        help='Force the original NetCDF calculation even when a point CSV exists.')
    args = parser.parse_args(argv)
    if args.exposure_csv_dir is not None and args.use_netcdf:
        parser.error('--exposure-csv-dir and --use-netcdf are mutually exclusive')
    raw, output = project_path(args.raw_data_dir), project_path(args.output_dir)
    csv_dir = project_path(args.exposure_csv_dir) if args.exposure_csv_dir else raw / 'EXPOSURE_CSV'
    use_csv = not args.use_netcdf and (
        args.exposure_csv_dir is not None or (csv_dir / EXPOSURE_CSV_FILENAME).is_file()
    )
    if use_csv:
        print(f'Using complete point exposure CSV: {csv_dir / EXPOSURE_CSV_FILENAME}')
        table = calculate_from_exposure_csv(raw, csv_dir)
    else:
        rain_root = project_path(args.mswep_dir) if args.mswep_dir else raw / 'MSWEP'
        pop_root = project_path(args.population_dir) if args.population_dir else raw / 'LandScan'
        if not rain_root.is_dir():
            raise FileNotFoundError(f'MSWEP directory not found: {rain_root}; use --mswep-dir '
                                    f'or provide a complete {csv_dir / EXPOSURE_CSV_FILENAME}.')
        population_files = discover_population_files(pop_root)
        trends = calculate_trend_statistics(raw)
        frame = read_ibtracs(raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv')
        points = validate_exposure_rows(calculate_panel_d_points(frame, population_files, rain_root), frame)
        table = combine_panel_data(trends, points)
    destination = output / 'FigS1_plot_data.csv'
    write_plot_data(destination, table)
    print(f'Saved {len(table)} rows: {destination}')


if __name__ == '__main__':
    main()
