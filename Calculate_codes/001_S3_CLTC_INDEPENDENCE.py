#!/usr/bin/env python3

import argparse

from pathlib import Path

import tempfile

import numpy as np

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_FILE = PROJECT_DIR / "Database/CLTC/FigS3_plot_data.csv"

PLOT_COLUMNS = [
    "Panel", "X_Label", "Metric", "Trend_per_Decade", "CI95_per_Decade",
    "MK_P", "Unit",
]

EXPECTED_ROWS = [
    ("A", "Coastal TC", "Coast-tracking index"),
    ("A", "CLTC", "Coast-tracking index"),
    ("B", "Coastal TC", "Translation speed"),
    ("B", "CLTC", "Translation speed"),
    ("B", "Steering flow", "Steering flow speed"),
]

EXPECTED_UNITS = ["decade^-1"] * 2 + ["m s^-1 decade^-1"] * 3

NUMERIC_COLUMNS = ["Trend_per_Decade", "CI95_per_Decade", "MK_P"]

def resolve_project_path(value):
    """Resolve relative CLI paths from the directory containing Codes/Database."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def validate_plot_data(table):
    """Validate the five ordered S3 results, including actual numeric MK P values."""
    if list(table.columns) != PLOT_COLUMNS:
        raise ValueError(f"Expected plotting columns in order: {PLOT_COLUMNS}")
    keys = list(table[["Panel", "X_Label", "Metric"]].itertuples(index=False, name=None))
    if keys != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly these five rows in order: {EXPECTED_ROWS}")
    if table["Unit"].tolist() != EXPECTED_UNITS:
        raise ValueError(f"Expected row units: {EXPECTED_UNITS}")
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(table[column]):
            raise ValueError(f"{column} must contain numeric values")
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing or nonfinite values")
    if (table["CI95_per_Decade"] < 0).any():
        raise ValueError("CI95_per_Decade must be a nonnegative confidence-interval half-width")
    if not table["MK_P"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("MK_P must contain Mann-Kendall P values between 0 and 1")

def write_plot_data(table, output_file):
    """Write full-precision values atomically after checking CSV round-trip values."""
    validate_plot_data(table)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv",
        prefix=f".{output_file.stem}_", dir=output_file.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False, float_format="%.17g")
    try:
        saved = pd.read_csv(temporary, float_precision="round_trip", keep_default_na=False)
        validate_plot_data(saved)
        pd.testing.assert_frame_equal(table.reset_index(drop=True), saved, check_exact=True)
        temporary.replace(output_file)
    finally:
        temporary.unlink(missing_ok=True)


import pymannkendall as mk
from scipy import stats

YEAR_MIN, YEAR_MAX = 1980, 2024
STEP_HOURS, LINGER_HOURS, MIN_POINTS = 3, 24, 9
DISTANCE_KM, WIND_LIMIT_KT, MIN_STEP_KM = 100.0, 34.0, 0.1
KNOT_TO_MS = 0.514444
LABELS_A = ['Coastal TC', 'CLTC']
def read_tracks(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    cols = ["SID", "ISO_TIME", "LAT", "LON", "CAL_DIST", "USA_WIND", "STORM_SPEED"]
    df = pd.read_csv(path, usecols=cols, keep_default_na=False, low_memory=False)
    df["SID"] = df["SID"].astype(str).str.strip()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for col in cols[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["STORM_SPEED"] < 0, "STORM_SPEED"] = np.nan
    df = df[
        df["SID"].ne("")
        & df["ISO_TIME"].notna()
        & df["ISO_TIME"].dt.year.between(YEAR_MIN, YEAR_MAX)
    ]
    df = (
        df.dropna(subset=["LAT", "LON", "CAL_DIST"])
        .drop_duplicates(["SID", "ISO_TIME"])
        .sort_values(["SID", "ISO_TIME"])
    )
    sid_lmi = df.assign(_WIND=df["USA_WIND"].fillna(0.0)).groupby("SID")["_WIND"].max()
    df = df[df["SID"].isin(sid_lmi[sid_lmi >= WIND_LIMIT_KT].index)]
    df = (
        df[(df["LAT"] > 20.0) & (df["CAL_DIST"].abs() < DISTANCE_KM)]
        .sort_values(["SID", "ISO_TIME"])
        .reset_index(drop=True)
    )
    print(f"Input: {path}")
    print(f"NH (>20°N) coastal SIDs: {df['SID'].nunique():,}; records: {len(df):,}")
    return df


def identify_cltc_points(coastal):
    df = coastal.copy()
    df["_BLOCK"] = (
        df.groupby("SID")["ISO_TIME"]
        .diff()
        .ne(pd.Timedelta(hours=STEP_HOURS))
        .groupby(df["SID"])
        .cumsum()
        .astype(int)
    )
    events = []
    for (sid, block_id), block in df.groupby(["SID", "_BLOCK"], sort=False):
        block = block.sort_values("ISO_TIME").copy()
        duration = (block["ISO_TIME"].iloc[-1] - block["ISO_TIME"].iloc[0]).total_seconds() / 3600.0
        if len(block) >= MIN_POINTS and duration >= LINGER_HOURS:
            block["Event_ID"] = f"{sid}_E{int(block_id):03d}"
            block["Event_Start"] = block["ISO_TIME"].iloc[0]
            events.append(block)
    if not events:
        raise RuntimeError("No NH (>20°N) CLTC events found.")
    return pd.concat(events, ignore_index=True)


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlon = np.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def exact_pairs(df, keys):
    cols = keys + ["ISO_TIME", "LAT", "LON", "CAL_DIST"]
    left = df[cols].rename(columns={
        "ISO_TIME": "TIME_1", "LAT": "LAT_1", "LON": "LON_1", "CAL_DIST": "DIST_1"
    })
    right = df[cols].rename(columns={
        "ISO_TIME": "TIME_2", "LAT": "LAT_2", "LON": "LON_2", "CAL_DIST": "DIST_2"
    })
    left["TARGET_TIME"] = left["TIME_1"] + pd.Timedelta(hours=STEP_HOURS)
    pairs = left.merge(
        right,
        left_on=keys + ["TARGET_TIME"],
        right_on=keys + ["TIME_2"],
        how="inner",
        validate="one_to_one",
    )
    if pairs.empty:
        return pairs
    step = haversine_km(pairs["LAT_1"], pairs["LON_1"], pairs["LAT_2"], pairs["LON_2"])
    cross = (pairs["DIST_2"] - pairs["DIST_1"]).abs() / step.where(step > MIN_STEP_KM)
    pairs["COAST_TRACKING_INDEX"] = 1.0 - cross.clip(0.0, 1.0)
    return pairs.dropna(subset=["COAST_TRACKING_INDEX"])


def coastal_tc_tracking_annual(coastal):
    pairs = exact_pairs(coastal, ["SID"])
    sid = pairs.groupby("SID", as_index=False).agg(
        FIRST_TIME=("TIME_1", "min"), VALUE=("COAST_TRACKING_INDEX", "mean")
    )
    sid["YEAR"] = sid["FIRST_TIME"].dt.year
    return (
        sid.groupby("YEAR")["VALUE"]
        .mean()
        .reindex(range(YEAR_MIN, YEAR_MAX + 1))
        .rename("COASTAL_TC")
        .reset_index()
    )


def cltc_tracking_annual(cltc):
    pairs = exact_pairs(cltc, ["SID", "Event_ID"])
    event = pairs.groupby(["SID", "Event_ID"], as_index=False).agg(
        EVENT_START=("TIME_1", "min"), VALUE=("COAST_TRACKING_INDEX", "mean")
    )
    storm = event.groupby("SID", as_index=False).agg(
        FIRST_EVENT=("EVENT_START", "min"), VALUE=("VALUE", "mean")
    )
    storm["YEAR"] = storm["FIRST_EVENT"].dt.year
    return (
        storm.groupby("YEAR")["VALUE"]
        .mean()
        .reindex(range(YEAR_MIN, YEAR_MAX + 1))
        .rename("CLTC")
        .reset_index()
    )


def coastal_tc_speed_annual(coastal):
    df = coastal.dropna(subset=["STORM_SPEED"]).copy()
    df["VALUE"] = df["STORM_SPEED"] * KNOT_TO_MS
    storm = df.groupby("SID", as_index=False).agg(
        FIRST_TIME=("ISO_TIME", "min"), VALUE=("VALUE", "mean")
    )
    storm["YEAR"] = storm["FIRST_TIME"].dt.year
    return (
        storm.groupby("YEAR")["VALUE"]
        .mean()
        .reindex(range(YEAR_MIN, YEAR_MAX + 1))
        .rename("COASTAL_TC")
        .reset_index()
    )


def cltc_speed_annual(cltc):
    """
    Three averaging steps:
    trajectory points -> CLTC event mean -> TC mean -> annual mean.
    """
    df = cltc.dropna(subset=["STORM_SPEED"]).copy()
    df["VALUE"] = df["STORM_SPEED"] * KNOT_TO_MS
    # 1) trajectory points -> CLTC event mean
    event = df.groupby(["SID", "Event_ID"], as_index=False).agg(
        EVENT_START=("ISO_TIME", "min"), VALUE=("VALUE", "mean")
    )
    # 2) CLTC event mean -> TC mean
    storm = event.groupby("SID", as_index=False).agg(
        FIRST_EVENT=("EVENT_START", "min"), VALUE=("VALUE", "mean")
    )
    storm["YEAR"] = storm["FIRST_EVENT"].dt.year
    # 3) TC mean -> annual mean (equal TC weight)
    return (
        storm.groupby("YEAR")["VALUE"]
        .mean()
        .reindex(range(YEAR_MIN, YEAR_MAX + 1))
        .rename("CLTC")
        .reset_index()
    )


def steering_flow_annual(path):
    """
    Three averaging steps:
    trajectory points -> CLTC event mean -> TC mean -> annual mean.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(
        path,
        usecols=["SID", "Event_ID", "ISO_TIME", "STEER_SPEED_m_s"],
        low_memory=False,
    )
    df["SID"] = df["SID"].astype(str).str.strip()
    df["Event_ID"] = df["Event_ID"].astype(str).str.strip()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["STEER_SPEED_m_s"] = pd.to_numeric(df["STEER_SPEED_m_s"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "STEER_SPEED_m_s"])
    df = df[df["SID"].ne("") & df["Event_ID"].ne("")].copy()
    # 1) trajectory points -> CLTC event mean
    event = df.groupby(["SID", "Event_ID"], as_index=False).agg(
        EVENT_START=("ISO_TIME", "min"),
        VALUE=("STEER_SPEED_m_s", "mean"),
        N_POINTS=("STEER_SPEED_m_s", "count"),
    )
    # 2) CLTC event mean -> TC mean
    storm = event.groupby("SID", as_index=False).agg(
        FIRST_EVENT=("EVENT_START", "min"),
        VALUE=("VALUE", "mean"),
        N_EVENTS=("Event_ID", "count"),
    )
    storm["YEAR"] = storm["FIRST_EVENT"].dt.year
    storm = storm[storm["YEAR"].between(YEAR_MIN, YEAR_MAX)].copy()
    # 3) TC mean -> annual mean (equal TC weight)
    annual = storm.groupby("YEAR", as_index=False).agg(
        STEERING_FLOW=("VALUE", "mean"), N_TCS=("SID", "count")
    )
    print(f"Steering-flow input: {path}")
    print(
        f"Valid steering-flow events: {len(event):,}; "
        f"TCs: {len(storm):,}; years: {len(annual):,}"
    )
    return annual[["YEAR", "STEERING_FLOW"]]


def trend_statistics(years, values):
    x, y = np.asarray(years, float), np.asarray(values, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(y) < 3:
        raise RuntimeError("At least three valid annual values are required.")
    if np.unique(y).size == 1:
        return {"Trend": 0.0, "CI95": 0.0, "MK_P": 1.0}
    ols = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(y) - 2)
    return {
        "Trend": float(ols.slope * 10),
        "CI95": float(tcrit * ols.stderr * 10),
        "MK_P": float(mk.original_test(y).p),
    }


def significance_mark(p):
    return "**" if p < 0.05 else "*" if p < 0.10 else ""


def make_summary(results_a, translation_results, steering_result):
    rows = []
    for label, result in zip(LABELS_A, results_a):
        rows.append({
            "Panel": "A",
            "X_Label": label,
            "Metric": "Coast-tracking index",
            "Trend_per_Decade": result["Trend"],
            "CI95_per_Decade": result["CI95"],
            "Significance": significance_mark(result["MK_P"]),
            "MK_P": result["MK_P"],
            "Unit": "decade^-1",
        })
    for label, result in zip(["Coastal TC", "CLTC"], translation_results):
        rows.append({
            "Panel": "B",
            "X_Label": label,
            "Metric": "Translation speed",
            "Trend_per_Decade": result["Trend"],
            "CI95_per_Decade": result["CI95"],
            "Significance": significance_mark(result["MK_P"]),
            "MK_P": result["MK_P"],
            "Unit": "m s^-1 decade^-1",
        })
    rows.append({
        "Panel": "B",
        "X_Label": "Steering flow",
        "Metric": "Steering flow speed",
        "Trend_per_Decade": steering_result["Trend"],
        "CI95_per_Decade": steering_result["CI95"],
        "Significance": significance_mark(steering_result["MK_P"]),
        "MK_P": steering_result["MK_P"],
        "Unit": "m s^-1 decade^-1",
    })
    return pd.DataFrame(rows)
def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=PROJECT_DIR / 'Raw_data')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'Database/CLTC')
    parser.add_argument('--steering-file', type=Path)
    args = parser.parse_args(argv)
    raw, output = project_path(args.raw_data_dir), project_path(args.output_dir)
    steering_path = project_path(args.steering_file) if args.steering_file else raw / 'STEER_FLOW/STEERING_FLOW_LMI_24h_100km_34kts_NH20.csv'
    coastal = read_tracks(raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv')
    cltc = identify_cltc_points(coastal)
    tracking = coastal_tc_tracking_annual(coastal).merge(cltc_tracking_annual(cltc), on='YEAR', how='outer')
    speed = coastal_tc_speed_annual(coastal).merge(cltc_speed_annual(cltc), on='YEAR', how='outer')
    steering = steering_flow_annual(steering_path)
    tracking_results = [trend_statistics(tracking['YEAR'], tracking[column]) for column in ('COASTAL_TC', 'CLTC')]
    speed_results = [trend_statistics(speed['YEAR'], speed[column]) for column in ('COASTAL_TC', 'CLTC')]
    steering_result = trend_statistics(steering['YEAR'], steering['STEERING_FLOW'])
    table = make_summary(tracking_results, speed_results, steering_result).loc[:, PLOT_COLUMNS]
    validate_plot_data(table)
    destination = output / 'FigS3_plot_data.csv'
    write_plot_data(table, destination)
    print(f'Saved {len(table)} rows: {destination}')


if __name__ == '__main__':
    main()
