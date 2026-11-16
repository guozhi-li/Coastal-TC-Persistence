#!/usr/bin/env python3
import argparse

import math

from pathlib import Path

import tempfile

import numpy as np

import pandas as pd

import pymannkendall as mk

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'Database/CLTC'

IB_YEAR_MIN, IB_YEAR_MAX = 1980, 2024

ADT_YEAR_MIN, ADT_YEAR_MAX = 1980, 2017

WIND_LIMIT_KT = 34.0

INDEX_MONTHS = tuple(range(1, 13))

MIN_VALID_INDEX_MONTHS = 10

BAR_COLUMNS = (
    'Panel', 'Condition', 'Method', 'Trend_events_decade',
    'CI95_events_decade', 'MK_P',
)

PDF_COLUMNS = ('Duration_h', 'Density_percent')

SHARE_COLUMNS = ('Threshold_h', 'Below_percent')

BAR_CASES = (
    ('A', '12 h', 'LMI'),
    ('A', '24 h', 'LMI'),
    ('A', '36 h', 'LMI'),
    ('B', '±50 km', 'LMI'),
    ('B', '±100 km', 'LMI'),
    ('B', '±150 km', 'LMI'),
    ('C', '1400', 'LMI'),
    ('C', '2000', 'LMI'),
    ('C', '5000', 'LMI'),
    ('C', '10000', 'LMI'),
    ('D', 'IBTrACS (3 h)', 'LMI'),
    ('D', 'IBTrACS (3 h)', 'MIN'),
    ('D', 'IBTrACS (6 h)', 'LMI'),
    ('D', 'IBTrACS (6 h)', 'MIN'),
    ('D', 'ADT-HURSAT (6 h)', 'LMI'),
    ('D', 'ADT-HURSAT (6 h)', 'MIN'),
    ('E', 'Original', 'LMI'),
    ('E', 'ENSO-removed', 'LMI'),
    ('E', 'AMO-removed', 'LMI'),
)

PDF_POINT_COUNT = 600

PDF_MARKERS = (12, 24, 36)

OUTPUT_FILENAMES = (
    'FigS2_bar_statistics.csv', 'FigS2_duration_pdf.csv',
    'FigS2_duration_shares.csv',
)

def validate_bars(table):
    """Return the 19 original bar estimates in panel/category/method order."""
    if table.columns.duplicated().any() or set(table.columns) != set(BAR_COLUMNS):
        raise ValueError(f'S2 bars require exactly these columns: {BAR_COLUMNS}')
    table = table.loc[:, list(BAR_COLUMNS)].copy()
    keys = ['Panel', 'Condition', 'Method']
    if len(table) != len(BAR_CASES) or table[keys].isna().any().any():
        raise ValueError('S2 requires 19 complete bar records: A=3, B=3, C=4, D=6, E=3')
    if table.duplicated(keys).any():
        raise ValueError('S2 bar panel/condition/method combinations must be unique')
    observed = set(table[keys].itertuples(index=False, name=None))
    if observed != set(BAR_CASES):
        raise ValueError('Unexpected or missing S2 bar panel/condition/method combinations')
    numeric = ['Trend_events_decade', 'CI95_events_decade', 'MK_P']
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError('S2 bar trends, confidence half-widths and MK P values must be finite')
    if table['CI95_events_decade'].lt(0).any():
        raise ValueError('S2 confidence half-widths must be nonnegative')
    if not table['MK_P'].between(0, 1).all():
        raise ValueError('S2 MK P values must lie within [0, 1]')
    order = pd.MultiIndex.from_tuples(BAR_CASES, names=keys)
    return table.set_index(keys).reindex(order).reset_index().loc[:, list(BAR_COLUMNS)]

def validate_pdf(table):
    """Validate and order the saved curve without fitting or renormalizing it."""
    if table.columns.duplicated().any() or set(table.columns) != set(PDF_COLUMNS):
        raise ValueError(f'S2 PDF requires exactly these columns: {PDF_COLUMNS}')
    table = table.loc[:, list(PDF_COLUMNS)].copy()
    if len(table) != PDF_POINT_COUNT:
        raise ValueError(f'S2 PDF requires exactly {PDF_POINT_COUNT} saved curve points')
    for column in PDF_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('S2 PDF durations and densities must be finite')
    table = table.sort_values('Duration_h', kind='stable').reset_index(drop=True)
    x = table['Duration_h'].to_numpy(dtype=float)
    density = table['Density_percent'].to_numpy(dtype=float)
    if x[0] != 0 or not (np.diff(x) > 0).all() or x[-1] < 48:
        raise ValueError('S2 PDF must start at zero, increase strictly and extend to at least 48 h')
    if not np.allclose(np.diff(x), (x[-1] - x[0]) / (len(x) - 1), rtol=1e-10, atol=1e-12):
        raise ValueError('S2 PDF must retain its original evenly spaced grid')
    if (density < 0).any() or not (density > 0).any():
        raise ValueError('S2 PDF density must be nonnegative and not identically zero')
    return table

def validate_shares(table):
    """Validate the empirical percentages; never derive them from the PDF."""
    if table.columns.duplicated().any() or set(table.columns) != set(SHARE_COLUMNS):
        raise ValueError(f'S2 duration shares require exactly these columns: {SHARE_COLUMNS}')
    table = table.loc[:, list(SHARE_COLUMNS)].copy()
    if len(table) != len(PDF_MARKERS):
        raise ValueError('S2 requires the three empirical shares below 12, 24 and 36 h')
    for column in SHARE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('S2 empirical thresholds and percentages must be finite')
    if set(table['Threshold_h']) != set(PDF_MARKERS):
        raise ValueError('S2 empirical thresholds must be exactly 12, 24 and 36 h')
    table = table.sort_values('Threshold_h', kind='stable').reset_index(drop=True)
    shares = table['Below_percent'].to_numpy(dtype=float)
    if not ((shares >= 0) & (shares <= 100)).all() or (np.diff(shares) < 0).any():
        raise ValueError('S2 empirical shares must increase monotonically within [0, 100]')
    return table

def read_tracks(path, wind_column, year_min, year_max):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    required = ["SID", "ISO_TIME", "LAT", wind_column, "CAL_DIST"]
    frame = pd.read_csv(path, usecols=required, keep_default_na=False, low_memory=False)
    for column in ["LAT", wind_column, "CAL_DIST"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    frame["SID"] = frame["SID"].astype(str).str.strip()
    frame = frame[
        frame["SID"].ne("") & frame["ISO_TIME"].notna()
        & frame["ISO_TIME"].dt.year.between(year_min, year_max, inclusive="both")
    ].dropna(subset=["LAT", "CAL_DIST"])
    frame = frame.drop_duplicates(["SID", "ISO_TIME"], keep="first")
    frame = frame.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    sid_lmi = frame.groupby("SID")[wind_column].max()
    eligible_sids = sid_lmi[sid_lmi >= WIND_LIMIT_KT].index
    frame = frame[frame["SID"].isin(eligible_sids)].copy()
    if not (frame["CAL_DIST"].lt(0).any() and frame["CAL_DIST"].gt(0).any()):
        raise ValueError(f"{path.name}: CAL_DIST must contain negative land and positive ocean.")
    print(f"Input: {path}")
    print(f"Eligible SIDs: {len(eligible_sids):,}; retained points: {len(frame):,}")
    return frame

def region_mask(frame, region):
    if region == "GLOBAL":
        return pd.Series(True, index=frame.index)
    if region == "NH":
        return frame["LAT"] > 0.0
    if region == "NH20":
        return frame["LAT"] > 20.0
    raise ValueError(f"Unknown region: {region}")

def continuous_segments(
    frame,
    region,
    distance_km,
    step_hours,
    wind_column,
    method="LMI",
):
    selected = frame[
        region_mask(frame, region)
        & (frame["CAL_DIST"].abs() < float(distance_km))
    ].copy()
    if method == "MIN":
        # For MIN, every point must meet the instantaneous threshold. Missing
        # wind is below threshold and therefore breaks the continuous segment.
        selected = selected[
            selected[wind_column].fillna(0.0) >= WIND_LIMIT_KT
        ].copy()
    elif method != "LMI":
        raise ValueError(f"Unknown intensity method: {method}")
    records = []
    for sid, storm in selected.groupby("SID", sort=False):
        storm = storm.sort_values("ISO_TIME")
        block_ids = (
            storm["ISO_TIME"].diff()
            .ne(pd.Timedelta(hours=step_hours))
            .cumsum()
        )
        for segment_number, (_, block) in enumerate(
            storm.groupby(block_ids, sort=False), start=1
        ):
            block = block.sort_values("ISO_TIME")
            start = block["ISO_TIME"].iloc[0]
            end = block["ISO_TIME"].iloc[-1]
            records.append({
                "SID": sid,
                "Segment_ID": f"{sid}_S{segment_number:03d}",
                "Start_Time": start,
                "End_Time": end,
                "Start_Year": int(start.year),
                "Duration_Hours": (end - start).total_seconds() / 3600.0,
                "Obs_Count": int(len(block)),
                "Region": region,
                "Distance_km": float(distance_km),
                "Method": method,
            })
    return pd.DataFrame(records, columns=[
        "SID", "Segment_ID", "Start_Time", "End_Time", "Start_Year",
        "Duration_Hours", "Obs_Count", "Region", "Distance_km", "Method",
    ])

def qualifying_events(segments, linger_hours, step_hours):
    minimum_points = math.ceil(float(linger_hours) / step_hours) + 1
    return segments[
        (segments["Obs_Count"] >= minimum_points)
        & (segments["Duration_Hours"] >= float(linger_hours))
    ].copy()

def annual_counts(events, year_min, year_max):
    years = np.arange(year_min, year_max + 1, dtype=int)
    if events.empty:
        counts = pd.Series(0, index=years, dtype=int)
    else:
        counts = (
            events.groupby("Start_Year").size()
            .reindex(years, fill_value=0)
            .astype(int)
        )
    return pd.DataFrame({"ISO_YEAR": years, "Count": counts.to_numpy()})

def trend_statistics(annual):
    years = annual["ISO_YEAR"].to_numpy(dtype=float)
    counts = annual["Count"].to_numpy(dtype=float)
    ols = stats.linregress(years, counts)
    tcrit = stats.t.ppf(0.975, len(years) - 2)
    mk_p = 1.0 if np.unique(counts).size <= 1 else float(mk.original_test(counts).p)
    return {
        "Trend_events_decade": float(ols.slope * 10.0),
        "CI95_events_decade": float(tcrit * ols.stderr * 10.0),
        "MK_P": mk_p,
    }

def evaluate_case(
    frame, wind_column, year_min, year_max, step_hours, method,
    region, distance_km, linger_hours, panel, condition,
):
    segments = continuous_segments(frame, region, distance_km, step_hours, wind_column, method)
    events = qualifying_events(segments, linger_hours, step_hours)
    annual = annual_counts(events, year_min, year_max)
    row = {"Panel": panel, "Condition": condition, "Method": method, **trend_statistics(annual)}
    return row, annual

def _annualize_monthly_index(monthly, value_name, year_min, year_max):
    """Convert a monthly climate index to complete Jan-Dec annual means."""
    table = monthly.copy()
    table["Year"] = pd.to_numeric(table["Year"], errors="coerce")
    table["Month"] = pd.to_numeric(table["Month"], errors="coerce")
    table[value_name] = pd.to_numeric(table[value_name], errors="coerce")
    table.loc[table[value_name].abs() > 20.0, value_name] = np.nan
    table = table[
        table["Year"].between(year_min, year_max)
        & table["Month"].isin(INDEX_MONTHS)
    ].copy()
    grouped = table.groupby("Year")[value_name].agg(["mean", "count"])
    required_years = pd.Index(range(year_min, year_max + 1), name="Year")
    grouped = grouped.reindex(required_years)
    bad = grouped.index[
        grouped["mean"].isna() | (grouped["count"] < MIN_VALID_INDEX_MONTHS)
    ].tolist()
    if bad:
        raise ValueError(
            f"{value_name} lacks at least {MIN_VALID_INDEX_MONTHS} valid monthly "
            f"values in years: {bad}"
        )
    series = grouped["mean"].astype(float)
    series.name = value_name
    return series

def read_nino34_annual(path, year_min, year_max):
    """Read Niño-3.4 data in wide or long monthly layout."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = raw_line.replace(",", " ").split()
        if len(tokens) < 3:
            continue
        try:
            year = int(float(tokens[0]))
        except ValueError:
            continue
        if not 1800 <= year <= 2200:
            continue
        if len(tokens) >= 13:
            for month, token in enumerate(tokens[1:13], start=1):
                try:
                    value = float(token)
                except ValueError:
                    value = np.nan
                rows.append((year, month, value))
        else:
            try:
                month = int(float(tokens[1]))
                value = float(tokens[2])
            except ValueError:
                continue
            if 1 <= month <= 12:
                rows.append((year, month, value))
    if not rows:
        raise ValueError(f"No monthly Niño-3.4 values parsed from {path}")
    monthly = pd.DataFrame(rows, columns=["Year", "Month", "ENSO"])
    return _annualize_monthly_index(monthly, "ENSO", year_min, year_max)

def _find_numeric_column(frame, excluded):
    best_column, best_count = None, -1
    for column in frame.columns:
        if column in excluded:
            continue
        count = pd.to_numeric(frame[column], errors="coerce").notna().sum()
        if count > best_count:
            best_column, best_count = column, count
    return best_column

def read_amo_annual(path, year_min, year_max):
    """Read AMO data from date, long, wide, decimal-year, or annual layout."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, comment="#").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"No AMO data in {path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    normalized = {column: column.lower().replace(" ", "_") for column in frame.columns}
    date_column = next(
        (column for column, key in normalized.items() if "date" in key or "time" in key), None
    )
    year_column = next(
        (column for column, key in normalized.items()
         if key in {"year", "yr"} or key.endswith("_year")), None
    )
    month_column = next(
        (column for column, key in normalized.items()
         if key in {"month", "mon", "mn"} or key.endswith("_month")), None
    )
    value_column = next(
        (column for column, key in normalized.items() if "amo" in key or "amv" in key), None
    )
    if date_column is not None:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if value_column is None:
            value_column = _find_numeric_column(frame, {date_column})
        monthly = pd.DataFrame({
            "Year": dates.dt.year, "Month": dates.dt.month,
            "AMO": pd.to_numeric(frame[value_column], errors="coerce"),
        })
        return _annualize_monthly_index(monthly, "AMO", year_min, year_max)
    if year_column is None:
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.between(1800, 2201).mean() >= 0.8:
                year_column = column
                break
    if year_column is None:
        raise ValueError(f"Cannot identify a year column in {path}: {list(frame.columns)}")
    year_values = pd.to_numeric(frame[year_column], errors="coerce")
    numeric_other = [
        column for column in frame.columns if column != year_column
        and pd.to_numeric(frame[column], errors="coerce").notna().sum() > 0
    ]
    if month_column is None and len(numeric_other) >= 12:
        rows = []
        for row_index, year_value in year_values.items():
            if not np.isfinite(year_value):
                continue
            for month, column in enumerate(numeric_other[:12], start=1):
                rows.append((int(year_value), month, frame.at[row_index, column]))
        return _annualize_monthly_index(
            pd.DataFrame(rows, columns=["Year", "Month", "AMO"]),
            "AMO", year_min, year_max,
        )
    if value_column is None or value_column == year_column:
        value_column = _find_numeric_column(frame, {year_column, month_column})
    if value_column is None:
        raise ValueError(f"Cannot identify an AMO value column in {path}")
    amo_values = pd.to_numeric(frame[value_column], errors="coerce")
    if month_column is not None:
        monthly = pd.DataFrame({
            "Year": np.floor(year_values),
            "Month": pd.to_numeric(frame[month_column], errors="coerce"),
            "AMO": amo_values,
        })
        return _annualize_monthly_index(monthly, "AMO", year_min, year_max)
    fractional = np.abs(year_values - np.floor(year_values)) > 1e-6
    if fractional.any() or year_values.duplicated().any():
        year_floor = np.floor(year_values)
        month = np.rint((year_values - year_floor) * 12.0).astype("Int64") + 1
        monthly = pd.DataFrame({
            "Year": year_floor, "Month": month.clip(1, 12), "AMO": amo_values,
        })
        return _annualize_monthly_index(monthly, "AMO", year_min, year_max)
    annual = pd.Series(
        amo_values.to_numpy(), index=year_values.astype("Int64"), name="AMO"
    )
    annual = annual[~annual.index.isna()]
    annual.index = annual.index.astype(int)
    annual = annual.reindex(range(year_min, year_max + 1))
    annual.loc[annual.abs() > 20.0] = np.nan
    if annual.isna().any():
        raise ValueError(f"AMO annual values missing in years: {annual.index[annual.isna()].tolist()}")
    return annual.astype(float)

def residual_removed_count_statistics(annual, climate_index, control_name):
    """Remove count ~ standardized climate index and trend residual + mean."""
    data = annual[["ISO_YEAR", "Count"]].copy()
    data[control_name] = data["ISO_YEAR"].map(climate_index)
    if data[control_name].isna().any():
        missing = data.loc[data[control_name].isna(), "ISO_YEAR"].astype(int).tolist()
        raise ValueError(f"{control_name} missing for CLTC years: {missing}")
    counts = data["Count"].to_numpy(dtype=float)
    index_raw = data[control_name].to_numpy(dtype=float)
    index_sd = float(np.std(index_raw, ddof=1))
    if not np.isfinite(index_sd) or np.isclose(index_sd, 0.0):
        raise ValueError(f"{control_name} has zero variance")
    index_z = (index_raw - np.mean(index_raw)) / index_sd
    fit = stats.linregress(index_z, counts)
    adjusted = counts - (fit.intercept + fit.slope * index_z) + float(np.mean(counts))
    adjusted_annual = pd.DataFrame({"ISO_YEAR": data["ISO_YEAR"].to_numpy(int), "Count": adjusted})
    return trend_statistics(adjusted_annual)

def calculate_bar_statistics(database_root):
    """Reproduce all 19 notebook analyses from the original track/index inputs."""
    database_root = Path(database_root)
    ibtracs_file = database_root / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv'
    adt_file = database_root / 'ADT-HURSAT/PRE_DATA_ADT_HURSAT_1980_2017_CAL_DIST.csv'
    ibtracs_3h = read_tracks(ibtracs_file, 'USA_WIND', IB_YEAR_MIN, IB_YEAR_MAX)
    # Preserve the original LMI eligibility established before 6-hour subsampling.
    ibtracs_6h = ibtracs_3h[
        ibtracs_3h['ISO_TIME'].dt.hour.mod(6).eq(0)
        & ibtracs_3h['ISO_TIME'].dt.minute.eq(0)
    ].copy()
    adt_6h = read_tracks(adt_file, 'ADT_WIND', ADT_YEAR_MIN, ADT_YEAR_MAX)
    land_frames = {'1400': ibtracs_3h}
    datasets = {
        'IBTrACS (3 h)': (ibtracs_3h, 'USA_WIND', IB_YEAR_MIN, IB_YEAR_MAX, 3),
        'IBTrACS (6 h)': (ibtracs_6h, 'USA_WIND', IB_YEAR_MIN, IB_YEAR_MAX, 6),
        'ADT-HURSAT (6 h)': (adt_6h, 'ADT_WIND', ADT_YEAR_MIN, ADT_YEAR_MAX, 6),
    }
    rows = []
    baseline_annual = None
    for panel, condition, method in BAR_CASES:
        if panel == 'E':
            continue
        frame, wind, first, last, step = datasets['IBTrACS (3 h)']
        distance, duration = 100, 24
        if panel == 'A':
            duration = int(condition[:-2])
        elif panel == 'B':
            distance = int(condition[1:-3])
        elif panel == 'C':
            if condition not in land_frames:
                path = database_root / f'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_{condition}km2.csv'
                land_frames[condition] = read_tracks(path, 'USA_WIND', IB_YEAR_MIN, IB_YEAR_MAX)
            frame = land_frames[condition]
        elif panel == 'D':
            frame, wind, first, last, step = datasets[condition]
        result, annual = evaluate_case(
            frame, wind, first, last, step, method,
            'NH20', distance, duration, panel, condition,
        )
        rows.append(result)
        if panel == 'A' and duration == 24:
            baseline_annual = annual[['ISO_YEAR', 'Count']].copy()
    if baseline_annual is None:
        raise RuntimeError('The original 24-hour baseline annual counts were not generated')
    nino34 = read_nino34_annual(database_root / 'ENSO_AMO/nino34.anom.data.txt', IB_YEAR_MIN, IB_YEAR_MAX)
    amo = read_amo_annual(database_root / 'ENSO_AMO/csu_amo.csv', IB_YEAR_MIN, IB_YEAR_MAX)
    climate_results = (
        ('Original', trend_statistics(baseline_annual)),
        ('ENSO-removed', residual_removed_count_statistics(baseline_annual, nino34, 'ENSO')),
        ('AMO-removed', residual_removed_count_statistics(baseline_annual, amo, 'AMO')),
    )
    for condition, result in climate_results:
        rows.append({'Panel': 'E', 'Condition': condition, 'Method': 'LMI', **result})
    return validate_bars(pd.DataFrame(rows))

def write_table(path, table):
    """Write UTF-8 CSV atomically with round-trip floating-point precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False, float_format='%.17g', lineterminator='\n')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


IB_STEP_HOURS = 3

def prepare_pdf(frame):
    """NH (>20°N) residence-time PDF for LMI within ±100 km."""
    segments = continuous_segments(
        frame, region="NH20", distance_km=100, step_hours=IB_STEP_HOURS,
        wind_column="USA_WIND", method="LMI",
    )
    durations = segments.loc[segments["Obs_Count"] >= 2, "Duration_Hours"].to_numpy(float)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if durations.size < 2 or np.unique(durations).size < 2:
        raise RuntimeError("Not enough distinct NH20 coastal durations for a PDF.")
    kde = stats.gaussian_kde(np.concatenate([-durations, durations]))
    xmax = max(48.0, math.ceil(float(np.percentile(durations, 99.5)) / 12.0) * 12.0)
    x = np.linspace(0.0, xmax, 600)
    density = 2.0 * kde(x)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    density /= integrate(density, x)
    return {
        "x": x,
        "density_percent": 100.0 * density,
        "xmax": xmax,
        "below": {threshold: 100.0 * float(np.mean(durations < threshold)) for threshold in PDF_MARKERS},
    }
def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=PROJECT_DIR / 'Raw_data')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    raw, output = project_path(args.raw_data_dir), project_path(args.output_dir)
    bars = calculate_bar_statistics(raw)
    frame = read_tracks(raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv', 'USA_WIND', IB_YEAR_MIN, IB_YEAR_MAX)
    distribution = prepare_pdf(frame)
    pdf = validate_pdf(pd.DataFrame({'Duration_h': distribution['x'], 'Density_percent': distribution['density_percent']}))
    shares = validate_shares(pd.DataFrame({
        'Threshold_h': list(PDF_MARKERS),
        'Below_percent': [distribution['below'][threshold] for threshold in PDF_MARKERS],
    }))
    for name, table in zip(OUTPUT_FILENAMES, (bars, pdf, shares)):
        write_table(output / name, table)
        print(f'Saved {len(table)} rows: {output / name}')


if __name__ == '__main__':
    main()
