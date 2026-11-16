#!/usr/bin/env python3

import argparse

import math

from pathlib import Path

import tempfile

from matplotlib.path import Path as MplPath

import numpy as np

import pandas as pd

import pymannkendall as mk

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_DIR/'Database/CLTC'

TRACK_COLUMNS = ["SID", "LON", "LAT", "USA_WIND", "CAL_DIST", "ISO_TIME"]

YEAR_MIN, YEAR_MAX = 1980, 2024

WIND_LIMIT_KT = 34.0

DISTANCE_KM = 100.0

LINGER_HOURS = 24

TIME_STEP_HOURS = 3

MIN_POINTS = math.ceil(LINGER_HOURS / TIME_STEP_HOURS) + 1

LINGER_STEPS = MIN_POINTS

PATH_WNA = MplPath([
    (-105, 30), (-98, 18), (-92, 16.6), (-90, 15), (-85, 13),
    (-83, 9), (-81, 8), (-79.3, 9.3), (-78, 8.8), (-76, 6),
    (-48, 6), (-48, 60), (-105, 60),
])

PATH_EP = MplPath([
    (-105.0, 30.0), (-98.0, 18.0), (-92.0, 16.6), (-90.0, 15.0),
    (-85.0, 13.0), (-83.0, 9.0), (-81.0, 8.0), (-79.3, 9.3),
    (-78.0, 8.8), (-76.0, 6.0), (-130.0, 6.0), (-130.0, 37.0),
    (-105.0, 37.0),
])

PATH_WNP = MplPath([
    (102.0, 15.0), (110.0, 8.0), (113.0, 5.0), (148.0, 5.0),
    (148.0, 55.0), (102.0, 55.0), (102.0, 15.0),
])

MAIN_REGIONS = ('Global', 'NH20')
ALL_REGIONS = MAIN_REGIONS
SERIES_COLUMNS = ('Region', 'Year', 'Count', 'Fitted', 'CI95_half_width')
STAT_COLUMNS = ('Region', 'Trend_events_decade', 'CI95_events_decade', 'MK_P')
DURATION_METRICS = ('CLTC duration', 'Coastal-TC duration', 'Duration fraction')
DURATION_UNITS = ('h decade^-1', 'h decade^-1', 'percentage points decade^-1')
DURATION_COLUMNS = ('Metric', 'Trend_per_Decade', 'CI95_per_Decade', 'MK_P', 'Trend_Unit')


def validate_plot_tables(series, statistics):
    """Validate stored curves and genuine numerical Mann-Kendall P values."""
    if series.columns.duplicated().any() or set(series.columns) != set(SERIES_COLUMNS):
        raise ValueError(f'M1 time series require exactly {SERIES_COLUMNS}')
    if statistics.columns.duplicated().any() or set(statistics.columns) != set(STAT_COLUMNS):
        raise ValueError(f'M1 statistics require exactly {STAT_COLUMNS}')
    series = series[list(SERIES_COLUMNS)].copy()
    statistics = statistics[list(STAT_COLUMNS)].copy()
    for column in SERIES_COLUMNS[1:]:
        series[column] = pd.to_numeric(series[column], errors='raise')
    for column in STAT_COLUMNS[1:]:
        statistics[column] = pd.to_numeric(statistics[column], errors='raise')
    if not np.isfinite(series[list(SERIES_COLUMNS[1:])].to_numpy(float)).all():
        raise ValueError('M1 time series contain nonfinite values')
    if not np.isfinite(statistics[list(STAT_COLUMNS[1:])].to_numpy(float)).all():
        raise ValueError('M1 statistics contain nonfinite values')
    expected = {(region, year) for region in MAIN_REGIONS for year in range(1980, 2025)}
    keys = ['Region', 'Year']
    if len(series) != 90 or series.duplicated(keys).any() or set(series[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('M1 requires one record per year, 1980-2024, for each of Global/NH20')
    if len(statistics) != 2 or statistics.Region.duplicated().any() or set(statistics.Region) != set(ALL_REGIONS):
        raise ValueError('M1 requires two regional statistics for Global/NH20')
    if (series.Count.lt(0).any() or not np.equal(series.Count, np.floor(series.Count)).all()
            or series.CI95_half_width.lt(0).any() or statistics.CI95_events_decade.lt(0).any()):
        raise ValueError('Counts must be nonnegative integers and confidence half-widths nonnegative')
    if not statistics.MK_P.between(0, 1).all():
        raise ValueError('Mann-Kendall P values must lie within [0, 1]')
    ordered = pd.MultiIndex.from_product([MAIN_REGIONS, range(1980, 2025)], names=keys)
    series = series.set_index(keys).reindex(ordered).reset_index()
    statistics = statistics.set_index('Region').reindex(ALL_REGIONS).reset_index()
    return series, statistics

def validate_duration_statistics(table):
    """Require the three NH >20N duration trends and numerical MK P values."""
    if table.columns.duplicated().any() or set(table.columns) != set(DURATION_COLUMNS):
        raise ValueError(f'M1 duration statistics require exactly {DURATION_COLUMNS}')
    table = table[list(DURATION_COLUMNS)].copy()
    if len(table) != 3 or table.Metric.duplicated().any() or set(table.Metric) != set(DURATION_METRICS):
        raise ValueError('M1 requires three NH >20N duration statistics')
    numeric = ['Trend_per_Decade', 'CI95_per_Decade', 'MK_P']
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not np.isfinite(table[numeric].to_numpy(float)).all():
        raise ValueError('M1 duration statistics contain nonfinite values')
    if table.CI95_per_Decade.lt(0).any() or not table.MK_P.between(0, 1).all():
        raise ValueError('Duration confidence half-widths must be nonnegative and MK P values within [0, 1]')
    table = table.set_index('Metric').reindex(DURATION_METRICS).reset_index()
    if tuple(table.Trend_Unit) != DURATION_UNITS:
        raise ValueError('M1 duration units must be hours and percentage points per decade')
    return table


def get_point_region(lon, lat):
    point = (lon, lat)
    if PATH_WNA.contains_point(point):
        return "WNA"
    if PATH_EP.contains_point(point):
        return "EP"
    if PATH_WNP.contains_point(point):
        return "WNP"
    return "OTHER"

def load_tracks(input_file):
    """Read the six required fields and apply the whole-SID LMI criterion."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Missing input: {input_file}")
    data = pd.read_csv(
        input_file, usecols=TRACK_COLUMNS, low_memory=False, keep_default_na=False
    )
    data["SID"] = data["SID"].astype(str).str.strip()
    data = data[data["SID"].ne("")].copy()
    for column in ["LON", "LAT", "USA_WIND", "CAL_DIST"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], errors="coerce")

    # Missing winds do not qualify a storm, but remain in a qualifying track.
    # Calculate LMI before filtering years or individual track locations.
    sid_max_wind = (
        data.assign(_WIND_FOR_FILTER=data["USA_WIND"].fillna(0.0))
        .groupby("SID")["_WIND_FOR_FILTER"].max()
    )
    eligible_sids = sid_max_wind[sid_max_wind >= WIND_LIMIT_KT].index
    data = data[data["SID"].isin(eligible_sids)].copy()
    data = data[
        data["ISO_TIME"].notna()
        & data["ISO_TIME"].dt.year.between(YEAR_MIN, YEAR_MAX, inclusive="both")
    ].copy()
    data = data.dropna(subset=["SID", "LON", "LAT", "CAL_DIST", "ISO_TIME"]).copy()
    data["point_region"] = [
        get_point_region(lon, lat) for lon, lat in zip(data["LON"], data["LAT"])
    ]
    return data

def region_subset(data, region):
    """Apply point-level regional cuts before identifying continuous events."""
    if region == "Global":
        return data
    if region == "NH":
        return data[data["LAT"] > 0]
    if region == "NH20":
        return data[data["LAT"] > 20]
    if region.endswith("20"):
        basin = region[:-2]
        return data[(data["point_region"] == basin) & (data["LAT"] > 20)]
    raise ValueError(f"Unknown region: {region}")

def annual_lmi_counts(data, region):
    """Count continuous >=24-h coastal blocks in LMI-eligible tracks."""
    selected = region_subset(data, region)
    selected = selected[selected["CAL_DIST"].abs() < DISTANCE_KM]
    event_years = []
    for _, group in selected.groupby("SID", sort=False):
        ordered = group.sort_values("ISO_TIME")
        block_ids = ordered["ISO_TIME"].diff().ne(
            pd.Timedelta(hours=TIME_STEP_HOURS)
        ).cumsum()
        for _, block in ordered.groupby(block_ids, sort=False):
            if len(block) >= MIN_POINTS:
                event_years.append(int(block["ISO_TIME"].iloc[0].year))
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    counts = (
        pd.Series(event_years, dtype="int64").value_counts()
        .reindex(years, fill_value=0).sort_index().to_numpy(dtype=float)
    )
    return years, counts, len(event_years)

def mann_kendall_p(values):
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

def trend_with_ci(years, values):
    """OLS trend and confidence band; significance from Mann-Kendall."""
    result = stats.linregress(years, values)
    fitted = result.intercept + result.slope * years
    residual_se = np.sqrt(np.sum((values - fitted) ** 2) / (len(years) - 2))
    sxx = np.sum((years - years.mean()) ** 2)
    tcrit = stats.t.ppf(0.975, len(years) - 2)
    band = tcrit * residual_se * np.sqrt(
        1 / len(years) + (years - years.mean()) ** 2 / sxx
    )
    slope_ci = tcrit * result.stderr * 10
    return result, fitted, band, slope_ci, mann_kendall_p(values)

def load_duration_tracks(input_file):
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

def assign_duration_blocks(frame):
    """Assign exact 3-hour continuous blocks within each SID."""
    output = frame.sort_values(["SID", "ISO_TIME"]).copy()
    output["_BLOCK"] = (
        output.groupby("SID", sort=False)["ISO_TIME"]
        .diff()
        .ne(pd.Timedelta(hours=TIME_STEP_HOURS))
        .groupby(output["SID"], sort=False)
        .cumsum()
        .astype(int)
    )
    return output

def identify_duration_cltc_points(coastal):
    """Return points in exact-3-hour coastal blocks lasting at least 24 h."""
    blocked = assign_duration_blocks(coastal)
    accepted = []
    for (sid, block_id), block in blocked.groupby(
        ["SID", "_BLOCK"], sort=False
    ):
        block = block.sort_values("ISO_TIME").copy()
        duration = (
            block["ISO_TIME"].iloc[-1] - block["ISO_TIME"].iloc[0]
        ).total_seconds() / 3600.0
        if len(block) >= LINGER_STEPS and duration >= LINGER_HOURS:
            block["Event_ID"] = f"{sid}_E{int(block_id):03d}"
            accepted.append(block)
    if not accepted:
        return pd.DataFrame(columns=list(blocked.columns) + ["Event_ID"])
    return pd.concat(accepted, ignore_index=True)

def duration_hours_by_year(blocked, keys):
    """Count exact 3-hour intervals by the year of each interval start."""
    hours = {year: 0.0 for year in range(YEAR_MIN, YEAR_MAX + 1)}
    for _, block in blocked.groupby(keys, sort=False):
        block = block.sort_values("ISO_TIME")
        if len(block) < 2:
            continue
        starts = block["ISO_TIME"].iloc[:-1]
        for year, count in starts.dt.year.value_counts().items():
            if YEAR_MIN <= int(year) <= YEAR_MAX:
                hours[int(year)] += TIME_STEP_HOURS * int(count)
    return pd.Series(hours, dtype=float)

def duration_annual(coastal, cltc):
    coastal_hours = duration_hours_by_year(
        assign_duration_blocks(coastal), ["SID", "_BLOCK"]
    )
    if cltc.empty:
        cltc_hours = pd.Series(
            0.0, index=range(YEAR_MIN, YEAR_MAX + 1), dtype=float
        )
    else:
        cltc_hours = duration_hours_by_year(cltc, ["Event_ID"])
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    annual = pd.DataFrame({
        "YEAR": years,
        "CLTC_DURATION_HOURS": cltc_hours.reindex(
            years, fill_value=0.0
        ).to_numpy(),
        "COASTAL_TC_DURATION_HOURS": coastal_hours.reindex(
            years, fill_value=0.0
        ).to_numpy(),
    })
    annual["DURATION_FRACTION"] = annual["CLTC_DURATION_HOURS"].div(
        annual["COASTAL_TC_DURATION_HOURS"].replace(0.0, np.nan)
    )
    if (annual["DURATION_FRACTION"] > 1.000001).any():
        raise RuntimeError("CLTC duration exceeds coastal-TC duration.")
    return annual

def duration_trend_statistics(years, values):
    """Absolute OLS slope/CI and original Mann-Kendall significance."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(y) < 3:
        raise RuntimeError("At least three valid annual values are required.")
    if np.unique(y).size == 1:
        slope_decade = 0.0
        ci95_decade = 0.0
        ols_p = 1.0
        mk_p = 1.0
        mk_z = 0.0
    else:
        ols = stats.linregress(x, y)
        tcrit = stats.t.ppf(0.975, len(y) - 2)
        slope_decade = float(ols.slope * 10.0)
        ci95_decade = float(tcrit * ols.stderr * 10.0)
        ols_p = float(ols.pvalue)
        mk_result = mk.original_test(y)
        mk_p = float(mk_result.p)
        mk_z = float(mk_result.z)
    return {
        "VALID_YEARS": int(len(y)),
        "MEAN": float(np.mean(y)),
        "RAW_TREND_PER_DECADE": slope_decade,
        "RAW_CI95_PER_DECADE": ci95_decade,
        "PLOTTED_TREND_PER_DECADE": slope_decade,
        "PLOTTED_CI95_PER_DECADE": ci95_decade,
        "OLS_P": ols_p,
        "MK_Z": mk_z,
        "MK_P": mk_p,
    }

def calculate_nh20_duration(input_file):
    """Recalculate precisely the former S1B NH >20N three-bar group."""
    frame = load_duration_tracks(input_file)
    coastal = frame.loc[frame["LAT"].gt(20.0) & frame["CAL_DIST"].abs().lt(100.0)].copy()
    annual = duration_annual(coastal, identify_duration_cltc_points(coastal))
    numerator = duration_trend_statistics(annual.YEAR, annual.CLTC_DURATION_HOURS)
    denominator = duration_trend_statistics(annual.YEAR, annual.COASTAL_TC_DURATION_HOURS)
    fraction = duration_trend_statistics(annual.YEAR, annual.DURATION_FRACTION)
    fraction["PLOTTED_TREND_PER_DECADE"] *= 100.0
    fraction["PLOTTED_CI95_PER_DECADE"] *= 100.0
    return annual, (numerator, denominator, fraction)

def prepare_plot_data(input_file):
    tracks = load_tracks(input_file)
    curve_rows, statistic_rows = [], []
    for region in ALL_REGIONS:
        years, counts, _ = annual_lmi_counts(tracks, region)
        regression, fitted, band, ci, p = trend_with_ci(years, counts)
        statistic_rows.append({'Region': region, 'Trend_events_decade': float(regression.slope*10),
                               'CI95_events_decade': float(ci), 'MK_P': float(p)})
        if region in MAIN_REGIONS:
            curve_rows.extend({'Region': region, 'Year': int(year), 'Count': int(count),
                               'Fitted': float(fit), 'CI95_half_width': float(half_width)}
                              for year, count, fit, half_width in zip(years, counts, fitted, band))
    curves, statistics = validate_plot_tables(pd.DataFrame(curve_rows), pd.DataFrame(statistic_rows))
    _, duration_results = calculate_nh20_duration(input_file)
    duration_rows = [
        {'Metric': metric, 'Trend_per_Decade': result['PLOTTED_TREND_PER_DECADE'],
         'CI95_per_Decade': result['PLOTTED_CI95_PER_DECADE'], 'MK_P': result['MK_P'],
         'Trend_Unit': unit}
        for metric, unit, result in zip(DURATION_METRICS, DURATION_UNITS, duration_results)
    ]
    return curves, statistics, validate_duration_statistics(pd.DataFrame(duration_rows))

def write_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False, float_format='%.17g', lineterminator='\n')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f'Saved {len(table)} rows: {path}')

def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=PROJECT_DIR / 'Raw_data')
    parser.add_argument('--input', type=Path, help='Optional override for the track CSV.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    raw = project_path(args.raw_data_dir)
    source = project_path(args.input) if args.input else raw / 'IBTrACS/PRE_DATA_IBT_1980_2024_CAL_DIST_1400km2.csv'
    output = project_path(args.output_dir)
    curves, statistics, duration = prepare_plot_data(source)
    write_table(curves, output / 'FigM1_annual_curves.csv')
    write_table(statistics, output / 'FigM1_trend_statistics.csv')
    write_table(duration, output / 'FigM1_duration_statistics.csv')

if __name__ == '__main__':
    main()
