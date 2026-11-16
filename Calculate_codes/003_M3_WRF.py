#!/usr/bin/env python3
import argparse
from pathlib import Path

from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / "Raw_data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Database/WRF"
TRACK_CSV = DEFAULT_RAW_DATA_DIR / "WRF/WRF_ideal_experience_TC_trajectory.csv"
PVT_ROOT = DEFAULT_RAW_DATA_DIR / "WRF/PVT"

SSTS = ("302K-2K", "302K")
DISPLAY_SST = {"302K": "302 K", "302K-2K": "300 K"}
WINDS = tuple(range(34, 41))
DISTANCES = (-50, 0, 50)
TRACK_HOURS = np.arange(0, 25, 3, dtype=float)
PVT_MEAN_LABEL = "PVT (3–10 km)"
BOX_LABELS = (PVT_MEAN_LABEL, "PVT (1 km)", "HA", "VA", "DH")

PVT_HOURS = TRACK_HOURS[1:-1]
PVT_MEAN_HEIGHTS_KM = tuple(range(3, 11))
PVT_COMPONENTS = (
    ("pvt_cx_ms", "PVT (1 km)"),
    ("ha_cx_ms", "HA"),
    ("va_cx_ms", "VA"),
    ("dh_cx_ms", "DH"),
)

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
             if any((root / sst).is_dir() for sst in SSTS)]
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


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_tracks():
    columns = ["sst", "initial_wind_ms", "distance", "forecast_hour",
               "x_coast_km", "x_from_initial_km"]
    df = pd.read_csv(require_file(TRACK_CSV), usecols=columns, low_memory=False)
    df["sst"] = df["sst"].astype(str).str.strip()
    df["distance"] = (df["distance"].astype(str).str.strip()
                      .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True))
    for col in ("initial_wind_ms", "forecast_hour", "x_coast_km",
                "x_from_initial_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[
        df["sst"].isin(SSTS)
        & df["initial_wind_ms"].isin(WINDS)
        & df["distance"].isin([str(d) for d in DISTANCES] + ["CTRL"])
        & df["forecast_hour"].isin(TRACK_HOURS)
    ].copy()
    keys = ["sst", "initial_wind_ms", "distance", "forecast_hour"]
    if df.duplicated(keys).any():
        raise ValueError("SLP trajectory CSV contains duplicate case/time rows")
    expected = pd.MultiIndex.from_product(
        [SSTS, WINDS, [str(d) for d in DISTANCES] + ["CTRL"], TRACK_HOURS],
        names=keys,
    )
    missing = expected.difference(pd.MultiIndex.from_frame(df[keys]))
    if len(missing):
        raise ValueError(f"Missing SLP trajectory rows, first 5: {list(missing[:5])}")
    if not np.isfinite(df["x_from_initial_km"]).all():
        raise ValueError("Non-finite SLP x_from_initial_km")
    exp = df["distance"].ne("CTRL")
    if not np.isfinite(df.loc[exp, "x_coast_km"]).all():
        raise ValueError("Non-finite EXP x_coast_km")
    return df.set_index(keys).sort_index()


def track_run(tracks, sst, wind, distance):
    return tracks.loc[(sst, wind, str(distance))].reindex(TRACK_HOURS)


def pvt_file(sst, wind, distance, height_km):
    return (PVT_ROOT / sst / str(wind) / str(distance)
            / f"pvt_vectors_{height_km}km.csv")


def read_pvt_case(sst, wind, distance, height_km=1):
    path = require_file(pvt_file(sst, wind, distance, height_km))
    cols = ["time", "height_m", *[c for c, _ in PVT_COMPONENTS]]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    if len(df) != len(PVT_HOURS):
        raise ValueError(f"{path}: expected seven 3–21 h records, found {len(df)}")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
    df = df.sort_values("time").reset_index(drop=True)
    if df["time"].duplicated().any():
        raise ValueError(f"{path}: duplicate times")
    if not df["time"].diff().iloc[1:].eq(pd.Timedelta(hours=3)).all():
        raise ValueError(f"{path}: times are not consecutive 3-h records")
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[cols[1:]].to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite PVT values")
    if not np.allclose(df["height_m"], height_km * 1000.0):
        raise ValueError(f"{path}: data are not at {height_km} km")
    return df


def read_pvt_mean_profile(sst, wind, distance, reference_times):
    """Return an 8-height by 7-time array of zonal PVT velocities."""
    values = []
    for height_km in PVT_MEAN_HEIGHTS_KM:
        table = read_pvt_case(sst, wind, distance, height_km)
        if not table["time"].equals(reference_times):
            raise ValueError(
                f"{sst}/{wind}/{distance}/{height_km}km: "
                "times differ from the 1-km PVT records"
            )
        values.append(table["pvt_cx_ms"].to_numpy(float))
    return np.stack(values)


def paired_case_differences():
    records = []
    for sst in SSTS:
        for wind in WINDS:
            ctrl_pvt = read_pvt_case(sst, wind, "CTRL")
            ctrl_profile = read_pvt_mean_profile(
                sst, wind, "CTRL", ctrl_pvt["time"]
            )
            for distance in DISTANCES:
                exp_pvt = read_pvt_case(sst, wind, distance)
                if not exp_pvt["time"].equals(ctrl_pvt["time"]):
                    raise ValueError(f"{sst}/{wind}/{distance}: EXP/CTRL PVT times differ")
                exp_profile = read_pvt_mean_profile(
                    sst, wind, distance, exp_pvt["time"]
                )
                row = {
                    "source_sst": sst,
                    "display_sst": DISPLAY_SST[sst],
                    "initial_wind_ms": wind,
                    "distance_km": distance,
                    PVT_MEAN_LABEL: float(np.mean(
                        exp_profile - ctrl_profile
                    )),
                }
                for col, name in PVT_COMPONENTS:
                    row[name] = float(np.mean(
                        exp_pvt[col].to_numpy(float)
                        - ctrl_pvt[col].to_numpy(float)
                    ))
                records.append(row)
    result = pd.DataFrame.from_records(records)
    if len(result) != len(SSTS) * len(WINDS) * len(DISTANCES):
        raise ValueError("Incorrect number of paired case differences")
    return result


def mean_track_offsets(tracks):
    """Mean paired EXP-minus-CTRL zonal offsets, by SST and distance."""
    lines = []
    for distance in DISTANCES:
        for sst in SSTS:
            offsets = []
            for wind in WINDS:
                exp = track_run(tracks, sst, wind, distance)
                ctrl = track_run(tracks, sst, wind, "CTRL")
                difference = (
                    exp["x_from_initial_km"].to_numpy(float)
                    - ctrl["x_from_initial_km"].to_numpy(float)
                )
                if not np.isclose(difference[0], 0.0, atol=0.01):
                    raise ValueError(
                        f"{sst}/{wind}/{distance}: offset at 0 h is not zero"
                    )
                offsets.append(difference)
            lines.append((sst, distance, np.mean(np.stack(offsets), axis=0)))
    return lines

def build_plot_tables(offset_lines, paired):
    """Keep only final A coordinates and the box/mean statistics used by B."""
    trajectory_rows = []
    for sst, distance, offsets in offset_lines:
        for hour, x_km in zip(TRACK_HOURS, distance + offsets):
            trajectory_rows.append({
                "source_sst": sst, "initial_distance_km": distance,
                "time_h": int(hour), "plot_x_km": float(x_km),
            })
    box_rows = []
    for sst in SSTS:
        subset = paired.loc[paired.source_sst.eq(sst)]
        if len(subset) != len(WINDS) * len(DISTANCES):
            raise ValueError(f"{sst}: expected 21 paired case means per component")
        for label in BOX_LABELS:
            values = subset[label].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{sst}/{label}: non-finite paired case means")
            stats = boxplot_stats(values, whis=1.5)[0]
            box_rows.append({
                "source_sst": sst, "component": label,
                "whisker_low_ms": stats["whislo"], "q1_ms": stats["q1"],
                "median_ms": stats["med"], "q3_ms": stats["q3"],
                "whisker_high_ms": stats["whishi"], "mean_ms": float(values.mean()),
            })
    if len(trajectory_rows) != 54 or len(box_rows) != 10:
        raise ValueError("Expected 54 trajectory rows and 10 box-statistic rows")
    return pd.DataFrame(trajectory_rows), pd.DataFrame(box_rows)


def save_table(table, path):
    temporary = path.with_suffix(".csv.tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", float_format="%.17g")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved: {path} ({len(table)} rows)")


def main(argv=None):
    global TRACK_CSV, PVT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", "--database-root", dest="raw_data_dir",
                        type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help="Root containing the original WRF calculation inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the two plotting CSV files.")
    parser.add_argument("--pvt-dir", type=Path,
                        help="Optional directory containing the 302K-2K and 302K PVT folders.")
    args = parser.parse_args(argv)
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    TRACK_CSV = raw_data_dir / "WRF/WRF_ideal_experience_TC_trajectory.csv"
    PVT_ROOT = resolve_pvt_root(raw_data_dir, args.pvt_dir)
    print(f"PVT inputs: {PVT_ROOT}", flush=True)
    output_dir = args.output_dir.expanduser().resolve()
    tracks = read_tracks()
    offset_lines = mean_track_offsets(tracks)
    paired = paired_case_differences()
    trajectories, boxes = build_plot_tables(offset_lines, paired)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_table(trajectories, output_dir / "FigM3A_track_coordinates.csv")
    save_table(boxes, output_dir / "FigM3B_motion_boxstats.csv")


if __name__ == "__main__":
    main()
