#!/usr/bin/env python3
"""Run selected calculation or plotting scripts, stopping on the first error."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
FIGURES = {
    'M1': ('001_M1_CAL_CLTC', 'CLTC', None),
    'M2': ('002_M2_SST', 'SST', None),
    'M3': ('003_M3_WRF', 'WRF', None),
    'M4': ('004_M4_RAIN_POP_EXPOSURE', 'EXPOSURE', None),
    'S1': ('001_S1_CLTC_REGION_TRENDS', 'CLTC', 'FigS1_plot_data.csv'),
    'S2': ('001_S2_CLTC_SENSITIVITY', 'CLTC', None),
    'S3': ('001_S3_CLTC_INDEPENDENCE', 'CLTC', 'FigS3_plot_data.csv'),
    'S4': ('002_S4_RH', 'RH', None),
    'S5': ('002_S5_VWS', 'VWS', None),
    'S6': ('002_S6_VWS_CMIP6_MERRA2', 'VWS_ATTRIBUTION', None),
    'S7': ('003_S7_WRF_Domains', 'WRF', None),
    'S8': ('003_S8_WRF_track', 'WRF', 'FigS8_track_coordinates.csv'),
    'S9': ('003_S9_WRF_PVT_Diagnostics', 'WRF', None),
    'S10': ('004_S10_GISA_EXPOSURE', 'URBAN', None),
}
COAST_FIGURES = {'M2', 'M4', 'S4', 'S5', 'S6'}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def copy_coastline(raw, database):
    """Carry the complete shapefile to the plotting Database, without calculation."""
    for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        name = 'GSHHS_f_L1_1400' + suffix
        source, target = raw / 'GSHHG' / name, database / 'GSHHG' / name
        if not source.is_file():
            if suffix == '.cpg':
                continue
            raise FileNotFoundError(f'Missing coastline component: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calculate', 'plot'))
    parser.add_argument('--figures', nargs='+', choices=tuple(FIGURES), default=list(FIGURES))
    parser.add_argument('--raw-data-dir', type=project_path, default=PROJECT / 'Raw_data')
    parser.add_argument('--database-dir', type=project_path, default=PROJECT / 'Database')
    parser.add_argument('--results-dir', type=project_path, default=PROJECT / 'Results')
    parser.add_argument('--mswep-dir', type=project_path)
    parser.add_argument('--population-dir', type=project_path)
    parser.add_argument('--exposure-csv-dir', type=project_path,
                        help='S1/M4 per-time exposure CSVs; used for calculation or direct plotting.')
    parser.add_argument('--use-database', action='store_true',
                        help='Plot S1/M4 from Database even if per-time CSVs are available.')
    parser.add_argument('--use-netcdf', action='store_true',
                        help='Calculate S1/M4 from original grids even if per-time CSVs are available.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing or copying assets.')
    args = parser.parse_args(argv)
    if args.use_database and args.stage != 'plot':
        parser.error('--use-database is only available for the plot stage.')
    if args.use_netcdf and args.stage != 'calculate':
        parser.error('--use-netcdf is only available for the calculate stage.')
    if args.exposure_csv_dir and (args.use_database or args.use_netcdf):
        parser.error('--exposure-csv-dir cannot be combined with --use-database or --use-netcdf.')
    selected = list(dict.fromkeys(args.figures))
    if args.stage == 'calculate' and COAST_FIGURES.intersection(selected):
        if args.dry_run:
            print(f'Copy GSHHG components from {args.raw_data_dir}/GSHHG to {args.database_dir}/GSHHG')
        else:
            copy_coastline(args.raw_data_dir, args.database_dir)
    for figure in selected:
        stem, subdir, data_file = FIGURES[figure]
        folder = 'Calculate_codes' if args.stage == 'calculate' else 'Plotting_codes'
        command = [sys.executable, '-X', 'utf8', str(PROJECT / folder / (stem + '.py'))]
        if args.stage == 'calculate':
            command += ['--raw-data-dir', str(args.raw_data_dir), '--output-dir', str(args.database_dir / subdir)]
            if args.use_netcdf and figure in ('S1', 'M4'):
                command += ['--use-netcdf']
            if args.mswep_dir and figure in ('S1', 'M4'):
                command += ['--mswep-dir', str(args.mswep_dir)]
            if args.population_dir and figure == 'S1':
                command += ['--population-dir', str(args.population_dir)]
        else:
            result_name = {'M4': '004_M4_TC_RAIN_POP_EXPOSURE', 'S8': '003_S8_WRF_Tracks'}.get(figure, stem)
            command += ['--output-dir', str(args.results_dir / result_name)]
            if figure in ('S1', 'M4'):
                command += ['--raw-data-dir', str(args.raw_data_dir)]
                if args.use_database:
                    command += ['--use-database']
            if data_file:
                command += ['--data-file', str(args.database_dir / subdir / data_file)]
            else:
                command += ['--data-dir', str(args.database_dir / subdir)]
            if figure in COAST_FIGURES:
                command += ['--coastline', str(args.database_dir / 'GSHHG/GSHHS_f_L1_1400.shp')]
        if args.exposure_csv_dir and figure in ('S1', 'M4'):
            command += ['--exposure-csv-dir', str(args.exposure_csv_dir)]
        print(f'[{figure}] {subprocess.list2cmdline(command)}', flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=PROJECT)


if __name__ == '__main__':
    main()
