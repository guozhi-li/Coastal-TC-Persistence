"""Extract annual area-weighted bin means from the original monthly CMIP6 grids."""
import argparse
import importlib
from pathlib import Path
import tempfile

import dask
import pandas as pd

from cmip6_inputs import read_annual_bins


def source_metadata(open_netcdf, fields):
    """Keep ensemble member, native grid label and units from the input headers."""
    headers = []
    for variable, path in fields.items():
        with open_netcdf(path, decode_times=False) as ds:
            headers.append((ds.attrs['source_variant'], ds.attrs['source_grid'],
                            ds[variable].attrs['units']))
    if len(set(headers)) != 1:
        raise ValueError(f'CMIP6 component variables have different members, grids or units: {fields}')
    return dict(zip(('Variant', 'Grid', 'Units'), headers[0]))


def prepare(raw_root, output_root, variable):
    if variable == 'SST':
        module = importlib.import_module('002_M2_SST')
        module.configure_paths(raw_root)
        inventory = module.validate_cmip6_inventory()
        models = module.CMIP6_MODELS
        experiments = dict(module.CMIP6_EXPERIMENTS)
        edges = module.CMIP6_DISTANCE_EDGES_KM
        def extract(model, experiment):
            frame = module.raw_cmip6_sst_bins(inventory, model, experiment)
            info = source_metadata(module.open_netcdf, {'tos': inventory[model, experiment]})
            return frame.assign(**info)
    else:
        module = importlib.import_module('002_S6_VWS_CMIP6_MERRA2')
        module.configure_paths(raw_root)
        models, experiments = module.MODELS, module.EXPERIMENTS
        edges = module.DISTANCE_EDGES_KM
        def extract(model, experiment):
            frame = module.raw_cmip6_vws_bins(model, experiment)
            files = {name: module.cmip6_path(model, experiment, name) for name in ('ua', 'va')}
            return frame.assign(**source_metadata(module.open_netcdf, files))

    frames = []
    for model in models:
        for experiment in experiments:
            print(f'Extracting {variable}: {model} / {experiment}', flush=True)
            with dask.config.set(scheduler='synchronous'):
                frame = extract(model, experiment)
            frames.append(frame.assign(Model=model, Experiment=experiment))
    table = pd.concat(frames, ignore_index=True)
    path = output_root / 'CMIP6' / f'{variable.lower()}_annual_bins.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.csv.gz', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        table.to_csv(temporary, index=False, float_format='%.17g',
                     compression={'method': 'gzip', 'mtime': 0})
        read_annual_bins(temporary, models, experiments, edges, module.CMIP6_START_YEAR,
                         module.CMIP6_END_YEAR, variable)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved {len(table)} annual bin observations: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, required=True)
    parser.add_argument('--output-raw-data-dir', type=Path, required=True)
    parser.add_argument('--variables', nargs='+', choices=['SST', 'VWS'], default=['SST', 'VWS'])
    args = parser.parse_args()
    for variable in args.variables:
        prepare(args.raw_data_dir.resolve(), args.output_raw_data_dir.resolve(), variable)


if __name__ == '__main__':
    main()
