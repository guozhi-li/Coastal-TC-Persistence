"""Read annual native-grid CMIP6 bin means used by the figure calculations."""
from itertools import product

import numpy as np
import pandas as pd


def read_annual_bins(path, models, experiments, edges, first_year, last_year, variable):
    table = pd.read_csv(path, float_precision='round_trip')
    keys = ['Model', 'Experiment', 'Distance_Left_km', 'Year']
    expected = pd.MultiIndex.from_tuples(
        product(models, experiments, edges[:-1], range(first_year, last_year + 1)),
        names=keys)
    actual = pd.MultiIndex.from_frame(table[keys])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError(f'{path}: incomplete or duplicated model, experiment, bin or year')
    right = table.Distance_Left_km.map(dict(zip(edges[:-1], edges[1:])))
    center = (table.Distance_Left_km + right) / 2
    if not (table.Distance_Right_km.eq(right) & table.Distance_Center_km.eq(center)).all():
        raise ValueError(f'{path}: inconsistent distance bounds')
    if not np.isfinite(table[variable]).all() or not table.Grid_Cell_Count.gt(0).all():
        raise ValueError(f'{path}: invalid annual {variable} observations or bin counts')
    return table.sort_values(keys).groupby(['Model', 'Experiment'], sort=False)
