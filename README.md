# Coastal-TC-Persistence

Calculation and plotting scripts for the study *Rising coastal-zone persistence of tropical cyclones*. The scripts reproduce the quantitative panels in the main figures (M1–M4) and Extended Data figures (S1–S10).

## Data

Choose the archive that matches your starting point. Either archive is sufficient for its workflow.

| Workflow | Download from Zenodo |
| --- | --- |
| Regenerate the figure data and then plot the figures | [Raw_data.zip — analysis inputs](https://doi.org/10.5281/zenodo.24270841) |
| Plot the figures directly from the archived results | [Database.zip — processed figure data](https://doi.org/10.5281/zenodo.24269322) |

The analysis inputs include tropical cyclone tracks, distance-to-coast fields, environmental and climate model data, steering-flow estimates, diagnostics from idealized WRF experiments, and estimates of population and urban grid cell exposure to heavy precipitation.

## Setup

Download or clone this repository. From the repository root, create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate coastal-tc
```

## Plot the figures from processed data

Extract `Database.zip` so that the `Database/` folder is alongside `run_pipeline.py`. Then run:

```bash
python run_pipeline.py plot --use-database
```

## Recalculate the figure data and plot the figures

Extract `Raw_data.zip` so that the `Raw_data/` folder is alongside `run_pipeline.py`. Keep the file names and directory structure inside the archive. Then run:

```bash
python run_pipeline.py calculate
python run_pipeline.py plot --use-database
```

The first command runs `Calculate_codes/` and writes the processed data to `Database/`. The second runs `Plotting_codes/` and saves the figures in `Results/`, with a separate folder for each figure.

To run selected figures, add `--figures` to either command. For example, to regenerate and plot main Fig. 2 and Extended Data Figs. 4 and 5:

```bash
python run_pipeline.py calculate --figures M2 S4 S5
python run_pipeline.py plot --use-database --figures M2 S4 S5
```

The script for Extended Data Fig. 7 generates panels c–f. Extended Data Fig. 10 uses Natural Earth map layers, which Cartopy downloads on first use unless they are already cached.
