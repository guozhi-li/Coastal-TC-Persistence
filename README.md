# Coastal-TC-Persistence

Calculation and plotting scripts for the study *Rising coastal-zone persistence of tropical cyclones*. The scripts reproduce the quantitative panels in the main figures (M1–M4) and Extended Data figures (S1–S10).

## Data

`Raw_data/` contains the analysis inputs used to calculate the figure data. `Database/` contains the processed data used for plotting. Place either folder in the repository root and preserve its internal directory structure.

The processed data used by the current plotting scripts are available on [Zenodo (v1.1.0)](https://doi.org/10.5281/zenodo.24269322). Download and extract `Database.zip`, then place the `Database/` folder alongside `run_pipeline.py`.

## Project structure

```text
Coastal-TC-Persistence/
├── Calculate_codes/    # Scripts that read Raw_data/ and generate Database/
├── Plotting_codes/     # Scripts that read Database/ and generate figures
├── Raw_data/           # Analysis inputs
├── Database/           # Processed figure data
├── Results/            # Generated figures
├── environment.yml
└── run_pipeline.py
```

## Setup

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate coastal-tc
```

## Reproduce the figures from processed data

With `Database/` in the repository root, run:

```bash
python run_pipeline.py plot --use-database
```

Figures are saved in `Results/`. To plot a single figure, specify its identifier. For example, to reproduce Fig. 2:

```bash
python run_pipeline.py plot --figures M2 --use-database
```

## Recalculate the figure data from the analysis inputs

With `Raw_data/` in the repository root, run:

```bash
python run_pipeline.py calculate
python run_pipeline.py plot --use-database
```

The first command generates `Database/`. The second reads those data and saves the figures in `Results/`.
