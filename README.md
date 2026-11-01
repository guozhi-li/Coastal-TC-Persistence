# Coastal-TC-Persistence

Plotting scripts for reproducing the main and supplementary figures of a study on the coastal-zone persistence of tropical cyclones.

## Data

Processed data are available on Zenodo:
[10.5281/zenodo.23863244](https://doi.org/10.5281/zenodo.23863244).

Download and extract `Database.zip`, then place the `Database` folder alongside the `Codes` folder:

```text
Coastal-TC-Persistence/
├── Codes/
├── Database/
└── Results/
```

The plotting scripts read the processed data directly. The `Results` folder is created when the scripts are run.

## Usage

Run individual plotting scripts from the project directory, for example:

```bash
python Codes/001_M1_CAL_CLTC.py
```

Keep the original names and directory structure of the data files.
