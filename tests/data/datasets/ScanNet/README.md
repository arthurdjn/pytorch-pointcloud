# ScanNet

This folder contains a small subset of the ScanNet dataset for testing purposes.

## Data

The generated data will contain random points with no semantic meaning, for testing purposes. The structure of the data will respect the original ScanNet dataset structure however.

## Generation

The data was generated using the `scripts/generate.py` script.

### Raw data generation

To generate a subset of the raw data, run the following command:

> [!NOTE]
> You must provide the path to the original raw S3DIS dataset in the `--data-dir` argument.

```bash
python scripts/generate.py raw \
    ./raw \
    --split train \
    --version v2
```

### Processed data generation

To generate the processed data, run the following command:

```bash
python scripts/generate.py process ./raw --version v2 --split train
```

> [!NOTE]
> The processed data is saved in the `processed` folder.
