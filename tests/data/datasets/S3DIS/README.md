# S3DIS

This folder contains a small subset of the S3DIS dataset for testing purposes.

## Data

The data is taken from the original S3DIS dataset, and contains 10 points per objects and two objects per category.

## Generation

The data was generated using the `scripts/generate.py` script.

### Raw data generation

To generate a subset of the raw data, run the following command:

> [!NOTE]
> You must provide the path to the original raw S3DIS dataset in the `--data-dir` argument.

```bash
python scripts/generate.py raw \
    /HDD/Datasets/S3DIS/raw \
    ./raw \
    --max-points 10 \
    --max-rooms 2
```

### Processed data generation

To generate the processed data, run the following command:

```bash
python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed data is saved in the `processed` folder.
