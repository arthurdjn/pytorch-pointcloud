# ScanObjectNN

This folder contains synthetic ScanObjectNN data for testing purposes.

## Data

The generated data contains random point clouds with no semantic meaning. Each `.h5` file has one object per class (15 objects total, labels 0-14), with 10 points each. The directory structure mirrors the original ScanObjectNN dataset layout (splits, variants, background/no-background).

## Generation

The data was generated using the `scripts/generate.py` script.

### Raw data generation

To regenerate the raw `.h5` files, run the following command:

```bash
python scripts/generate.py raw ./raw
```

### Processed data generation

To regenerate the processed `.npz` files from the raw data, run the following command:

```bash
python scripts/generate.py process ./raw
```

> [!NOTE]
> The processed data is saved in the `processed` folder.

### Full regeneration

To regenerate both raw and processed data at once:

```bash
bash scripts/generate.sh
```
