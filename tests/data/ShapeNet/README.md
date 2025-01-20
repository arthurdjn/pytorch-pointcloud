# ShapeNet

This folder contains a small subset of the ShapeNet dataset for testing purposes.

## Data

The data is taken from the original ShapeNet dataset, and contains 10 points per objects and two objects per category.

## Generation

The data was generated using the `scripts/generate.py` script.

> [!NOTE]
> You must provide the path to the original raw ShapeNet dataset in the `--data-dir` argument.

```bash
python scripts/generate.py \
    --data-dir /HDD/Datasets/ShapeNet/raw \
    --output-dir ./raw \
    --max-points 10 \
    --max-objects 4
```
