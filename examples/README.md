# Examples

One directory per model, one plain-PyTorch script per task and dataset:

```text
examples/<model>/<task>_<dataset>_<kind>.py
```

`kind` is `train` (the authors' recipe, from scratch or from a pretrained encoder) or `benchmark` (a registry
checkpoint evaluated with the reference protocol through the library's inferers and metrics). Every script is
argparse only: `--help` lists its flags.

```bash
uv run --no-sync python examples/pointnext/segmentation_s3dis_benchmark.py --root data --model pointnext-sm.s3dis-area5.openpoints
uv run --no-sync python examples/pointnext/segmentation_s3dis_train.py --root data --model pointnext-sm.s3dis-area5.openpoints --epochs 100 --output runs/pointnext
```

## Flags

| Flag                                            | Scripts    | Meaning                                                                     |
| ----------------------------------------------- | ---------- | --------------------------------------------------------------------------- |
| `--root`                                        | all        | Dataset root, `data` or `TORCH_POINTCLOUD_DATA_DIR` by default              |
| `--model`                                       | all        | Registered model name (`tp.list_models()`)                                  |
| `--device`, `--seed`, `--batch-size`, `--num-workers` | all   | Runtime                                                                     |
| `--download`, `--force-process`                 | datasets with a download or a cache | Fetch the release, rebuild the `processed/` cache  |
| `--limit N`                                     | benchmarks | Evaluate at most `N` samples                                                |
| `--epochs`, `--lr`, `--eval-every`, `--output`  | training   | Schedule, validation period, directory for `last.pt` and `best.pt`         |
| `--limit-train-batches`, `--limit-val-batches`  | training   | Shorten an epoch                                                            |

Protocol knobs are per script (`--sw-batch-size`, `--areas`, `--split-file`, `--max-sweeps`, ...).

`benchmark.sh` runs every pretrained checkpoint of the zoo, one line per checkpoint:

```bash
ROOT=/path/to/datasets bash examples/benchmark.sh
```

## Results

The measured scores are the `score` column of the model catalog, `docs/data/models.csv`, next to the `reference`
numbers the authors publish; the registry `metrics` are rendered on the
[model pages](https://pytorch-pointcloud.org/latest/models/overview/).

## Tests

`tests/examples/test_examples.py` runs every script on the dummy datasets under `tests/data/datasets`:

```bash
uv run --no-sync pytest tests/examples --run-examples --run-pretrained
```

Training scripts need `--run-examples`; benchmarks load registry weights, so they also need `--run-pretrained`.
