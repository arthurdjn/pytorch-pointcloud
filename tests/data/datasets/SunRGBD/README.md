# SUN RGB-D

Tiny SUN RGB-D fixture obtained by subsampling real scenes from the official release. For each
split we keep the first 3 scenes (in `allsplit.mat` order); the early frames include both
annotated and empty-box scenes, so the fixture exercises the populated and the zero-box paths.

Unlike the other datasets, SUN RGB-D's loader streams from the official zips (metadata + split out
of `SUNRGBDtoolbox.zip`, depth/RGB out of `SUNRGBD.zip`) without ever extracting them, so `raw/`
ships *tiny subset zips* rather than loose files:

- `raw/SUNRGBDtoolbox.zip`: the real `allsplit.mat` (0.16 MB, so `read_split`'s full 5285 / 5050
  count assert passes) plus a `SUNRGBDMeta.mat` rebuilt with only the kept scenes.
- `raw/SUNRGBD.zip`: only the kept scenes' depth/RGB PNG members.
- `processed/<split>/`: the loader's output on the raw fixture, subsampled to 2048 points (boxes are
  absolute, so subsampling the cloud leaves them untouched).

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/SunRGBD` by default (override with
`--src-dir`). The `raw` command builds the subset zips from the real release; the `process` command
runs the unchanged loader on those zips and subsamples, so `processed` is literally `process(raw)`.

```bash
uv run --no-sync python scripts/generate.py raw
uv run --no-sync python scripts/generate.py process
```

### Full regeneration

```bash
bash scripts/generate.sh
```
