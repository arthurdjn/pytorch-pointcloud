# Semantic3D

Tiny Semantic3D fixture covering one train scene (with labels) and one held-out
test scene (without labels), each 1024 points. The fixture is synthetic by
default — Semantic3D requires manual download and license acceptance, so the
generator falls back to structured random data when no source is available.

## Generation

`scripts/generate.py` reads from `$TORCH_POINTCLOUD_DATA_DIR/Semantic3D/raw/`
when present (override with `--src-dir`). When the source isn't available, the
script writes synthetic ASCII files matching the real schema
(`x y z intensity r g b` rows, plus a `.labels` file for train scenes).

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```

You can pass `--train-scenes` / `--test-scenes` to choose other scene names, and
`--num-points` to change the fixture density.
