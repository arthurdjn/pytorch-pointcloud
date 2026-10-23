# 3DMatch

Synthetic 3DMatch: two test scenes of 3 fragments and one train scene of 3 fragments in the PREDATOR release layout
(`indoor/<split>/<scene>/cloud_bin_*.pth`), with the GeoTransformer pair lists under `metadata/`.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
