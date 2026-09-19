# ShapeNetPart

Synthetic ShapeNetPart: 4 objects per category (1024 points with normals and part labels; 2 train, 1 val, 1 test), plus
the split lists and the `processed/` cache.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```
