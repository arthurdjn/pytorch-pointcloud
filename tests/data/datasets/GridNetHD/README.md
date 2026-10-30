# GridNet-HD

Synthetic GridNet-HD Hub release: one train, one val and one test zone (1024 points each, LAS 1.2) in raw EPSG:32632 coordinates, the test zone without the `ground_truth` field.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
