# ModelNetNormalResampled

Synthetic ModelNetNormalResampled: 2 train and 2 test objects per class (1024 points with normals) for variants 10 and
40, plus the split lists and the processed `.dat` caches.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
uv run --no-sync python scripts/generate.py process ./raw
```
