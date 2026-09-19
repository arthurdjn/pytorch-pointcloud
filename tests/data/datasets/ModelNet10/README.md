# ModelNet10

Synthetic ModelNet10: 2 train and 2 test `.off` meshes per class (256 vertices each), plus the processed `train.pt` /
`test.pt`.

```bash
uv run --no-sync python scripts/generate.py raw ./raw --variant 10
uv run --no-sync python scripts/generate.py process ./raw --variant 10
```
