# ModelNet40

Synthetic ModelNet40: 2 train and 2 test `.off` meshes per class (256 vertices each), plus the processed `train.pt` /
`test.pt`.

```bash
uv run --no-sync python scripts/generate.py raw ./raw --variant 40
uv run --no-sync python scripts/generate.py process ./raw --variant 40
```
