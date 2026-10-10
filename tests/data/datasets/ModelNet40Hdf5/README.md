# ModelNet40Hdf5

Synthetic `modelnet40_ply_hdf5_2048`: two train shards and one test shard of 4 objects (2048 points + normals each),
with the release-style `train_files.txt` / `test_files.txt` lists. The shards are read directly, so there is no
processed cache.

```bash
uv run --no-sync python scripts/generate.py ./raw
```
