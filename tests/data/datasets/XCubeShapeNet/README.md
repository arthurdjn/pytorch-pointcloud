# XCubeShapeNet

Tiny XCubeShapeNet fixture in the exact format of the official release
([xrenaa/XCube-Shapenet-Dataset](https://huggingface.co/datasets/xrenaa/XCube-Shapenet-Dataset)):
one synthetic 194-voxel "shape" pickled by the original 2024 fvdb build (openvdb PR-1808) that XCube
was developed against, listed in all three split files. The `points` entry is a pickled `GridBatch`
(a V01-framed in-memory NanoVDB `OnIndex` buffer) and `normals` a `JaggedTensor` aligned to the grid's
voxel order, so the tests exercise the real `XCubeShapeNet.process()` decoding path (stub unpickler +
NanoVDB file-header shim read by fvdb-core).

## Generation

`scripts/generate.py` must run with the *original* fvdb installed (fvdb-core cannot produce the old
pickle bytes). With the `xcube` conda env described in `notebooks/xcube/`:

### Raw

```bash
conda run -n xcube python scripts/generate.py raw ./raw
```

> [!NOTE]
> The processed `.npz` cache is intentionally not committed: the dataset tests run
> `XCubeShapeNet.process()` against this raw pickle in a temporary directory.
