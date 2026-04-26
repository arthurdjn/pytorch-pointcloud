# Transforms

| **Transform**                                                                   | **Description**                                                 |
| :------------------------------------------------------------------------------ | :-------------------------------------------------------------- |
| [`DictTransform`](../api/transforms/transforms.md#dicttransform)                | Base class for dictionary transforms.                           |
| [`RandomSample`](../api/transforms/transforms.md#randomsample)                  | Randomly sample a subset of the data.                           |
| [`RandomSampleFaceVertices`](../api/transforms/transforms.md#randomsamplefacevertices) | Randomly sample face vertices from a point cloud.        |
| [`SampleFarthestPoints`](../api/transforms/transforms.md#samplefarthestpoints)  | Sample farthest points from a point cloud.                      |
| [`NormalizeScale`](../api/transforms/transforms.md#normalizescale)              | Normalize and scale the data.                                   |
| [`RemoveNearOrigin`](../api/transforms/transforms.md#removenearorigin)          | Remove points that are too close to the origin.                 |
| [`Abs`](../api/transforms/transforms.md#abs)                                    | Take the absolute value of the data.                            |
| [`bounding_box`](../api/transforms/functional.md#bounding_box) (functional)      | Min/max bounds along a dimension.                               |
| [`InboxMask`](../api/transforms/transforms.md#inboxmask)                        | Create a mask for the data that is within a given bounding box. |
| [`ApplyMask`](../api/transforms/transforms.md#applymask)                        | Apply a mask to the data.                                       |
