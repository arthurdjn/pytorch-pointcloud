# NuScenesMini

Synthetic nuScenes `v1.0-mini`: one `mini_val` scene with 2 keyframes of 3 LiDAR clouds each (1024 points), plus the 9
metadata tables. It holds a moving car, a pedestrian, a barrier and a non-detection category.

```bash
uv run --no-sync python scripts/generate.py
```
