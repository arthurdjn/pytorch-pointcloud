# SemanticKITTI

Synthetic SemanticKITTI: sequences 00, 08 and 11 with 2 scans of 1024 points each. Sequence 11 has no labels, like the
real test split.

```bash
uv run --no-sync python scripts/generate.py raw ./raw
```
