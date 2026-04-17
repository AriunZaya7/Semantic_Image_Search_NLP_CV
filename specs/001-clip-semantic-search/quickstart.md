# Quickstart: CLIP Semantic Search

**Feature**: 001-clip-semantic-search
**Date**: 2026-04-17

---

## Prerequisites

```bash
# 1. Install dependencies
uv sync

# 2. Generate mock embeddings (no GPU required)
python src/create_mock_embeddings.py
# Output: embeddings/ViT-B-32_openai.npz
```

---

## Running the Search Function

```python
# From a Python REPL or script at the repository root
from src.search import search

results = search("a dog running on the beach", top_k=5)
for r in results:
    print(f"{r.score:.4f}  {r.path}")
```

Expected output (scores will vary with mock/random embeddings):
```
0.3412  data/images/22928793.jpg
0.2954  data/images/16151663.jpg
...
```

---

## Running Tests

```bash
# All unit tests — CPU only, no GPU required
pytest tests/unit/test_search.py -v
```

All tests must pass before merging. No GPU marker is required for these tests.

---

## Validation Checklist

- [x] `python src/create_mock_embeddings.py` runs without error and creates
  `embeddings/ViT-B-32_openai.npz`
- [x] `from src.search import search` imports without error
- [x] `search("sunset over mountains", top_k=3)` returns exactly 3 results sorted
  descending by score
- [x] `search("", top_k=5)` raises `ValueError`
- [x] `search("test", top_k=0)` raises `ValueError`
- [x] `search("test", embeddings_path="nonexistent.npz")` raises `FileNotFoundError`
- [x] `pytest tests/unit/test_search.py` passes on CPU-only machine (15/15, 5.22s)

---

## Common Issues

**`FileNotFoundError: Embeddings file not found`**
→ Run `python src/create_mock_embeddings.py` first to generate the `.npz` file.

**`ValueError: Model mismatch`**
→ The `.npz` was generated with a different model. Delete `embeddings/` and regenerate.

**`ModuleNotFoundError: No module named 'open_clip'`**
→ Run `uv sync` to install dependencies.

**Slow first call**
→ The CLIP model is downloaded (~350 MB) on first use and cached by OpenCLIP.
Subsequent calls reuse the cached model with no disk I/O.
