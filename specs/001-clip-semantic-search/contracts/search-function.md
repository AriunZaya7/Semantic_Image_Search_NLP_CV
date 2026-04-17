# Contract: search() Function

**Module**: `src/search.py`
**Feature**: 001-clip-semantic-search
**Date**: 2026-04-17

---

## Function Signature

```python
def search(
    query: str,
    top_k: int = 5,
    embeddings_path: str = "embeddings/ViT-B-32_openai.npz",
) -> list[SearchResult]:
    ...
```

---

## Inputs

| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | — | Non-empty, non-whitespace |
| `top_k` | `int` | No | `5` | Must be `> 0` |
| `embeddings_path` | `str` | No | `"embeddings/ViT-B-32_openai.npz"` | Must be a readable `.npz` file |

---

## Output

**Type**: `list[SearchResult]`

```python
class SearchResult(NamedTuple):
    path: str    # relative image path as stored in .npz
    score: float # cosine similarity in [−1.0, 1.0]
```

**Guarantees**:
- Length is `min(top_k, corpus_size)`.
- List is sorted in **descending** order by `score` (`result[0].score >= result[1].score`).
- Returns an empty list (not an error) when corpus is empty.

---

## Errors

| Condition | Exception | Message pattern |
|-----------|-----------|-----------------|
| `query` is empty or whitespace | `ValueError` | `"Query must be a non-empty string"` |
| `top_k <= 0` | `ValueError` | `"top_k must be a positive integer, got {top_k}"` |
| `.npz` file not found | `FileNotFoundError` | `"Embeddings file not found: {path}"` |
| Required key missing in `.npz` | `KeyError` | `"Missing key '{key}' in embeddings file"` |
| `model_name` mismatch | `ValueError` | `"Model mismatch: expected 'ViT-B-32', got '{actual}'"` |
| Embedding dimension ≠ 512 | `ValueError` | `"Expected embedding dim 512, got {actual}"` |
| `len(embeddings) != len(paths)` | `ValueError` | `"Shape mismatch: {n_emb} embeddings but {n_paths} paths"` |

---

## Behaviour Contract

1. **Idempotent**: multiple calls with the same arguments return identical results
   (given the same `.npz` file on disk).
2. **No side effects**: the function does NOT modify the `.npz` file or any global state
   other than the lazy model/tokenizer cache.
3. **Model loaded once**: the CLIP model and tokenizer are initialised at most once per
   process (module-level singleton or `functools.lru_cache`). Subsequent calls reuse the
   cached objects.
4. **CPU-safe**: the function MUST run correctly on a CPU-only machine. GPU is used
   opportunistically if available but is never required.
5. **No pickle**: `np.load` MUST be called with `allow_pickle=False`.

---

## Usage Example

```python
from src.search import search

results = search("a dog running on the beach", top_k=5)
for r in results:
    print(f"{r.score:.4f}  {r.path}")
```

Expected output format:
```
0.3412  data/images/22928793.jpg
0.3187  data/images/16151663.jpg
0.2954  data/images/23008340.jpg
0.2801  data/images/48909501.jpg
0.2633  data/images/3996401.jpg
```

---

## Out of Scope

- Re-embedding images (handled by `src/embed_images.py`)
- CLI interface (separate feature)
- Batch queries (single query per call)
- Filtering by metadata (no metadata in v1)
- GPU device selection by the caller (internal concern)
