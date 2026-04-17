# Data Model: CLIP Semantic Search

**Feature**: 001-clip-semantic-search
**Date**: 2026-04-17

---

## Entities

### TextQuery

The input to the search function. A natural language string supplied by the caller.

| Field | Type | Constraints |
|-------|------|-------------|
| `text` | `str` | Non-empty, non-whitespace-only |

**Validation rules**:
- MUST NOT be empty or whitespace-only → `ValueError("Query must be a non-empty string")`
- No maximum length enforced at the function level (CLIP tokeniser truncates at 77 tokens
  internally)

**Lifecycle**: Created by the caller, consumed immediately by the search function. Never
persisted.

---

### EmbeddingIndex

The in-memory representation of the `.npz` corpus loaded from disk.

| Field | Type | Shape / Constraints |
|-------|------|---------------------|
| `embeddings` | `np.ndarray[float32]` | `[N, 512]`, L2-normalised rows |
| `paths` | `np.ndarray[str]` | `[N]`, relative file paths |
| `model_name` | `str` | MUST equal `"ViT-B-32"` |

**Validation rules**:
- File MUST exist at `embeddings/ViT-B-32_openai.npz` → `FileNotFoundError` on miss
- `model_name` MUST equal `"ViT-B-32"` → `ValueError` on mismatch
- `embeddings.shape[1]` MUST equal `512` → `ValueError` on mismatch
- `len(embeddings)` MUST equal `len(paths)` → `ValueError` on shape mismatch

**Lifecycle**: Loaded once (lazily or at module import). Never mutated after load.

---

### SearchResult

A single ranked result returned to the caller.

| Field | Type | Constraints |
|-------|------|-------------|
| `path` | `str` | Non-empty; relative path as stored in `.npz` |
| `score` | `float` | In `[−1.0, 1.0]`; cosine similarity (dot product of unit vectors) |

**Representation**: Returned as a plain Python `dataclass` or `NamedTuple`:

```python
from typing import NamedTuple

class SearchResult(NamedTuple):
    path: str
    score: float
```

Using a `NamedTuple` keeps the interface lightweight (no extra dependencies) while allowing
callers to access fields by name (`result.path`, `result.score`) or by position.

**Lifecycle**: Created by the search function, returned to the caller. Never persisted.

---

### SearchRequest (implicit)

Not a standalone object — captured by the function signature:

| Parameter | Type | Default | Constraints |
|-----------|------|---------|-------------|
| `query` | `str` | — (required) | Non-empty |
| `top_k` | `int` | `5` | `> 0` → `ValueError` on violation |
| `embeddings_path` | `str` | `"embeddings/ViT-B-32_openai.npz"` | Must point to valid `.npz` |

`embeddings_path` defaults to the project-standard path but is overridable so tests can
inject a temporary `.npz` fixture without monkey-patching.

---

## State Transitions

The search function is stateless from the caller's perspective. Internally:

```
call search()
    │
    ▼
Validate inputs (query, top_k)
    │
    ▼
Load EmbeddingIndex from .npz
    │  ─ FileNotFoundError if missing
    │  ─ ValueError if model/dim/shape invalid
    ▼
Encode query → 512-d unit vector (torch, no_grad, cpu)
    │
    ▼
Compute similarity scores = embeddings @ query_vec   [shape: N]
    │
    ▼
Select top-K indices (argsort descending, clip to N)
    │
    ▼
Return List[SearchResult] (path, score), sorted descending
```

---

## Storage Format

**File**: `embeddings/ViT-B-32_openai.npz`
**Format**: NumPy compressed archive (`np.savez`)

| Key | dtype | Shape | Description |
|-----|-------|-------|-------------|
| `embeddings` | `float32` | `[N, 512]` | L2-normalised CLIP image vectors |
| `paths` | `<U...` (Unicode str) | `[N]` | Relative image file paths |
| `model_name` | `<U...` (Unicode str) | `()` scalar | MUST be `"ViT-B-32"` |

Loading: `np.load(path, allow_pickle=False)`
