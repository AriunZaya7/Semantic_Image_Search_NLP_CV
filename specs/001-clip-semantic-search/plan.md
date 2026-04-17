# Implementation Plan: CLIP Semantic Search

**Branch**: `001-clip-semantic-search` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-clip-semantic-search/spec.md`

## Summary

Implement a Python callable `search(query, top_k=5)` in `src/search.py` that encodes a natural
language query using OpenCLIP ViT-B-32 (pretrained `openai`), loads pre-computed 512-dimensional
image embeddings from `embeddings/ViT-B-32_openai.npz`, computes cosine similarity via dot
product (embeddings are L2-normalised), and returns the top-K `(path, score)` pairs sorted
descending. Model and tokenizer are loaded once at module level to avoid per-call overhead.
All error cases (empty query, bad `top_k`, missing/corrupt file, model mismatch, wrong dimension)
raise descriptive exceptions. Tests run on CPU with mock embeddings.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: open-clip-torch ≥ 3.3.0, torch ≥ 2.11.0, numpy ≥ 2.4.4
**Storage**: NumPy `.npz` files — `embeddings/ViT-B-32_openai.npz`
**Testing**: pytest (CPU-only; mock `.npz` via `src/create_mock_embeddings.py` output)
**Target Platform**: Local machine (macOS/Linux); GPU optional, never required
**Project Type**: Python library (callable function, no CLI surface for this feature)
**Performance Goals**: Top-5 results returned in < 500 ms on CPU (excluding first model load)
**Constraints**: Must pass `pytest` on CPU-only machine; no GPU required for tests
**Scale/Scope**: Single user, local corpus (~200 images in mock; real corpus size TBD)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Cross-Modal Embedding First | ✅ PASS | Core of the feature — dot-product over CLIP embeddings |
| II. Simplicity & Reproducibility | ✅ PASS | Single callable, no new setup steps beyond `uv sync` |
| III. Data Integrity | ✅ PASS | FR-006 (model_name check) + FR-012 (dim=512 check) enforced |
| IV. Testability | ✅ PASS | FR-009: tests run on CPU with mock embeddings, no GPU needed |
| V. Minimal Footprint | ✅ PASS | No new dependencies; search logic lives in `src/search.py` only |

**Gate result**: All principles satisfied. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/001-clip-semantic-search/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── search-function.md   # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/
├── search.py            # search() function (this feature)
├── embed_images.py      # image embedding pipeline (existing stub)
├── app.py               # Streamlit UI (existing stub)
├── verify.py            # verification utilities (existing stub)
└── create_mock_embeddings.py  # mock data generator (existing)

tests/
└── unit/
    └── test_search.py   # unit tests for search() — CPU-only, mock embeddings
```

**Structure Decision**: Single-project layout. All search logic goes into `src/search.py`.
Tests are co-located under `tests/unit/` to keep the repo flat and consistent with the
constitution's reproducibility requirement (one `pytest` invocation from root).

## Complexity Tracking

> No constitution violations detected — section intentionally empty.
