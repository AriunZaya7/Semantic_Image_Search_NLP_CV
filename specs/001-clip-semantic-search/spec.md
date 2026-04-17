# Feature Specification: CLIP Semantic Search

**Feature Branch**: `001-clip-semantic-search`
**Created**: 2026-04-17
**Status**: Draft
**Input**: User description: "CLIP semantic search function that takes a natural language query and returns top-K images ranked by cosine similarity"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Text Query Returns Ranked Images (Priority: P1)

A user provides a natural language text query (e.g., "a dog running on the beach") and receives
an ordered list of the top-K most semantically relevant images from the indexed corpus, ranked
from highest to lowest cosine similarity score.

**Why this priority**: This is the core deliverable — the entire feature exists to satisfy this
scenario. Without it, there is no product.

**Independent Test**: Can be fully tested by calling the search function with a text query string
and a pre-built set of mock embeddings (no GPU required), verifying that the returned list has at
most K items, is sorted in descending similarity order, and each item carries an image path and
score.

**Acceptance Scenarios**:

1. **Given** a corpus of pre-computed image embeddings and a text query,
   **When** the search function is called with `query="sunset over mountains"` and `top_k=5`,
   **Then** it returns exactly 5 results, each with an image path and a similarity score in [0, 1],
   sorted from highest to lowest score.

2. **Given** a corpus smaller than K (e.g., 3 images, `top_k=10`),
   **When** the search function is called,
   **Then** it returns all 3 results (no padding, no error) sorted by similarity.

3. **Given** an empty corpus (zero embeddings),
   **When** the search function is called,
   **Then** it returns an empty list and does not raise an exception.

---

### User Story 2 - Configurable Number of Results (Priority: P2)

A caller can specify how many results to retrieve (K) without modifying source code or
configuration files — K is a parameter of the search function.

**Why this priority**: Downstream consumers (UI, evaluation scripts) need different K values.
Hardcoding K would require forking the function.

**Independent Test**: Call the function with multiple different `top_k` values (1, 5, 20) against
the same corpus and verify each call returns exactly `min(top_k, corpus_size)` items.

**Acceptance Scenarios**:

1. **Given** a corpus of 50 images,
   **When** the function is called with `top_k=1`,
   **Then** exactly 1 result is returned.

2. **Given** a corpus of 50 images,
   **When** the function is called with `top_k=50`,
   **Then** all 50 results are returned in descending similarity order.

---

### User Story 3 - Similarity Scores Exposed to Caller (Priority: P3)

Each result includes the cosine similarity score alongside the image path, so callers can apply
their own threshold filtering or display confidence information to end users.

**Why this priority**: Scores are necessary for downstream ranking decisions and UI display but
do not affect the core search correctness already covered by P1.

**Independent Test**: Inspect the return value of a search call and assert that each result
object/tuple exposes both an image path (string) and a numeric score.

**Acceptance Scenarios**:

1. **Given** a search call that returns results,
   **When** the caller accesses each result,
   **Then** each result exposes `.path` (or equivalent) as a non-empty string and `.score`
   (or equivalent) as a float in [−1, 1] (cosine similarity range).

2. **Given** two results in the returned list at positions 0 and 1,
   **When** their scores are compared,
   **Then** `score[0] >= score[1]` (descending order enforced).

---

### Edge Cases

- What happens when the query is an empty string or whitespace-only?
  → The function MUST raise a `ValueError` with a descriptive message.
- What happens when the embeddings file is missing or corrupt?
  → The function MUST raise an `IOError`/`FileNotFoundError` with the file path in the message.
- What happens when the embeddings file was produced by a different model than the one used to
  encode the query?
  → The function MUST detect the `model_name` mismatch and raise a descriptive error (per
  Constitution Principle III — Data Integrity).
- What happens when `top_k` is 0 or negative?
  → The function MUST raise a `ValueError`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The search function MUST accept a natural language text string as its primary input.
- **FR-002**: The search function MUST accept a `top_k` integer parameter (default: 5) controlling
  the maximum number of results returned.
- **FR-003**: The search function MUST return results sorted in descending cosine similarity order.
- **FR-004**: Each result MUST expose the image file path and the cosine similarity score.
- **FR-005**: The function MUST load pre-computed image embeddings from
  `embeddings/ViT-B-32_openai.npz`; it MUST NOT re-embed images on every call.
- **FR-006**: The function MUST validate that the `model_name` stored in the embeddings file
  matches `"ViT-B-32"`, raising a descriptive error on mismatch.
- **FR-007**: The function MUST handle a corpus smaller than `top_k` by returning all available
  results without error.
- **FR-008**: The function MUST raise a `ValueError` for invalid inputs (empty query, `top_k ≤ 0`).
- **FR-009**: The function MUST be callable without a GPU (CPU inference acceptable for query
  encoding; mock embeddings acceptable in tests).
- **FR-010**: The text query MUST be encoded using the ViT-B-32 model loaded via
  `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`.
- **FR-011**: Query tokenisation MUST use `open_clip.get_tokenizer('ViT-B-32')`.
- **FR-012**: The function MUST expect embedding vectors of exactly 512 dimensions (as produced
  by ViT-B-32); if the loaded embeddings have a different second dimension the function MUST raise
  a descriptive `ValueError`.

### Key Entities

- **TextQuery**: A natural language string provided by the caller; the unit of search input.
  Tokenised with `open_clip.get_tokenizer('ViT-B-32')` before encoding.
- **ImageEmbedding**: A normalised float32 vector of exactly 512 dimensions representing an image
  in ViT-B-32 CLIP embedding space, paired with an image file path. Stored collectively in
  `embeddings/ViT-B-32_openai.npz`.
- **SearchResult**: A pairing of an image file path (string) and a cosine similarity score (float).
  The ordered collection of these is the function's return value.
- **EmbeddingIndex**: The loaded, in-memory representation of `embeddings/ViT-B-32_openai.npz` —
  the corpus of 512-dimensional ImageEmbeddings against which queries are scored.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A text query against a corpus of 200 images returns the top-5 results in under
  500 ms on a CPU-only machine (excluding first-time model load).
- **SC-002**: Results are correctly sorted — the highest-scoring result is always at position 0,
  verified across at least 10 distinct queries.
- **SC-003**: The function passes all edge-case scenarios (empty query, missing file, model
  mismatch, `top_k` out of range) with descriptive errors, not crashes or silent failures.
- **SC-004**: The function is testable on a CPU-only machine using mock embeddings with zero
  additional setup beyond `uv sync`.

## Assumptions

- The image corpus has already been embedded and saved to `embeddings/ViT-B-32_openai.npz`
  before the search function is called. Embedding generation is out of scope for this feature.
- The CLIP model is **ViT-B-32** loaded via
  `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`. Switching models is
  a separate concern and constitutes a breaking change per Constitution Principle I.
- The tokenizer is `open_clip.get_tokenizer('ViT-B-32')`.
- Embedding vectors are exactly **512-dimensional** float32, L2-normalised, as produced by
  ViT-B-32. Cosine similarity therefore reduces to a dot product; the function may rely on this.
- Image paths stored in the `.npz` file are relative to the repository root; the function returns
  them as-is without resolving to absolute paths.
- Thread safety and concurrent access to the embeddings file are out of scope for v1.
- The function is a Python callable (not a CLI command); a CLI wrapper is a separate concern.
