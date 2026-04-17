---
description: "Task list for CLIP Semantic Search implementation"
---

# Tasks: CLIP Semantic Search

**Input**: Design documents from `specs/001-clip-semantic-search/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/search-function.md ✅

**Tests**: Included — required by Constitution Principle IV (pytest MUST pass on CPU-only machine)
and Success Criterion SC-004.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths included in all descriptions

---

## Phase 1: Setup

**Purpose**: Project scaffolding needed before any implementation begins.

- [x] T001 Create `tests/` and `tests/unit/` directories with `__init__.py` files
- [x] T002 [P] Create `tests/unit/__init__.py` (empty, marks package)
- [x] T003 [P] Create `tests/conftest.py` with `mock_index` pytest fixture that writes a
  temporary `ViT-B-32_openai.npz` (50 normalised float32 vectors, 512-d, random paths) to
  `tmp_path` — no GPU required

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures and shared infrastructure that every user story depends on.
No user story implementation can begin until this phase is complete.

**⚠️ CRITICAL**: US1, US2, and US3 all depend on these tasks.

- [x] T004 Define `SearchResult(NamedTuple)` with fields `path: str` and `score: float` at
  the top of `src/search.py`
- [x] T005 [P] Implement `_load_index(embeddings_path: str)` in `src/search.py` that loads
  `embeddings/ViT-B-32_openai.npz` with `np.load(..., allow_pickle=False)`, validates
  `model_name == "ViT-B-32"`, `shape[1] == 512`, and `len(embeddings) == len(paths)`, and
  raises `FileNotFoundError` / `ValueError` / `KeyError` per contracts/search-function.md
- [x] T006 Implement module-level lazy singleton in `src/search.py` that loads
  `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')` and
  `open_clip.get_tokenizer('ViT-B-32')` on first call, calls `model.eval()`, and moves model
  to CPU — reused across all `search()` calls

**Checkpoint**: Foundation ready — US1, US2, US3 implementation can now begin. ✅

---

## Phase 3: User Story 1 — Text Query Returns Ranked Images (Priority: P1) 🎯 MVP

**Goal**: A text query produces an ordered list of top-K `(path, score)` image results.

**Independent Test**: Call `search("any text", top_k=5, embeddings_path=<fixture>)` using the
`mock_index` fixture; assert 5 results returned, sorted descending, each with non-empty `path`
and float `score`.

### Tests for User Story 1

- [x] T007 [P] [US1] Write `tests/unit/test_search.py::test_returns_top_k_results` — calls
  `search("sunset", top_k=5, embeddings_path=mock_index)` and asserts `len(results) == 5`
- [x] T008 [P] [US1] Write `tests/unit/test_search.py::test_results_sorted_descending` —
  asserts `results[i].score >= results[i+1].score` for all consecutive pairs
- [x] T009 [P] [US1] Write `tests/unit/test_search.py::test_empty_query_raises` — asserts
  `ValueError` on `search("", ...)` and `search("   ", ...)`
- [x] T010 [P] [US1] Write `tests/unit/test_search.py::test_missing_file_raises` — asserts
  `FileNotFoundError` when `embeddings_path="nonexistent.npz"`

### Implementation for User Story 1

- [x] T011 [US1] Implement `search(query: str, top_k: int = 5, embeddings_path: str = "embeddings/ViT-B-32_openai.npz") -> list[SearchResult]` in `src/search.py`:
  validate `query` (non-empty/non-whitespace → `ValueError`), call `_load_index`, encode query
  with the lazy singleton via `torch.no_grad()`, L2-normalise the query vector, compute
  `scores = embeddings @ query_vec`, run `np.argsort(scores)[::-1][:top_k]`, return
  `[SearchResult(path, score) for ...]`

**Checkpoint**: `search("test query", top_k=5, embeddings_path=<fixture>)` works end-to-end;
T007–T010 pass. ✅

---

## Phase 4: User Story 2 — Configurable Number of Results (Priority: P2)

**Goal**: `top_k` is a runtime parameter; callers receive exactly `min(top_k, corpus_size)` results.

**Independent Test**: Call `search("test", top_k=1, ...)`, `search("test", top_k=50, ...)`,
and `search("test", top_k=200, ...)` against the 50-item mock corpus; assert lengths are 1, 50,
and 50 respectively.

### Tests for User Story 2

- [x] T012 [P] [US2] Write `tests/unit/test_search.py::test_top_k_one` — `search("t", top_k=1,
  ...)` returns exactly 1 result
- [x] T013 [P] [US2] Write `tests/unit/test_search.py::test_top_k_exceeds_corpus` — `search("t",
  top_k=200, ...)` against 50-item corpus returns exactly 50 results
- [x] T014 [P] [US2] Write `tests/unit/test_search.py::test_invalid_top_k_raises` — asserts
  `ValueError` on `top_k=0` and `top_k=-1`
- [x] T015 [P] [US2] Write `tests/unit/test_search.py::test_empty_corpus_returns_empty` —
  fixture with 0 embeddings; asserts `search(...)` returns `[]` without exception

### Implementation for User Story 2

- [x] T016 [US2] Add `top_k` validation in `src/search.py::search` — raise
  `ValueError(f"top_k must be a positive integer, got {top_k}")` when `top_k <= 0`
- [x] T017 [US2] Ensure `np.argsort(scores)[::-1][:top_k]` in `src/search.py` naturally handles
  `top_k > N` (slicing beyond array length returns all elements — verified correct, invariant
  documented with inline comment)

**Checkpoint**: All four `top_k` scenarios (1, full corpus, over-corpus, invalid) pass; T012–T015
pass. ✅

---

## Phase 5: User Story 3 — Similarity Scores Exposed to Caller (Priority: P3)

**Goal**: Each `SearchResult` carries a float `score` in `[−1, 1]`; ordering is strictly
non-increasing.

**Independent Test**: Inspect return values of any `search()` call; assert `result.score` is
a Python `float`, that `score[0] >= score[1]` for all consecutive pairs, and that scores lie
in `[−1.0, 1.0]`.

### Tests for User Story 3

- [x] T018 [P] [US3] Write `tests/unit/test_search.py::test_result_has_path_and_score` —
  asserts each result in a 5-item response has a non-empty string `path` and a numeric `score`
- [x] T019 [P] [US3] Write `tests/unit/test_search.py::test_scores_in_valid_range` — asserts
  all scores are in `[−1.0, 1.0]` for a standard search call
- [x] T020 [P] [US3] Write `tests/unit/test_search.py::test_score_ordering_strict` — asserts
  `result[i].score >= result[i+1].score` for all i across 10 distinct query strings

### Implementation for User Story 3

- [x] T021 [US3] Verified in `src/search.py` that `SearchResult.score` values are cast to Python
  `float` via `float(scores[i])` before return — clean public interface confirmed

**Checkpoint**: All three score-related tests pass (T018–T020). ✅

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, documentation, and full-suite validation.

- [x] T022 [P] Add module-level docstring to `src/search.py` describing the public API
  (`search()`, `SearchResult`)
- [x] T023 Add inline comment in `src/search.py` next to `allow_pickle=False` explaining the
  security rationale
- [x] T024 [P] Run `pytest tests/unit/test_search.py -v` — 15/15 passed on CPU (5.22s)
- [x] T025 Execute the quickstart.md validation checklist — all 7 items verified
- [x] T026 [P] Verified `python src/create_mock_embeddings.py && search('sunset', top_k=3)` runs
  end-to-end: returns 3 scored results without error

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1 — P1)**: Depends on Phase 2 — MVP blocker
- **Phase 4 (US2 — P2)**: Depends on Phase 2; can start after Phase 3 checkpoint or in parallel
- **Phase 5 (US3 — P3)**: Depends on Phase 2; can start after Phase 3 checkpoint or in parallel
- **Phase 6 (Polish)**: Depends on all story phases being complete

### User Story Dependencies

- **US1 (P1)**: Depends on Foundational (Phase 2) only — no US2/US3 dependency
- **US2 (P2)**: Depends on Foundational (Phase 2) — independent of US1 completion; shares
  `search()` signature
- **US3 (P3)**: Depends on Foundational (Phase 2) — independent of US1/US2 completion; verifies
  the `SearchResult` contract

### Within Each User Story

- Tests MUST be written and FAIL before implementation tasks begin
- `SearchResult` definition (T004) before any return-value tests
- `_load_index` (T005) before `search()` body (T011)
- Lazy singleton (T006) before query encoding in T011

### Parallel Opportunities

- T001, T002, T003 — all Phase 1 tasks can run in parallel
- T005, T006 — independent of each other within Phase 2
- T007–T010 — all US1 tests can be written in parallel
- T012–T015 — all US2 tests can be written in parallel
- T018–T020 — all US3 tests can be written in parallel
- US2 and US3 phases can be worked concurrently after Phase 2 completes

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T006) — CRITICAL, blocks everything
3. Write US1 tests (T007–T010) — verify they FAIL
4. Implement `search()` (T011)
5. **STOP and VALIDATE**: `pytest tests/unit/test_search.py::test_returns_top_k_results`
   and `::test_results_sorted_descending` pass
6. Demo: `python -c "from src.search import search; print(search('sunset'))"` works

### Incremental Delivery

1. Setup + Foundational → skeleton ready
2. US1 → MVP: basic text-to-image search works
3. US2 → top_k fully validated (edge cases hardened)
4. US3 → score contract formally tested
5. Polish → full suite green, quickstart validated

### Parallel Team Strategy

With two developers:
- **Dev A after Phase 2**: US1 (P1) — core search function
- **Dev B after Phase 2**: US2 (P2) + US3 (P3) — parameter validation and score contract

---

## Notes

- `[P]` tasks modify different files or are independent stubs — safe to run concurrently
- `[US*]` label maps each task to its user story for traceability
- Tests MUST fail before implementation — do not write passing tests first
- The `mock_index` fixture in `conftest.py` (T003) is the only test dependency beyond `uv sync`
- `search()` is stateless from the caller's perspective; model caching is an internal detail
- Commit after each phase checkpoint to preserve incremental progress
