# Research: CLIP Semantic Search

**Feature**: 001-clip-semantic-search
**Date**: 2026-04-17
**Status**: Complete — no NEEDS CLARIFICATION items remained in spec

---

## 1. OpenCLIP Query Encoding

**Decision**: Use `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`
and `open_clip.get_tokenizer('ViT-B-32')`. Encode text with `model.encode_text(tokens)`,
then L2-normalise the result.

**Rationale**: The project already uses ViT-B-32/openai weights (mock embeddings are
generated with `model_name="ViT-B-32"`). Using the same model and pretraining ensures
query vectors live in the same embedding space as stored image vectors.

**Alternatives considered**:
- `sentence-transformers` CLIP wrapper — rejected; adds a dependency and is a wrapper
  around the same underlying model. Direct `open_clip` is already a project dependency.
- `transformers` CLIP — rejected; different tokenisation/normalisation pipeline, would
  produce incompatible embeddings.

**Best practices applied**:
- Load model and tokenizer **once at module level** (or via a lazy singleton), not inside
  the search function, to avoid repeated disk I/O on every call.
- Call `model.eval()` after loading; wrap encoding in `torch.no_grad()` to prevent
  gradient graph allocation.
- Move model to CPU explicitly (`model.to('cpu')`) for guaranteed CPU-only test execution;
  allow caller to override device if GPU is available.

---

## 2. Cosine Similarity via Dot Product

**Decision**: Compute similarity as `embeddings @ query_vec` (matrix–vector dot product)
where both sides are L2-normalised float32 NumPy arrays.

**Rationale**: Cosine similarity between unit vectors equals their dot product. The spec
(FR-005, Assumptions) confirms embeddings are L2-normalised at write time
(`create_mock_embeddings.py` normalises with `/ np.linalg.norm(...)`). Normalising the
query vector before the dot product completes the equivalence. This is O(N·D) with a
single NumPy call — the most efficient approach for the scale involved.

**Alternatives considered**:
- `scipy.spatial.distance.cosine` per-pair loop — rejected; O(N) Python loop, ~100× slower
  for N=200+.
- `sklearn.metrics.pairwise.cosine_similarity` — rejected; adds sklearn dependency for a
  one-liner that NumPy already handles.
- FAISS index — rejected; adds a heavy dependency; corpus size (~200–10k images) doesn't
  justify approximate nearest-neighbour indexing.

---

## 3. Top-K Selection

**Decision**: Use `np.argsort(scores)[::-1][:top_k]` to obtain the indices of the top-K
scores in descending order, then gather paths and scores by index.

**Rationale**: Simple, dependency-free, and correct for the corpus sizes in scope.
`np.argpartition` would be faster for very large K but adds complexity without benefit
at this scale.

**Alternatives considered**:
- `heapq.nlargest` — viable but mixes Python and NumPy; `np.argsort` keeps everything
  in NumPy and is simpler.
- `torch.topk` — would require keeping tensors in PyTorch throughout; NumPy is preferred
  for the return-value layer to avoid torch tensor leaking into the public interface.

---

## 4. Embeddings File Loading

**Decision**: Load `embeddings/ViT-B-32_openai.npz` with `np.load(path)`, extract
`embeddings` (float32 array, shape `[N, 512]`), `paths` (string array, shape `[N]`),
and `model_name` (scalar string or 0-d array).

**Rationale**: The file format is fixed by `create_mock_embeddings.py` and the project
convention. Validation order: (1) file exists, (2) required keys present, (3)
`model_name == "ViT-B-32"`, (4) `embeddings.shape[1] == 512`, (5)
`len(embeddings) == len(paths)`.

**Best practices applied**:
- Use `np.load(..., allow_pickle=False)` to prevent arbitrary code execution from
  untrusted `.npz` files.
- Raise `FileNotFoundError` (not a generic `Exception`) so callers can distinguish missing
  file from corrupt file.
- Raise `ValueError` for data integrity violations (wrong model, wrong dimension, shape
  mismatch) with the offending value in the message.

---

## 5. Test Strategy (CPU-only)

**Decision**: Use `create_mock_embeddings.py` output (or an in-test fixture that creates
a temporary `.npz` with `np.random` vectors) as the test corpus. No real images or GPU
required.

**Rationale**: Constitution Principle IV requires `pytest` to pass on CPU. The search
function's correctness (sorting, top-K selection, error handling) is fully verifiable
with random normalised vectors. Model encoding is the only GPU-optional step; it can
be mocked or skipped in unit tests by injecting a pre-computed query vector.

**Test fixture pattern**:
```python
@pytest.fixture
def mock_index(tmp_path):
    n, d = 50, 512
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = np.array([f"data/images/img_{i:03d}.jpg" for i in range(n)])
    npz = tmp_path / "ViT-B-32_openai.npz"
    np.savez(npz, embeddings=emb, paths=paths, model_name="ViT-B-32")
    return str(npz)
```
