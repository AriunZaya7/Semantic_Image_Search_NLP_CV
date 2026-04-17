"""Unit tests for src/search.py — Uses mock_index fixture from conftest.py."""

import numpy as np
import pytest

from src.search import SearchResult, search


# ===========================================================================
# User Story 1 — Text Query Returns Ranked Images
# ===========================================================================

def test_returns_top_k_results(mock_index):
    """search() returns exactly top_k results when corpus >= top_k."""
    results = search("sunset over mountains", top_k=5, embeddings_path=mock_index)
    assert len(results) == 5


def test_results_sorted_descending(mock_index):
    """Results are ordered from highest to lowest cosine similarity."""
    results = search("a dog on the beach", top_k=10, embeddings_path=mock_index)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            f"Not sorted at position {i}: {results[i].score} < {results[i+1].score}"
        )


def test_empty_query_raises(mock_index):
    """Empty or whitespace-only queries must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        search("", top_k=5, embeddings_path=mock_index)
    with pytest.raises(ValueError, match="non-empty"):
        search("   ", top_k=5, embeddings_path=mock_index)


def test_missing_file_raises():
    """FileNotFoundError is raised when the embeddings file does not exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        search("test query", top_k=5, embeddings_path="nonexistent.npz")


# ===========================================================================
# User Story 2 — Configurable Number of Results
# ===========================================================================

def test_top_k_one(mock_index):
    """top_k=1 returns exactly one result."""
    results = search("mountain lake", top_k=1, embeddings_path=mock_index)
    assert len(results) == 1


def test_top_k_full_corpus(mock_index):
    """top_k equal to corpus size returns all items."""
    results = search("forest path", top_k=50, embeddings_path=mock_index)
    assert len(results) == 50


def test_top_k_exceeds_corpus(mock_index):
    """top_k larger than corpus returns all corpus items without error."""
    results = search("night sky", top_k=200, embeddings_path=mock_index)
    assert len(results) == 50  # mock_index has 50 items


def test_invalid_top_k_zero_raises(mock_index):
    """top_k=0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        search("test", top_k=0, embeddings_path=mock_index)


def test_invalid_top_k_negative_raises(mock_index):
    """Negative top_k must raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        search("test", top_k=-1, embeddings_path=mock_index)


def test_empty_corpus_returns_empty(tmp_path):
    """Empty corpus (0 embeddings) returns an empty list, no exception."""
    npz_path = tmp_path / "ViT-B-32_openai.npz"
    np.savez(
        str(npz_path),
        embeddings=np.zeros((0, 512), dtype=np.float32),
        paths=np.array([], dtype=str),
        model_name="ViT-B-32",
    )
    results = search("anything", top_k=5, embeddings_path=str(npz_path))
    assert results == []


# ===========================================================================
# User Story 3 — Similarity Scores Exposed to Caller
# ===========================================================================

def test_result_has_path_and_score(mock_index):
    """Each SearchResult exposes a non-empty path and a numeric score."""
    results = search("city skyline", top_k=5, embeddings_path=mock_index)
    for r in results:
        assert isinstance(r, SearchResult)
        assert isinstance(r.path, str) and r.path, "path must be a non-empty string"
        assert isinstance(r.score, float), "score must be a float"


def test_scores_in_valid_range(mock_index):
    """Cosine similarity scores must lie in [−1.0, 1.0]."""
    results = search("ocean waves", top_k=10, embeddings_path=mock_index)
    for r in results:
        assert -1.0 <= r.score <= 1.0, f"Score out of range: {r.score}"


def test_score_ordering_strict(mock_index):
    """Scores are non-increasing across 10 distinct queries."""
    queries = [
        "a cat", "sunset", "mountain", "beach", "forest",
        "city", "river", "snow", "desert", "rain",
    ]
    for q in queries:
        results = search(q, top_k=20, embeddings_path=mock_index)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, (
                f"Query '{q}': not sorted at position {i}"
            )


# ===========================================================================
# Data integrity error handling
# ===========================================================================

def test_model_mismatch_raises(tmp_path):
    """Wrong model_name in .npz must raise ValueError."""
    n, d = 10, 512
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    npz_path = tmp_path / "ViT-B-32_openai.npz"
    np.savez(
        str(npz_path),
        embeddings=emb,
        paths=np.array([f"img_{i}.jpg" for i in range(n)]),
        model_name="ViT-L-14",  # wrong model
    )
    with pytest.raises(ValueError, match="Model mismatch"):
        search("test", embeddings_path=str(npz_path))


def test_wrong_dimension_raises(tmp_path):
    """Embeddings with wrong dimension must raise ValueError."""
    n = 10
    wrong_dim = 768  # not 512
    emb = np.random.randn(n, wrong_dim).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    npz_path = tmp_path / "ViT-B-32_openai.npz"
    np.savez(
        str(npz_path),
        embeddings=emb,
        paths=np.array([f"img_{i}.jpg" for i in range(n)]),
        model_name="ViT-B-32",
    )
    with pytest.raises(ValueError, match="dim"):
        search("test", embeddings_path=str(npz_path))
