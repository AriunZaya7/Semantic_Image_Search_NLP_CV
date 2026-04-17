"""
src/search.py — CLIP Semantic Search

Public API:
    search(query, top_k=5, embeddings_path="embeddings/ViT-B-32_openai.npz")
        -> list[SearchResult]

    SearchResult(path: str, score: float)
        Named tuple returned for each ranked image result.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import open_clip
import torch


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

class SearchResult(NamedTuple):
    path: str    # relative image path as stored in the .npz file
    score: float # cosine similarity in [−1.0, 1.0]; higher = more relevant


# ---------------------------------------------------------------------------
# Internal: embedding index loader
# ---------------------------------------------------------------------------

_EXPECTED_MODEL_NAME = "ViT-B-32"
_EXPECTED_DIM = 512


def _load_index(embeddings_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the .npz embedding index.

    Returns:
        (embeddings, paths) where embeddings.shape == (N, 512), L2-normalised.

    Raises:
        FileNotFoundError: if the file does not exist.
        KeyError: if a required array key is missing.
        ValueError: on model name mismatch, wrong dimension, or shape mismatch.
    """
    try:
        # allow_pickle=False prevents arbitrary code execution from untrusted files
        data = np.load(embeddings_path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    for key in ("embeddings", "paths", "model_name"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in embeddings file: {embeddings_path}")

    model_name = str(data["model_name"])
    if model_name != _EXPECTED_MODEL_NAME:
        raise ValueError(
            f"Model mismatch: expected '{_EXPECTED_MODEL_NAME}', got '{model_name}'"
        )

    embeddings: np.ndarray = data["embeddings"]
    paths: np.ndarray = data["paths"]

    if embeddings.ndim != 2 or embeddings.shape[1] != _EXPECTED_DIM:
        actual = embeddings.shape
        raise ValueError(
            f"Expected embedding dim {_EXPECTED_DIM}, got shape {actual}"
        )

    if len(embeddings) != len(paths):
        raise ValueError(
            f"Shape mismatch: {len(embeddings)} embeddings but {len(paths)} paths"
        )

    return embeddings, paths


# ---------------------------------------------------------------------------
# Internal: model / tokenizer
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_model_and_tokenizer():
    """Load ViT-B-32 (OpenAI weights) once per process and cache the result."""
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    model.to("cpu")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_k: int = 5,
    embeddings_path: str = "embeddings/ViT-B-32_openai.npz",
) -> list[SearchResult]:
    """Return the top-K images most semantically similar to *query*.

    Args:
        query:           Natural language search query (non-empty).
        top_k:           Maximum number of results to return (must be > 0).
        embeddings_path: Path to the .npz index file.

    Returns:
        List of SearchResult(path, score), sorted descending by cosine similarity.
        Length is min(top_k, corpus_size). Empty list when corpus is empty.

    Raises:
        ValueError:       Empty/whitespace query, top_k ≤ 0, or data integrity errors.
        FileNotFoundError: Missing embeddings file.
        KeyError:         Missing key in embeddings file.
    """
    # Input validation
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")

    # Load index
    embeddings, paths = _load_index(embeddings_path)

    # Empty corpus is valid — return nothing
    if len(embeddings) == 0:
        return []

    # Encode query
    model, tokenizer = _get_model_and_tokenizer()
    tokens = tokenizer([query])  # shape: (1, context_length)
    with torch.no_grad():
        query_vec = model.encode_text(tokens)  # shape: (1, 512)

    # L2-normalise so dot product == cosine similarity
    query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
    query_np = query_vec.squeeze(0).cpu().numpy().astype(np.float32)  # (512,)

    # Score and rank
    # embeddings are already L2-normalised → dot product == cosine similarity
    scores: np.ndarray = embeddings @ query_np  # (N,)

    # argsort ascending, reverse for descending; slice to top_k (safe when N < top_k)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        SearchResult(path=str(paths[i]), score=float(scores[i]))
        for i in top_indices
    ]
