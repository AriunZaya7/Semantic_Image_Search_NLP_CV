import numpy as np
import pytest


@pytest.fixture
def mock_index(tmp_path):
    """50 normalised float32 vectors (512-d) — no GPU required."""
    n, d = 50, 512
    rng = np.random.default_rng(seed=42)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = np.array([f"data/images/img_{i:03d}.jpg" for i in range(n)])
    npz_path = tmp_path / "ViT-B-32_openai.npz"
    np.savez(str(npz_path), embeddings=emb, paths=paths, model_name="ViT-B-32")
    return str(npz_path)
