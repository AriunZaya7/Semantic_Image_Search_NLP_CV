import numpy as np
import os

os.makedirs("embeddings", exist_ok=True)

# 200 fake normalised embeddings of shape (200, 512)
embeddings = np.random.randn(200, 512).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# fake paths matching real image filenames
paths = np.array([
    "data/images/22928793.jpg",
    "data/images/22930048.jpg",
    "data/images/23008340.jpg",
    "data/images/23018702.jpg",
    "data/images/21954377.jpg",
    "data/images/12243003.jpg",
    "data/images/16151663.jpg",
    "data/images/1440465.jpg",
    "data/images/2148982.jpg",
    "data/images/3996401.jpg",
    "data/images/48909501.jpg",
    "data/images/6901333.jpg",
    "data/images/10404007.jpg",
    "data/images/15875060.jpg",
    "data/images/16495609.jpg",
    "data/images/6276165800.jpg",
    "data/images/3662865.jpg",
    "data/images/29850055.jpg",
    "data/images/20977655.jpg",
    "data/images/24183660.jpg",
    *[f"data/images/img_{i:03d}.jpg" for i in range(180)]
])

np.savez(
    "embeddings/ViT-B-32_openai.npz",
    embeddings=embeddings,
    paths=paths,
    model_name="ViT-B-32"
)

print(f"Mock embeddings created: {embeddings.shape}")
print(f"Paths: {len(paths)}")
