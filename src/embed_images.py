
"""
This script:
1. Loads images from a folder
2. Converts them into embeddings using CLIP
3. Saves embeddings for later use
"""

import os
import torch
import clip
import numpy as np
from PIL import Image


# -----------------------------
# STEP 1: Setup device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# STEP 2: Load CLIP model
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=device)


# -----------------------------
# STEP 3: Embed all images
# -----------------------------
def embed_images(image_folder):
    embeddings = []
    image_paths = []

    for file in os.listdir(image_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            path = os.path.join(image_folder, file)

            try:
                image = Image.open(path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_embedding = model.encode_image(image_input)

                # Normalize embeddings
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                embeddings.append(image_embedding.cpu().numpy())
                image_paths.append(path)

                print(f"Encoded: {file}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    embeddings = np.vstack(embeddings)
    return embeddings, image_paths


# -----------------------------
# STEP 4: Save embeddings
# -----------------------------
def save_embeddings(embeddings, paths, file="embeddings.npz"):
    np.savez(file, embeddings=embeddings, paths=paths)
    print(" Embeddings saved image")


# -----------------------------
# MAIN (RUN THIS FILE ONLY ONCE)
# -----------------------------
if __name__ == "__main__":

    IMAGE_FOLDER = "../data/images"   # change path if needed

    print("Generating embeddings...")

    embeddings, paths = embed_images(IMAGE_FOLDER)
    save_embeddings(embeddings, paths)

    print(" Done Embeddings ready for search module.")