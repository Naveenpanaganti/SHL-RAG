"""
Pre-compute catalog embeddings locally and save to data/embeddings.npy.

Run this ONCE locally before deploying:
    python build_embeddings.py

The output file (data/embeddings.npy) is committed to git and loaded at
runtime instead of running an embedding model on the server.

Memory at runtime: ~2MB (377 vectors × 384 dims × 4 bytes) vs ~400MB for
loading any embedding model. Fits comfortably on Render free tier.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import os

CATALOG_PATH = "data/catalog.json"
OUTPUT_PATH = "data/embeddings.npy"

print("Loading catalog...")
with open(CATALOG_PATH, encoding="utf-8") as f:
    raw = json.load(f)

# Import normalization and text-building from vectorstore
from app.vectorstore import _normalize, _item_to_text

catalog = [_normalize(item) for item in raw if item.get("name") and item.get("link")]
print(f"Catalog items: {len(catalog)}")

texts = [_item_to_text(item) for item in catalog]
print(f"Texts to embed: {len(texts)}")

print("Loading fastembed model (BAAI/bge-small-en-v1.5)...")
from fastembed import TextEmbedding
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("Encoding...")
vectors = np.array(list(embedder.embed(texts)), dtype=np.float32)

# L2-normalize
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1.0
vectors /= norms

print(f"Vectors shape: {vectors.shape}")
np.save(OUTPUT_PATH, vectors)
print(f"Saved to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB)")
print("Done. Commit data/embeddings.npy to git.")
