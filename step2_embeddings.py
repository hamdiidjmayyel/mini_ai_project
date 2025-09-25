# step2_embeddings.py
# -------------------
# Step 2: Generate embeddings for preprocessed chunks and build a FAISS index
# Author: Hamdi Idjmayyel
# Description: This script loads chunks.pkl, creates embeddings using sentence-transformers,
#              builds a FAISS index for similarity search, and saves the index to disk.

import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# 1. Load preprocessed chunks
# -------------------------------
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks to embed: {len(chunks)}")
print("Example chunk:", chunks[0][:200], "...")

# -------------------------------
# 2. Create embeddings
# -------------------------------
# Using MiniLM model for lightweight semantic embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all chunks
print("Generating embeddings... This may take a few minutes.")
embeddings = embed_model.encode(chunks)
print("Embeddings generated!")
print("Embeddings shape:", embeddings.shape)  # Should be (num_chunks, 384)

# -------------------------------
# 3. Build FAISS index
# -------------------------------
dimension = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(np.array(embeddings, dtype=np.float32))  # Add all embeddings to index

print("FAISS index created!")
print("FAISS index contains", index.ntotal, "vectors")

# -------------------------------
# 4. Save FAISS index
# -------------------------------
faiss.write_index(index, "faiss_index.index")
print("FAISS index saved to faiss_index.index")
