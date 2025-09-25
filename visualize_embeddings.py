import pickle
import umap
import matplotlib.pyplot as plt

# Load chunks and embeddings
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Example: if you already have embeddings in memory, skip loading
# Otherwise, regenerate embeddings (MiniLM)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Reduce embeddings to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(embeddings)

print("Reduced embeddings shape:", emb_2d.shape)

plt.figure(figsize=(10, 8))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.5)
plt.title("Visualization of SQuAD Chunk Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()
# %matplotlib inline
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=5, alpha=0.5)

