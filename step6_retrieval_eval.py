import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load preprocessed chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load SQuAD data
with open("data/train-v2.0.json", "r") as f:
    squad = json.load(f)

# Build ground-truth dictionary
ground_truth = {}
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if "answers" in qa and len(qa["answers"]) > 0:
                ground_truth[qa["question"]] = qa["answers"][0]["text"]

print("Chunks, FAISS index, model, and SQuAD loaded.")

def retrieve_chunks(query, k=7):
    # Normalize query embedding for cosine similarity (optimized)
    query_emb = model.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks

k = 7  # top-k chunks (optimized from 3)
num_questions = 50  # test on more questions for better statistics

found_count = 0
questions_checked = 0

for question, answer in ground_truth.items():
    retrieved_chunks = retrieve_chunks(question, k=k)

    # Check if ground-truth answer exists in any chunk
    found = any(answer.lower() in chunk.lower() for chunk in retrieved_chunks)

    if found:
        found_count += 1
    # 🔹 Print the question, answer, and retrieved chunks
    print(f"\nQuestion: {question}")
    print(f"Ground-truth Answer: {answer}")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"Chunk {i}: {chunk[:200]}...")  # first 200 chars
    print(f"Answer found in chunks? {found}\n")

    questions_checked += 1
    if questions_checked >= num_questions:
        break
    questions_checked += 1
    if questions_checked >= num_questions:
        break

# Compute retrieval accuracy
accuracy = found_count / questions_checked * 100
print(f"Retrieval Accuracy (top-{k} chunks) on {questions_checked} questions: {accuracy:.2f}%")

print("\nSample retrieved chunks for first question:")
sample_question = list(ground_truth.keys())[0]
sample_answer = ground_truth[sample_question]
retrieved_chunks = retrieve_chunks(sample_question, k=k)
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"\nChunk {i}: {chunk[:400]}...")  # show first 400 chars
print(f"\nGround-truth answer: {sample_answer}")
