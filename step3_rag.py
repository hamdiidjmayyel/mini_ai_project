import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os

import json

import json

# Load SQuAD dataset to get ground-truth answers
with open("data/train-v2.0.json", "r") as f:
    squad = json.load(f)

ground_truth = {}
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if "answers" in qa and len(qa["answers"]) > 0:
                ground_truth[qa["question"]] = qa["answers"][0]["text"]

# Load SQuAD dataset (for ground-truth answers)
with open("data/train-v2.0.json", "r") as f:
    squad = json.load(f)

# Build a dict: question → first answer
ground_truth = {}
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if "answers" in qa and len(qa["answers"]) > 0:
                ground_truth[qa["question"]] = qa["answers"][0]["text"]


from dotenv import load_dotenv
load_dotenv()

# Load preprocessed chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Chunks, FAISS index, and embedding model loaded. Ready to query.")

from openai import OpenAI

# Create a client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def retrieve_chunks(query, k=3):
    """
    Retrieve top-k relevant chunks from FAISS for a given query.
    """
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks


def generate_answer(query, k=3):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(query, k=k)
    
    
        # 👉 Print retrieved chunks for inspection
    print("\n--- Retrieved Chunks ---")
    true_answer = ground_truth.get(query, None)
    for i, chunk in enumerate(relevant_chunks, 1):
        display_chunk = chunk
        if true_answer and true_answer.lower() in chunk.lower():
            display_chunk = chunk.replace(true_answer, f">>>{true_answer}<<<")
        print(f"\nChunk {i}:\n{display_chunk[:400]}...")  # show first 400 chars

    

    # Retrieve ground-truth answer if available
    true_answer = ground_truth.get(query, None)

    
    if true_answer:
        print(f"\n[Ground Truth Answer: {true_answer}]")

    # Build prompt
    prompt = "Answer the question using the following information:\n\n"
    prompt += "\n\n".join(relevant_chunks)
    prompt += f"\n\nQuestion: {query}\nAnswer:"

    # Call OpenAI GPT using new API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
        
    llm_answer = response.choices[0].message.content

    print("\n--- Comparison ---")
    print("LLM-generated answer:")
    print(llm_answer)
    if true_answer:
        print("\nGround-truth SQuAD answer:")
        print(true_answer)
    else:
        print("\nNo ground-truth answer found for this question.")

    return llm_answer


query = "What is non-alcoholic fatty liver disease?"
answer = generate_answer(query)

print("\n--- Final Answer ---")

print("Q:", query)
print("A:", answer)

