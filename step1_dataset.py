import json
from langchain.text_splitter import CharacterTextSplitter
import pickle

# -------------------------------
# 1. Load SQuAD v2.0 train set
# -------------------------------
with open("data/train-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

contexts = []
qas_pairs = []

for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        contexts.append(context)
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"] if qa["answers"] else ""
            qas_pairs.append((question, answer, context))

print(f"Total contexts: {len(contexts)}")
print(f"Total Q&A pairs: {len(qas_pairs)}")

# -------------------------------
# 2. Split contexts into chunks
# -------------------------------
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for context in contexts:
    context_chunks = splitter.split_text(context)
    chunks.extend(context_chunks)

print(f"Total chunks created: {len(chunks)}")

# Save chunks for next step
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved to chunks.pkl")
