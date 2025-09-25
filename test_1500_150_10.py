import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain.text_splitter import CharacterTextSplitter

# -------------------------------
# TEST PARAMETERS: 1500/150/10
# -------------------------------
CHUNK_SIZE = 1500       # Your requested chunk size
CHUNK_OVERLAP = 150     # Your requested overlap
K_CHUNKS = 10           # Your requested k value
USE_COSINE = True       # Keep cosine similarity optimization

print("🚀 Testing configuration: chunk_size=1500, overlap=150, k=10")
print(f"📊 Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}, k: {K_CHUNKS}")

# -------------------------------
# 1. Build chunks with new parameters
# -------------------------------
with open("data/train-v2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

contexts = []
for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        contexts.append(paragraph["context"])

# Create chunks with new parameters
splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separator=" "  # Split on spaces for better coherence
)

test_chunks = []
for context in contexts:
    context_chunks = splitter.split_text(context)
    test_chunks.extend(context_chunks)

print(f"📚 Created {len(test_chunks)} chunks with size {CHUNK_SIZE}")

# -------------------------------
# 2. Build FAISS index for test
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
print("🔄 Generating embeddings for test configuration...")
embeddings = model.encode(test_chunks, show_progress_bar=True)

# Use cosine similarity by normalizing embeddings and using Inner Product
if USE_COSINE:
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for cosine similarity
else:
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance

index.add(np.array(embeddings, dtype=np.float32))
print(f"✅ Test FAISS index created with {index.ntotal} vectors")

# -------------------------------
# 3. Build ground-truth dictionary
# -------------------------------
ground_truth = {}
for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if "answers" in qa and len(qa["answers"]) > 0:
                ground_truth[qa["question"]] = qa["answers"][0]["text"]

print(f"📋 Loaded {len(ground_truth)} ground-truth Q&A pairs")

# -------------------------------
# 4. Test retrieval function
# -------------------------------
def retrieve_chunks_test(query, k=K_CHUNKS):
    # Normalize query embedding for cosine similarity
    query_emb = model.encode([query])
    if USE_COSINE:
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # Search in FAISS index
    scores, indices = index.search(np.array(query_emb, dtype=np.float32), k)
    
    relevant_chunks = [test_chunks[i] for i in indices[0]]
    return relevant_chunks, scores[0]

# -------------------------------
# 5. Enhanced answer matching
# -------------------------------
def enhanced_answer_matching(answer, chunks):
    """Enhanced matching with multiple strategies"""
    answer_lower = answer.lower().strip()
    
    # Strategy 1: Exact substring match
    for chunk in chunks:
        if answer_lower in chunk.lower():
            return True, "exact_match"
    
    # Strategy 2: Word-level matching for better partial matches
    answer_words = set(answer_lower.split())
    if len(answer_words) > 1:  # Only for multi-word answers
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(answer_words.intersection(chunk_words))
            if overlap >= len(answer_words) * 0.8:  # 80% word overlap
                return True, "word_match"
    
    return False, "no_match"

# -------------------------------
# 6. Run test evaluation
# -------------------------------
print("\n🎯 Running test evaluation with 1500/150/10 configuration...")
print("=" * 70)

found_count = 0
questions_checked = 0
match_types = {"exact_match": 0, "word_match": 0}

# Test on same number of questions for comparison
NUM_TEST_QUESTIONS = 50

for question, answer in ground_truth.items():
    retrieved_chunks, scores = retrieve_chunks_test(question, k=K_CHUNKS)
    
    # Enhanced answer matching
    found, match_type = enhanced_answer_matching(answer, retrieved_chunks)
    
    if found:
        found_count += 1
        match_types[match_type] += 1
    
    # Print first few examples for debugging
    if questions_checked < 5:
        print(f"\nQuestion {questions_checked + 1}: {question}")
        print(f"Ground-truth Answer: {answer}")
        print(f"Answer found: {found} ({match_type})")
        print(f"Top retrieval scores: {scores[:3]}")
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            print(f"Chunk {i} ({len(chunk)} chars): {chunk[:150]}...")
        print("-" * 50)
    
    questions_checked += 1
    if questions_checked >= NUM_TEST_QUESTIONS:
        break

# -------------------------------
# 7. Results comparison
# -------------------------------
accuracy = found_count / questions_checked * 100

print(f"\n{'='*70}")
print("🎯 TEST RESULTS: 1500/150/10 CONFIGURATION")
print(f"{'='*70}")
print(f"📊 Retrieval Accuracy (top-{K_CHUNKS} chunks): {accuracy:.2f}%")
print(f"🔍 Questions tested: {questions_checked}")
print(f"✅ Answers found: {found_count}")
print(f"📋 Match breakdown: {match_types}")
print(f"💾 Total chunks: {len(test_chunks)}")
print(f"📏 Avg chunk size: {CHUNK_SIZE} chars")

# Compare with previous configurations
print(f"\n📈 COMPARISON WITH PREVIOUS CONFIGURATIONS:")
print(f"   Original (500/50/3): 40.00%")
print(f"   Optimized (1200/120/7): 94.00%")
print(f"   New Test (1500/150/10): {accuracy:.2f}%")

improvement_vs_original = accuracy - 40
improvement_vs_optimized = accuracy - 94

print(f"\n🚀 PERFORMANCE ANALYSIS:")
print(f"   vs Original: {improvement_vs_original:+.2f} percentage points")
print(f"   vs Previous Best: {improvement_vs_optimized:+.2f} percentage points")

if accuracy > 94:
    print("   🎉 NEW BEST CONFIGURATION!")
elif accuracy >= 90:
    print("   ✅ Excellent performance maintained")
elif accuracy >= 80:
    print("   👍 Good performance")
else:
    print("   ⚠️  Performance decrease detected")

# -------------------------------
# 8. Save test results if better
# -------------------------------
if accuracy >= 94:
    print(f"\n💾 Saving new configuration as it performs well...")
    
    # Save test chunks
    with open("chunks_test_1500_150_10.pkl", "wb") as f:
        pickle.dump(test_chunks, f)
    
    # Save test index
    faiss.write_index(index, "faiss_index_test_1500_150_10.index")
    
    print("✅ Saved test configuration files:")
    print("📁 chunks_test_1500_150_10.pkl")
    print("📁 faiss_index_test_1500_150_10.index")
    
    if accuracy > 94:
        print(f"\n🎊 CONGRATULATIONS! New best accuracy: {accuracy:.2f}%")
        print("   Consider applying this configuration to your production system.")

print(f"\n🏁 Test complete!")
print(f"📊 Final accuracy with 1500/150/10 configuration: {accuracy:.1f}%")