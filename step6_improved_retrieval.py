import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class ImprovedRetrieval:
    def __init__(self, chunk_size=1000, chunk_overlap=100, k=5, model_name="all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def create_improved_chunks(self, contexts):
        """Create chunks with improved parameters"""
        from langchain.text_splitter import CharacterTextSplitter
        
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            separator=" "  # Split on spaces for better coherence
        )
        
        chunks = []
        for context in contexts:
            context_chunks = splitter.split_text(context)
            chunks.extend(context_chunks)
        
        return chunks
    
    def build_improved_index(self, chunks):
        """Build FAISS index with improved embeddings"""
        print(f"Generating embeddings with {self.model_name}...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Use IndexFlatIP (Inner Product) for better semantic similarity
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        
        return index, embeddings
    
    def retrieve_chunks(self, query, chunks, index, k=None):
        """Retrieve chunks with improved search"""
        if k is None:
            k = self.k
            
        # Encode and normalize query
        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = index.search(np.array(query_emb, dtype=np.float32), k)
        
        relevant_chunks = [(chunks[i], scores[0][idx]) for idx, i in enumerate(indices[0])]
        return relevant_chunks
    
    def improved_answer_matching(self, answer, chunks_with_scores, threshold=0.3):
        """Improved answer matching with multiple strategies"""
        answer_lower = answer.lower().strip()
        
        # Strategy 1: Exact substring match
        for chunk, score in chunks_with_scores:
            if answer_lower in chunk.lower():
                return True, "exact_match", score
        
        # Strategy 2: Fuzzy matching for partial answers
        answer_words = set(answer_lower.split())
        for chunk, score in chunks_with_scores:
            chunk_words = set(chunk.lower().split())
            overlap = len(answer_words.intersection(chunk_words))
            if overlap >= len(answer_words) * 0.7:  # 70% word overlap
                return True, "fuzzy_match", score
        
        # Strategy 3: Semantic similarity for numerical/date answers
        if any(char.isdigit() for char in answer):
            answer_emb = self.model.encode([answer])
            answer_emb = answer_emb / np.linalg.norm(answer_emb, axis=1, keepdims=True)
            
            for chunk, score in chunks_with_scores:
                chunk_emb = self.model.encode([chunk])
                chunk_emb = chunk_emb / np.linalg.norm(chunk_emb, axis=1, keepdims=True)
                
                semantic_sim = cosine_similarity(answer_emb, chunk_emb)[0][0]
                if semantic_sim > threshold:
                    return True, "semantic_match", semantic_sim
        
        return False, "no_match", 0.0

def run_improved_evaluation():
    # Load SQuAD data
    with open("data/train-v2.0.json", "r") as f:
        squad = json.load(f)
    
    # Build ground-truth dictionary
    ground_truth = {}
    contexts = []
    
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])
            for qa in paragraph["qas"]:
                if "answers" in qa and len(qa["answers"]) > 0:
                    ground_truth[qa["question"]] = qa["answers"][0]["text"]
    
    print("Testing different parameter configurations...\n")
    
    # Test different configurations
    configs = [
        {"chunk_size": 500, "chunk_overlap": 50, "k": 3, "model": "all-MiniLM-L6-v2"},
        {"chunk_size": 800, "chunk_overlap": 80, "k": 5, "model": "all-MiniLM-L6-v2"},
        {"chunk_size": 1000, "chunk_overlap": 100, "k": 5, "model": "all-MiniLM-L6-v2"},
        {"chunk_size": 1200, "chunk_overlap": 120, "k": 7, "model": "all-MiniLM-L6-v2"},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing config: {config}")
        
        # Initialize retrieval system
        retrieval = ImprovedRetrieval(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            k=config["k"],
            model_name=config["model"]
        )
        
        # Create improved chunks
        chunks = retrieval.create_improved_chunks(contexts)
        print(f"Created {len(chunks)} chunks")
        
        # Build improved index
        index, embeddings = retrieval.build_improved_index(chunks)
        
        # Test on subset of questions
        found_count = 0
        questions_checked = 0
        match_types = {"exact_match": 0, "fuzzy_match": 0, "semantic_match": 0}
        
        for question, answer in ground_truth.items():
            chunks_with_scores = retrieval.retrieve_chunks(question, chunks, index, config["k"])
            found, match_type, confidence = retrieval.improved_answer_matching(answer, chunks_with_scores)
            
            if found:
                found_count += 1
                match_types[match_type] += 1
            
            questions_checked += 1
            if questions_checked >= 20:  # Test on 20 questions for faster evaluation
                break
        
        accuracy = found_count / questions_checked * 100
        results.append({
            "config": config,
            "accuracy": accuracy,
            "match_types": match_types,
            "total_chunks": len(chunks)
        })
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Match types: {match_types}")
        print(f"Total chunks: {len(chunks)}\n")
    
    # Print summary
    print("=" * 60)
    print("CONFIGURATION COMPARISON RESULTS:")
    print("=" * 60)
    
    best_config = max(results, key=lambda x: x["accuracy"])
    
    for i, result in enumerate(results):
        config = result["config"]
        print(f"\nConfig {i+1}:")
        print(f"  Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}, k: {config['k']}")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Total chunks: {result['total_chunks']}")
        print(f"  Match breakdown: {result['match_types']}")
        if result == best_config:
            print("  ⭐ BEST CONFIGURATION")
    
    print(f"\n🎯 Best accuracy achieved: {best_config['accuracy']:.2f}%")
    print(f"🎯 Best configuration: {best_config['config']}")
    
    return best_config

if __name__ == "__main__":
    best_config = run_improved_evaluation()