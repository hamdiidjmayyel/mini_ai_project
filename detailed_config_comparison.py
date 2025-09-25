import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import time

def test_configuration(chunk_size, chunk_overlap, k, model_name="all-MiniLM-L6-v2", num_questions=100):
    """Test a specific configuration and return detailed results"""
    
    print(f"\n🧪 Testing: chunk_size={chunk_size}, overlap={chunk_overlap}, k={k}")
    start_time = time.time()
    
    # Load data
    with open("data/train-v2.0.json", "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    
    # Extract contexts
    contexts = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])
    
    # Create chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separator=" "
    )
    
    chunks = []
    for context in contexts:
        context_chunks = splitter.split_text(context)
        chunks.extend(context_chunks)
    
    print(f"   📚 Created {len(chunks)} chunks")
    
    # Build embeddings and index
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    
    # Build ground truth
    ground_truth = {}
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if "answers" in qa and len(qa["answers"]) > 0:
                    ground_truth[qa["question"]] = qa["answers"][0]["text"]
    
    # Test retrieval
    def retrieve_chunks(query, k_val):
        query_emb = model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        scores, indices = index.search(np.array(query_emb, dtype=np.float32), k_val)
        return [chunks[i] for i in indices[0]], scores[0]
    
    # Enhanced answer matching
    def enhanced_answer_matching(answer, chunks_list):
        answer_lower = answer.lower().strip()
        
        # Exact match
        for chunk in chunks_list:
            if answer_lower in chunk.lower():
                return True, "exact_match"
        
        # Word overlap for multi-word answers
        answer_words = set(answer_lower.split())
        if len(answer_words) > 1:
            for chunk in chunks_list:
                chunk_words = set(chunk.lower().split())
                overlap = len(answer_words.intersection(chunk_words))
                if overlap >= len(answer_words) * 0.8:
                    return True, "word_match"
        
        return False, "no_match"
    
    # Run evaluation
    found_count = 0
    questions_checked = 0
    match_types = {"exact_match": 0, "word_match": 0}
    score_sum = 0
    chunk_size_sum = 0
    
    for question, answer in ground_truth.items():
        retrieved_chunks, scores = retrieve_chunks(question, k)
        found, match_type = enhanced_answer_matching(answer, retrieved_chunks)
        
        if found:
            found_count += 1
            match_types[match_type] += 1
        
        # Collect statistics
        score_sum += np.mean(scores)
        chunk_size_sum += np.mean([len(chunk) for chunk in retrieved_chunks])
        
        questions_checked += 1
        if questions_checked >= num_questions:
            break
    
    accuracy = found_count / questions_checked * 100
    avg_score = score_sum / questions_checked
    avg_chunk_size = chunk_size_sum / questions_checked
    processing_time = time.time() - start_time
    
    return {
        "config": f"{chunk_size}/{chunk_overlap}/{k}",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "accuracy": accuracy,
        "found_count": found_count,
        "total_questions": questions_checked,
        "match_types": match_types,
        "total_chunks": len(chunks),
        "avg_retrieval_score": avg_score,
        "avg_retrieved_chunk_size": avg_chunk_size,
        "processing_time": processing_time
    }

def main():
    print("🔬 COMPREHENSIVE CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Test configurations
    configurations = [
        (500, 50, 3),    # Original
        (800, 80, 5),    # Intermediate
        (1000, 100, 5),  # Good
        (1200, 120, 7),  # Previous best
        (1500, 150, 10), # Your requested config
        (1800, 180, 12), # Even larger for comparison
    ]
    
    results = []
    
    for chunk_size, chunk_overlap, k in configurations:
        try:
            result = test_configuration(chunk_size, chunk_overlap, k, num_questions=50)
            results.append(result)
            print(f"   ✅ Accuracy: {result['accuracy']:.2f}% | Time: {result['processing_time']:.1f}s")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'='*80}")
    print("📊 DETAILED COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Table header
    print(f"{'Config':<12} {'Accuracy':<9} {'Found':<6} {'Total Chunks':<12} {'Avg Score':<10} {'Time(s)':<8}")
    print("-" * 80)
    
    # Results table
    for result in results:
        print(f"{result['config']:<12} {result['accuracy']:<8.2f}% {result['found_count']:<6} "
              f"{result['total_chunks']:<12} {result['avg_retrieval_score']:<9.3f} {result['processing_time']:<7.1f}")
    
    # Best configuration analysis
    best_config = results[0]
    print(f"\n🏆 BEST PERFORMING CONFIGURATION:")
    print(f"   📊 Config: {best_config['config']}")
    print(f"   🎯 Accuracy: {best_config['accuracy']:.2f}%")
    print(f"   📚 Total chunks: {best_config['total_chunks']:,}")
    print(f"   ⚡ Processing time: {best_config['processing_time']:.1f} seconds")
    print(f"   📈 Match types: {best_config['match_types']}")
    
    # Your requested configuration analysis
    your_config = next((r for r in results if r['config'] == '1500/150/10'), None)
    if your_config:
        position = results.index(your_config) + 1
        print(f"\n🎯 YOUR REQUESTED CONFIGURATION (1500/150/10):")
        print(f"   📊 Accuracy: {your_config['accuracy']:.2f}%")
        print(f"   📍 Ranking: #{position} out of {len(results)}")
        print(f"   📚 Total chunks: {your_config['total_chunks']:,}")
        print(f"   ⚡ Processing time: {your_config['processing_time']:.1f} seconds")
        print(f"   📈 Match types: {your_config['match_types']}")
        
        if position == 1:
            print("   🎉 BEST CONFIGURATION!")
        elif position <= 2:
            print("   🥈 EXCELLENT - Top tier performance!")
        elif position <= 3:
            print("   🥉 VERY GOOD - Strong performance!")
        else:
            print(f"   👍 GOOD - Solid performance!")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    print("   Configuration trade-offs:")
    
    # Find most efficient (best accuracy/time ratio)
    efficiency_scores = [(r['accuracy'] / r['processing_time'], r) for r in results]
    efficiency_scores.sort(reverse=True)
    most_efficient = efficiency_scores[0][1]
    
    print(f"   🔥 Most Efficient: {most_efficient['config']} "
          f"({most_efficient['accuracy']:.1f}% in {most_efficient['processing_time']:.1f}s)")
    
    # Memory usage estimation
    print(f"\n💾 MEMORY USAGE ESTIMATION:")
    for result in results[:3]:  # Top 3 configs
        chunks_mb = (result['total_chunks'] * result['avg_retrieved_chunk_size'] * 4) / (1024*1024)  # Rough estimate
        print(f"   {result['config']}: ~{chunks_mb:.1f}MB embeddings + {result['total_chunks']:,} chunks")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Production recommendation
    production_configs = [r for r in results if r['accuracy'] >= 90 and r['processing_time'] <= 60]
    if production_configs:
        prod_config = production_configs[0]
        print(f"   🚀 Production Ready: {prod_config['config']} "
              f"({prod_config['accuracy']:.1f}% accuracy, {prod_config['processing_time']:.1f}s setup)")
    
    # Development recommendation
    dev_configs = [r for r in results if r['processing_time'] <= 30]
    if dev_configs:
        dev_config = max(dev_configs, key=lambda x: x['accuracy'])
        print(f"   🛠️  Development: {dev_config['config']} "
              f"({dev_config['accuracy']:.1f}% accuracy, fast {dev_config['processing_time']:.1f}s setup)")
    
    return results

if __name__ == "__main__":
    results = main()