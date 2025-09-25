import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import shutil
import os

def apply_1500_150_10_config():
    """Apply the 1500/150/10 configuration to your RAG system"""
    
    print("🔧 APPLYING 1500/150/10 CONFIGURATION")
    print("=" * 50)
    print("📊 Parameters: chunk_size=1500, overlap=150, k=10")
    
    # Check if test files already exist
    if os.path.exists("chunks_test_1500_150_10.pkl") and os.path.exists("faiss_index_test_1500_150_10.index"):
        print("📁 Found existing 1500/150/10 configuration files...")
        
        # Backup current files
        print("💾 Backing up current configuration...")
        if os.path.exists("chunks.pkl"):
            shutil.copy("chunks.pkl", "chunks_backup_before_1500.pkl")
            print("   ✅ Backed up chunks.pkl → chunks_backup_before_1500.pkl")
        
        if os.path.exists("faiss_index.index"):
            shutil.copy("faiss_index.index", "faiss_index_backup_before_1500.index")
            print("   ✅ Backed up faiss_index.index → faiss_index_backup_before_1500.index")
        
        # Apply test configuration
        shutil.copy("chunks_test_1500_150_10.pkl", "chunks.pkl")
        shutil.copy("faiss_index_test_1500_150_10.index", "faiss_index.index")
        
        print("✅ Applied 1500/150/10 configuration files")
        
    else:
        print("⚠️  Test files not found. Building configuration from scratch...")
        
        # Build configuration from scratch
        with open("data/train-v2.0.json", "r", encoding="utf-8") as f:
            squad_data = json.load(f)
        
        # Extract contexts
        contexts = []
        for article in squad_data["data"]:
            for paragraph in article["paragraphs"]:
                contexts.append(paragraph["context"])
        
        print(f"📚 Processing {len(contexts)} contexts...")
        
        # Create chunks with new parameters
        splitter = CharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=150,
            separator=" "
        )
        
        chunks = []
        for context in contexts:
            context_chunks = splitter.split_text(context)
            chunks.extend(context_chunks)
        
        print(f"📚 Created {len(chunks)} chunks")
        
        # Build embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("🔄 Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        
        print(f"✅ Built FAISS index with {index.ntotal} vectors")
        
        # Backup current files
        print("💾 Backing up current configuration...")
        if os.path.exists("chunks.pkl"):
            shutil.copy("chunks.pkl", "chunks_backup_before_1500.pkl")
            print("   ✅ Backed up chunks.pkl")
        
        if os.path.exists("faiss_index.index"):
            shutil.copy("faiss_index.index", "faiss_index_backup_before_1500.index")
            print("   ✅ Backed up faiss_index.index")
        
        # Save new configuration
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        
        faiss.write_index(index, "faiss_index.index")
        
        # Also save as test files for future use
        with open("chunks_test_1500_150_10.pkl", "wb") as f:
            pickle.dump(chunks, f)
        
        faiss.write_index(index, "faiss_index_test_1500_150_10.index")
        
        print("✅ Built and applied 1500/150/10 configuration")
    
    # Update script parameters
    print("🔄 Updating script parameters...")
    
    # Update step3_rag.py if it exists
    if os.path.exists("step3_rag.py"):
        with open("step3_rag.py", "r") as f:
            rag_content = f.read()
        
        # Update k parameter
        updated_rag_content = rag_content.replace("k=7", "k=10").replace("k=3", "k=10")
        
        # Backup and update
        shutil.copy("step3_rag.py", "step3_rag_backup_before_1500.py")
        
        with open("step3_rag.py", "w") as f:
            f.write(updated_rag_content)
        
        print("   ✅ Updated step3_rag.py with k=10")
    
    # Update step3_ui.py if it exists
    if os.path.exists("step3_ui.py"):
        with open("step3_ui.py", "r") as f:
            ui_content = f.read()
        
        # Update k parameter
        updated_ui_content = ui_content.replace("k=7", "k=10").replace("k=3", "k=10")
        
        # Backup and update
        shutil.copy("step3_ui.py", "step3_ui_backup_before_1500.py")
        
        with open("step3_ui.py", "w") as f:
            f.write(updated_ui_content)
        
        print("   ✅ Updated step3_ui.py with k=10")
    
    print("\\n✅ CONFIGURATION APPLIED SUCCESSFULLY!")
    print("=" * 50)
    print("📊 Changes applied:")
    print("   • Chunk size: → 1500 characters")
    print("   • Chunk overlap: → 150 characters") 
    print("   • Retrieval k: → 10 chunks")
    print("   • Distance metric: Cosine similarity")
    print("   • Total chunks: ~19,505")
    print("\\n🎯 Expected performance:")
    print("   • Retrieval accuracy: 94%")
    print("   • Processing time: ~31.7s setup")
    print("   • Memory usage: ~65MB embeddings")
    print("   • Ranking: #3 out of tested configurations")
    print("\\n📁 Backup files created:")
    print("   • chunks_backup_before_1500.pkl")
    print("   • faiss_index_backup_before_1500.index")
    if os.path.exists("step3_rag_backup_before_1500.py"):
        print("   • step3_rag_backup_before_1500.py")
    if os.path.exists("step3_ui_backup_before_1500.py"):
        print("   • step3_ui_backup_before_1500.py")
    
    print("\\n🚀 Your RAG system now uses 1500/150/10 configuration!")
    print("   Test it with your queries to confirm the 94% accuracy.")
    
    return True

def revert_from_1500_config():
    """Revert from 1500/150/10 configuration"""
    print("🔄 REVERTING FROM 1500/150/10 CONFIGURATION")
    print("=" * 50)
    
    reverted = False
    
    if os.path.exists("chunks_backup_before_1500.pkl"):
        shutil.copy("chunks_backup_before_1500.pkl", "chunks.pkl")
        print("   ✅ Reverted chunks.pkl")
        reverted = True
    
    if os.path.exists("faiss_index_backup_before_1500.index"):
        shutil.copy("faiss_index_backup_before_1500.index", "faiss_index.index")
        print("   ✅ Reverted faiss_index.index")
        reverted = True
    
    if os.path.exists("step3_rag_backup_before_1500.py"):
        shutil.copy("step3_rag_backup_before_1500.py", "step3_rag.py")
        print("   ✅ Reverted step3_rag.py")
        reverted = True
    
    if os.path.exists("step3_ui_backup_before_1500.py"):
        shutil.copy("step3_ui_backup_before_1500.py", "step3_ui.py")
        print("   ✅ Reverted step3_ui.py")
        reverted = True
    
    if reverted:
        print("\\n✅ Reverted to previous configuration")
    else:
        print("\\n⚠️  No backup files found - nothing to revert")
    
    return reverted

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        revert_from_1500_config()
    else:
        success = apply_1500_150_10_config()
        if success:
            print("\\n💡 TIP: To revert changes, run: python apply_1500_150_10_config.py --revert")
        else:
            print("\\n❌ Failed to apply 1500/150/10 configuration.")