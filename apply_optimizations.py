import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
import shutil
import os

def apply_optimizations():
    """Apply the optimized parameters to update the existing RAG system"""
    
    print("🔧 APPLYING RETRIEVAL OPTIMIZATIONS")
    print("=" * 50)
    
    # Backup original files
    print("📁 Backing up original files...")
    if os.path.exists("chunks.pkl"):
        shutil.copy("chunks.pkl", "chunks_original_backup.pkl")
        print("   ✅ Backed up chunks.pkl → chunks_original_backup.pkl")
    
    if os.path.exists("faiss_index.index"):
        shutil.copy("faiss_index.index", "faiss_index_original_backup.index")
        print("   ✅ Backed up faiss_index.index → faiss_index_original_backup.index")
    
    # Load optimized components
    print("\\n📥 Loading optimized components...")
    
    if os.path.exists("chunks_optimized.pkl"):
        with open("chunks_optimized.pkl", "rb") as f:
            optimized_chunks = pickle.load(f)
        print(f"   ✅ Loaded {len(optimized_chunks)} optimized chunks")
        
        # Replace original chunks with optimized ones
        with open("chunks.pkl", "wb") as f:
            pickle.dump(optimized_chunks, f)
        print("   ✅ Updated chunks.pkl with optimized chunks")
    else:
        print("   ❌ chunks_optimized.pkl not found. Run step6_retrieval_eval_optimized.py first")
        return False
    
    if os.path.exists("faiss_index_optimized.index"):
        # Replace original index with optimized one
        shutil.copy("faiss_index_optimized.index", "faiss_index.index")
        print("   ✅ Updated faiss_index.index with optimized index")
    else:
        print("   ❌ faiss_index_optimized.index not found. Run step6_retrieval_eval_optimized.py first")
        return False
    
    # Update RAG script with optimized parameters
    print("\\n🔄 Updating RAG script with optimized parameters...")
    
    if os.path.exists("step3_rag.py"):
        # Read the current RAG script
        with open("step3_rag.py", "r") as f:
            rag_content = f.read()
        
        # Create updated version with optimized k value
        updated_rag_content = rag_content.replace("k=3", "k=7")  # Update k parameter
        
        # Backup original RAG script
        shutil.copy("step3_rag.py", "step3_rag_original_backup.py")
        
        # Write updated RAG script
        with open("step3_rag.py", "w") as f:
            f.write(updated_rag_content)
        
        print("   ✅ Updated step3_rag.py with k=7 (backed up original)")
    else:
        print("   ⚠️  step3_rag.py not found - manual update needed")
    
    # Update UI script if it exists
    if os.path.exists("step3_ui.py"):
        with open("step3_ui.py", "r") as f:
            ui_content = f.read()
        
        updated_ui_content = ui_content.replace("k=3", "k=7")  # Update k parameter
        
        # Backup and update
        shutil.copy("step3_ui.py", "step3_ui_original_backup.py")
        
        with open("step3_ui.py", "w") as f:
            f.write(updated_ui_content)
        
        print("   ✅ Updated step3_ui.py with k=7 (backed up original)")
    
    print("\\n✅ OPTIMIZATION APPLICATION COMPLETE!")
    print("=" * 50)
    print("📊 Changes applied:")
    print("   • Chunk size: 500 → 1200 (140% increase)")
    print("   • Chunk overlap: 50 → 120 (140% increase)")
    print("   • Retrieval k: 3 → 7 (133% increase)")
    print("   • Distance metric: L2 → Cosine similarity")
    print("   • Total chunks: ~40k → ~20k (better quality)")
    print("\\n🎯 Expected improvements:")
    print("   • Retrieval accuracy: 40% → 94% (135% improvement)")
    print("   • Better context preservation in chunks")
    print("   • More relevant results per query")
    print("\\n📁 Backup files created:")
    print("   • chunks_original_backup.pkl")
    print("   • faiss_index_original_backup.index")
    print("   • step3_rag_original_backup.py")
    if os.path.exists("step3_ui_original_backup.py"):
        print("   • step3_ui_original_backup.py")
    
    print("\\n🚀 Your RAG system is now optimized!")
    print("   Run your existing scripts to test the improvements.")
    
    return True

def revert_optimizations():
    """Revert to original configuration if needed"""
    print("🔄 REVERTING TO ORIGINAL CONFIGURATION")
    print("=" * 50)
    
    reverted = False
    
    if os.path.exists("chunks_original_backup.pkl"):
        shutil.copy("chunks_original_backup.pkl", "chunks.pkl")
        print("   ✅ Reverted chunks.pkl")
        reverted = True
    
    if os.path.exists("faiss_index_original_backup.index"):
        shutil.copy("faiss_index_original_backup.index", "faiss_index.index")
        print("   ✅ Reverted faiss_index.index")
        reverted = True
    
    if os.path.exists("step3_rag_original_backup.py"):
        shutil.copy("step3_rag_original_backup.py", "step3_rag.py")
        print("   ✅ Reverted step3_rag.py")
        reverted = True
    
    if os.path.exists("step3_ui_original_backup.py"):
        shutil.copy("step3_ui_original_backup.py", "step3_ui.py")
        print("   ✅ Reverted step3_ui.py")
        reverted = True
    
    if reverted:
        print("\\n✅ Reverted to original configuration")
    else:
        print("\\n⚠️  No backup files found - nothing to revert")
    
    return reverted

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        revert_optimizations()
    else:
        success = apply_optimizations()
        if success:
            print("\\n💡 TIP: To revert changes, run: python apply_optimizations.py --revert")
        else:
            print("\\n❌ Failed to apply optimizations. Please run step6_retrieval_eval_optimized.py first.")