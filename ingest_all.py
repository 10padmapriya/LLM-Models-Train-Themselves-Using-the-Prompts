"""
ingest_all.py - One-time document ingestion
Run this ONCE before trainer.py to load all documents into ChromaDB.
"""
import os
from rag import RAGPipeline

def main():
    rag = RAGPipeline()

    sources = {
        "docs/domain_qa":  "Data Science & ML for Business",
        "docs/code":       "Code Assistance (pandas, sklearn, Python)",
        "docs/reasoning":  "General Reasoning & Problem Solving",
    }

    total_chunks = 0
    print("=" * 55)
    print("  DOCUMENT INGESTION")
    print("=" * 55)

    for path, description in sources.items():
        if os.path.exists(path):
            print(f"\n📁 {description}")
            print(f"   Source: {path}")
            n = rag.ingest(path)
            total_chunks += n
        else:
            print(f"\n⚠️  Folder not found: {path} — skipping")

    print(f"\n{'=' * 55}")
    print(f"  ✅ Done! {total_chunks} total chunks stored in ChromaDB")
    print(f"  Next step: python trainer.py")
    print(f"{'=' * 55}")

if __name__ == "__main__":
    main()
