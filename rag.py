"""
rag.py - RAG Pipeline
Loads documents, chunks them, stores embeddings in ChromaDB,
and retrieves relevant context for any query.
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


class RAGPipeline:
    def __init__(self, persist_dir="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_dir = persist_dir
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64
        )
        self.store = None

    def ingest(self, source_dir: str):
        """Load .txt files from a folder, chunk and store in ChromaDB."""
        loader = DirectoryLoader(
            source_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            silent_errors=True
        )
        docs = loader.load()
        if not docs:
            print(f"  No .txt files found in {source_dir}")
            return 0

        chunks = self.splitter.split_documents(docs)
        print(f"  {len(docs)} files → {len(chunks)} chunks")

        if self.store is None:
            self.store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.store.add_documents(chunks)

        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> str:
        """Return top-k relevant chunks joined as a single string."""
        if self.store is None:
            # Load existing store from disk
            self.store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
        docs = self.store.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()
    sources = ["docs/domain_qa", "docs/code", "docs/reasoning"]

    print("=== Ingesting all documents ===")
    for source in sources:
        if os.path.exists(source):
            print(f"\nIngesting: {source}")
            rag.ingest(source)

    print("\n=== Test retrieval ===")
    queries = [
        "How do I handle class imbalance in sklearn?",
        "What metrics should I show business stakeholders?",
        "How do I avoid overfitting?"
    ]
    for q in queries:
        context = rag.retrieve(q)
        print(f"\nQ: {q}")
        print(f"Context preview: {context[:200]}...")
