"""Ties all the steps together."""
from .pdf_loader import extract_pages
from .chunker import chunk_pages
from .embedder import Embedder
from .vector_store import VectorStore
from .llm import generate_answer


class RAGPipeline:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.store: VectorStore | None = None

    def ingest(self, pdf_file) -> int:
        """Read the PDF, chunk it, embed it, and index it. Returns number of chunks."""
        pages = extract_pages(pdf_file)
        if not pages:
            raise ValueError("No readable text found. The PDF may be a scanned image.")
        chunks = chunk_pages(pages)
        embeddings = self.embedder.encode([c["text"] for c in chunks])
        self.store = VectorStore(embeddings.shape[1])
        self.store.add(embeddings, chunks)
        return len(chunks)

    def ask(self, question: str, k: int = 4) -> dict:
        """Retrieve the top-k chunks and generate an answer from them."""
        if self.store is None:
            raise RuntimeError("Upload a PDF first.")
        q_emb = self.embedder.encode([question])
        sources = self.store.search(q_emb, k=k)
        answer = generate_answer(question, sources)
        return {"answer": answer, "sources": sources}
