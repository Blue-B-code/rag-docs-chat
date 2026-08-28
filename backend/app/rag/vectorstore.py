"""Qdrant vector store with local fastembed embeddings (no API key needed)."""
import uuid
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


class VectorStore:
    """Ingest documents and search by question, using local embeddings + Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.model = TextEmbedding(model_name=model)
        self._dim: int | None = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self.model.embed(texts, batch_size=32)]

    def _ensure_collection(self) -> None:
        if self._dim is None:
            raise RuntimeError("Vector dimension unknown — ingest a document first")
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self._dim, distance=models.Distance.COSINE
                ),
            )

    def ingest(self, chunks: list[dict[str, str]]) -> int:
        documents = [c["text"] for c in chunks]
        metadata = [{"source": c.get("source", "unknown")} for c in chunks]
        if not documents:
            return 0
        vectors = self._embed(documents)
        self._dim = len(vectors[0])
        self._ensure_collection()
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{metadata[i]}|{documents[i]}")),
                vector=vectors[i],
                payload={"text": documents[i], **metadata[i]},
            )
            for i in range(len(documents))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(documents)

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not question.strip():
            return []
        vector = self._embed([question])[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source"),
                    "score": round(float(hit.score), 4),
                }
            )
        return results
