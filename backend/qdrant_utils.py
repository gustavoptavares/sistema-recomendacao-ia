import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings.embedding_model import get_embedding_model
import uuid

class QdrantHandler:
    def __init__(self, use_memory_fallback=True):
        qdrant_url = os.getenv("QDRANT_URL", "")
        api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = None
        self.using_memory = False
        
        if qdrant_url and qdrant_url != ":memory:":
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=api_key if api_key else None,
                    timeout=5
                )
                self.client.get_collections()
                print(f"Connected to Qdrant at {qdrant_url}")
            except Exception as e:
                print(f"Warning: Could not connect to Qdrant at {qdrant_url}: {e}")
                if use_memory_fallback:
                    print("Falling back to in-memory Qdrant")
                    self.client = QdrantClient(":memory:")
                    self.using_memory = True
                else:
                    raise
        else:
            self.client = QdrantClient(":memory:")
            self.using_memory = True
            print("Using in-memory Qdrant")
        
        self.embedding_model = get_embedding_model()
        self.collection_name = "books_collection"
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def store_books(self, books: list):
        if not books:
            return
        
        points = []
        texts = [f"{b['title']} {b.get('description', '')} {' '.join(b.get('authors', []))}" for b in books]
        embeddings = self.embedding_model.embed_documents(texts)

        for i, book in enumerate(books):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, book['book_id'])) 
            points.append(PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload=book
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_dense(self, query: str, limit: int = 10, filter_criteria: dict = None):
        query_vector = self.embedding_model.embed_query(query)
        
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return [point.payload for point in hits.points]

    def get_all_documents_for_bm25(self):
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        return [r.payload for r in records]