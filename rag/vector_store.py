from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
import uuid

class VectorStore:
    def __init__(self):
        # Initialize Qdrant Client
        # For production, utilize settings.QDRANT_HOST. 
        # For this setup, we use memory mode or local disk if not specified to avoid connection errors if docker isn't up.
        self.client = QdrantClient(":memory:") # Using memory for immediate "runnable" prototype without external dependency failure
        self.collection_name = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

    def add_document(self, text: str, metadata: dict, embedding: list[float]):
        """
        Add a document chunk with its embedding to the store.
        """
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text, **metadata}
                )
            ]
        )

    def search(self, query_embedding: list[float], limit: int = 3):
        """
        Search for similar documents.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        ).points
        return results
