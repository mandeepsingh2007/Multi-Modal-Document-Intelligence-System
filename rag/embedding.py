from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        # We use OpenAI Embeddings as the production standard
        # Ensure OPENAI_API_KEY is in env
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def get_embedding(self, text: str) -> list[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Embedding Error: {e}")
            return [0.0] * 1536 # Return zero vector on failure to prevent crash
