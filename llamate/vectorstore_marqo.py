import marqo
import numpy as np
from llamate.store import MemoryStore
from typing import List
import os
import uuid

class MarqoVectorStore(MemoryStore):
    def __init__(self, url: str = "http://localhost:8882", index_name: str = "llamate", 
                 model: str = "hf/e5-base-v2", user_id: str = None):
        self.client = marqo.Client(url=url)
        self.index_name = f"{index_name}_{user_id}" if user_id else index_name
        self.model = model
        self.user_id = user_id
        
        # Create index if it doesn't exist
        try:
            self.client.create_index(self.index_name, model=self.model)
        except marqo.errors.MarqoWebError as e:
            if "IndexAlreadyExists" not in str(e):
                raise

    def add(self, text: str, vector_or_embedder):
        # Handle both vector and embedder cases for consistency with other stores
        from llamate.embedder import OpenAIEmbedder
        if isinstance(vector_or_embedder, OpenAIEmbedder):
            vector = vector_or_embedder.embed(text)
        else:
            vector = vector_or_embedder
            
        # Check for similar existing memories to avoid duplicates
        # Search for similar documents first
        try:
            results = self.client.index(self.index_name).search(
                q=text, 
                limit=1,
                search_method="TENSOR"
            )
            
            # If we found a close match and the score is high enough, skip adding
            if results.get("hits") and len(results["hits"]) > 0:
                top_score = results["hits"][0].get("_score", 0)
                if top_score > 0.9:  # High score = very similar
                    return
        except Exception:
            # If search fails, continue with adding
            pass
            
        # Add the document to Marqo
        doc_id = str(uuid.uuid4())
        self.client.index(self.index_name).add_documents(
            [{"_id": doc_id, "text": text, "vector": vector.tolist()}],
            tensor_fields=["text"]
        )

    def search(self, query: str, top_k: int = 3) -> List[str]:
        try:
            results = self.client.index(self.index_name).search(
                q=query,
                limit=top_k,
                search_method="TENSOR"
            )
            
            # Extract text from results
            hits = results.get("hits", [])
            return [hit.get("text", "") for hit in hits if hit.get("text")]
        except Exception as e:
            print(f"Error searching Marqo: {e}")
            return [] 