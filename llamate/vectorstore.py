import faiss
import numpy as np
import os
import uuid
import json

class FAISSVectorStore:
    def __init__(self, user_id, embedding_dim=1536):
        self.user_id = user_id
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.memory_path = f"memory_{user_id}.json"
        self.memory_store = []
        self._load()

    def add(self, text: str, embedder):
        vector = embedder.embed(text)
        self.index.add(np.array([vector]))
        self.memory_store.append({"id": str(uuid.uuid4()), "text": text, "vector": vector.tolist()})
        self._save()

    def search(self, query: str, top_k=3):
        if not self.memory_store:
            return []
        query_vector = self._embed_query(query)
        D, I = self.index.search(np.array([query_vector]), top_k)
        return [self.memory_store[i]["text"] for i in I[0] if i < len(self.memory_store)]

    def _embed_query(self, query: str):
        # Use the latest embedder on demand (cached by agent)
        from memgpt.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        return embedder.embed(query)

    def _save(self):
        with open(self.memory_path, "w") as f:
            json.dump(self.memory_store, f)

    def _load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory_store = json.load(f)
                vectors = [np.array(m["vector"]).astype("float32") for m in self.memory_store]
                if vectors:
                    self.index.add(np.array(vectors))
