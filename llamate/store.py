from abc import ABC, abstractmethod
from typing import List, Any

class MemoryStore(ABC):
    @abstractmethod
    def add(self, text: str, vector: List[float]):
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[str]:
        pass
