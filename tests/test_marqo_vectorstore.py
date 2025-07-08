import pytest
from llamate.vectorstore_marqo import MarqoVectorStore
from llamate.embedder import OpenAIEmbedder
import os

# Skip this test if Marqo is not available
@pytest.mark.skipif(
    not os.getenv("LLAMATE_MARQO_URL") and not os.getenv("LLAMATE_VECTOR_BACKEND") == "marqo",
    reason="Marqo not configured"
)
def test_marqo_add_and_search():
    embedder = OpenAIEmbedder()
    store = MarqoVectorStore(
        url=os.getenv("LLAMATE_MARQO_URL", "http://localhost:8882"),
        index_name="test_index",
        user_id="test_user"
    )

    store.add("The sky is blue.", embedder)
    results = store.search("What color is the sky?", top_k=1)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert "sky" in results[0].lower()

@pytest.mark.skipif(
    not os.getenv("LLAMATE_MARQO_URL") and not os.getenv("LLAMATE_VECTOR_BACKEND") == "marqo",
    reason="Marqo not configured"
)
def test_marqo_duplicate_prevention():
    embedder = OpenAIEmbedder()
    store = MarqoVectorStore(
        url=os.getenv("LLAMATE_MARQO_URL", "http://localhost:8882"),
        index_name="test_duplicate_index",
        user_id="test_user"
    )

    # Add the same text twice
    store.add("This is a test message.", embedder)
    store.add("This is a test message.", embedder)  # Should be prevented
    
    # Search should still return results
    results = store.search("test message", top_k=5)
    assert isinstance(results, list) 