from llamate.vectorstore_postgres import PostgresVectorStore
from llamate.embedder import OpenAIEmbedder


def test_add_and_search():
    embedder = OpenAIEmbedder()
    store = PostgresVectorStore(user_id="test_user", embedder=embedder)

    store.add("The sky is blue.", embedder)
    results = store.search("What color is the sky?", top_k=1)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert "sky" in results[0].lower()
