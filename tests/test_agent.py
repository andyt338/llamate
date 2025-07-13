from llamate.agent import MemoryAgent
from llamate.embedder import OpenAIEmbedder
from llamate.vectorstore_postgres import PostgresVectorStore


def test_chat_response():
    embedder = OpenAIEmbedder()
    store = PostgresVectorStore(user_id="test_user", embedder=embedder)
    store.add("Hello there!", embedder)

    agent = MemoryAgent(user_id="test_user", vectorstore=store, embedder=embedder)
    response = agent.chat("hello")
    assert "hello" in response.lower()
