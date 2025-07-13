from llamate.embedder import OpenAIEmbedder


def test_embedder_vector_shape():
    embedder = OpenAIEmbedder()
    vector = embedder.embed("test input")
    assert len(vector) == 1536  # Check length instead of shape
    assert isinstance(vector[0], float)  # Check it's a float
