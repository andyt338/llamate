from llamate.vectorstore import FAISSVectorStore
from llamate.vectorstore_postgres import PostgresVectorStore
from llamate.vectorstore_marqo import MarqoVectorStore
from llamate.embedder import OpenAIEmbedder
from llamate.config import get_vector_backend, get_database_url, get_marqo_url, get_marqo_index_name, get_marqo_model
import os

def get_vectorstore_from_env(user_id: str):
    backend = get_vector_backend()
    
    # Create embedder with configured model
    model = os.environ.get("LLAMATE_EMBEDDING_MODEL", "text-embedding-3-small")
    embedder = OpenAIEmbedder(model=model)

    if backend == "postgres":
        db_url = get_database_url()
        if not db_url:
            raise ValueError("LLAMATE_DATABASE_URL is not set in environment")
        return PostgresVectorStore(db_url=db_url, table=f"memory_{user_id}", embedder=embedder)
    
    elif backend == "marqo":
        marqo_url = get_marqo_url()
        marqo_index = get_marqo_index_name()
        marqo_model = get_marqo_model()
        return MarqoVectorStore(
            url=marqo_url, 
            index_name=marqo_index, 
            model=marqo_model, 
            user_id=user_id
        )

    return FAISSVectorStore(user_id=user_id, embedder=embedder)
