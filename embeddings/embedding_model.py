from langchain_openai import OpenAIEmbeddings
import os

def get_embedding_model():
    """Returns the embedding model instance."""
    # Se estiver em teste sem chave, evita crashar se o teste mockar depois
    if not os.getenv("OPENAI_API_KEY"):
         # Retornar None ou objeto dummy se necessário, ou deixar estourar se for runtime
         pass
    return OpenAIEmbeddings(model="text-embedding-3-small")