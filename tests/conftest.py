import os
import sys
import pytest
from unittest.mock import MagicMock

# Garante que o diretório raiz está no path (redundância de segurança)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Define variáveis de ambiente fictícias para todos os testes."""
    os.environ["OPENAI_API_KEY"] = "sk-fake-key"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["QDRANT_API_KEY"] = "fake-key"
    os.environ["GOOGLE_BOOKS_API_KEY"] = "fake-google-key"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = "ls-fake"

@pytest.fixture
def mock_qdrant_client(mocker):
    """Mock completo do cliente Qdrant."""
    mock = mocker.patch("backend.qdrant_utils.QdrantClient")
    client_instance = MagicMock()
    mock.return_value = client_instance
    
    # Mock para métodos comuns
    client_instance.get_collections.return_value.collections = []
    client_instance.search.return_value = []
    client_instance.upsert.return_value = True
    
    return client_instance

@pytest.fixture
def mock_embedding_model(mocker):
    """Mock do modelo de embeddings (evita chamadas OpenAI)."""
    # Importante: Mockamos onde ele é IMPORTADO (qdrant_utils), e não apenas onde é definido
    mock = mocker.patch("backend.qdrant_utils.get_embedding_model")
    model_instance = MagicMock()
    
    # Simula vetores de tamanho 3
    model_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    model_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    
    mock.return_value = model_instance
    return model_instance