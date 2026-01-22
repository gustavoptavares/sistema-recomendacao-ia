import pytest
from backend.qdrant_utils import QdrantHandler
from qdrant_client.models import PointStruct
from unittest.mock import MagicMock

def test_store_books(mock_qdrant_client, mock_embedding_model):
    handler = QdrantHandler()
    
    books = [{
        "book_id": "test_id",
        "title": "Test Title",
        "description": "Test Desc",
        "authors": ["Author"],
        "categories": [],
        "published_year": 2020,
        "thumbnail": ""
    }]
    
    handler.store_books(books)
    
    mock_embedding_model.embed_documents.assert_called_once()
    mock_qdrant_client.upsert.assert_called_once()

def test_search_dense(mock_qdrant_client, mock_embedding_model):
    handler = QdrantHandler()
    
    mock_result = MagicMock()
    mock_result.points = [MagicMock(payload={"title": "Hit Book"})]
    mock_qdrant_client.query_points.return_value = mock_result
    
    results = handler.search_dense("query")
    
    mock_embedding_model.embed_query.assert_called_with("query")
    mock_qdrant_client.query_points.assert_called_once()
    assert len(results) == 1
    assert results[0]["title"] == "Hit Book"