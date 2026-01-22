from unittest.mock import MagicMock
import pytest
from backend.rag_pipeline import RagPipeline

@pytest.fixture
def mock_rag_deps(mocker):
    # Mock QdrantHandler
    mocker.patch("backend.rag_pipeline.QdrantHandler")
    # Mock FlashRanker
    mocker.patch("backend.rag_pipeline.Ranker") 

def test_bm25_search(mock_rag_deps):
    pipeline = RagPipeline()
    docs = [
        {"book_id": "1", "title": "Python Programming", "description": "Code"},
        {"book_id": "2", "title": "Cooking 101", "description": "Food"}
    ]
    
    # Busca por "Python" deve retornar o doc 1 primeiro
    results = pipeline.bm25_search("python", docs)
    assert len(results) > 0
    assert results[0]["book_id"] == "1"

def test_reciprocal_rank_fusion(mock_rag_deps):
    pipeline = RagPipeline()
    list1 = [{"book_id": "A"}, {"book_id": "B"}]
    list2 = [{"book_id": "B"}, {"book_id": "A"}] # Ordem inversa
    
    # Ambos devem aparecer no resultado
    fused = pipeline.reciprocal_rank_fusion(list1, list2)
    assert len(fused) == 2
    ids = [d["book_id"] for d in fused]
    assert "A" in ids and "B" in ids

def test_run_pipeline_flow(mocker):
    # Setup complexo de mocks
    mock_qdrant = mocker.patch("backend.rag_pipeline.QdrantHandler").return_value
    mock_ranker = mocker.patch("backend.rag_pipeline.Ranker").return_value
    
    pipeline = RagPipeline()
    
    # Mock returns
    mock_qdrant.search_dense.return_value = [{"book_id": "1", "title": "Dense Doc"}]
    mock_qdrant.get_all_documents_for_bm25.return_value = [{"book_id": "1", "title": "BM25 Doc"}]
    mock_ranker.rerank.return_value = [{"id": "1", "score": 0.9}]
    
    results = pipeline.run_pipeline("query test")
    
    # Verifica chamadas
    mock_qdrant.search_dense.assert_called()
    mock_ranker.rerank.assert_called()
    assert len(results) == 1
    assert results[0]["book_id"] == "1"