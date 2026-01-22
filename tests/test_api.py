from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("backend.main.graph.invoke")
def test_recommend_endpoint(mock_graph_invoke):
    # Mock do retorno do grafo
    mock_graph_invoke.return_value = {
        "recommendations": [{"title": "Mock Book"}],
        "explanations": ["Mock Explanation"]
    }
    
    payload = {"query": "livros sobre testes"}
    response = client.post("/recommend", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["title"] == "Mock Book"
    assert data["explanations"][0] == "Mock Explanation"