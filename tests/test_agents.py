import pytest
from backend.agents.state import AgentState
from backend.agents.recommendation_agent import generate_recommendations
from backend.agents.explanation_agent import explain_recommendations
from backend.agents.retriever_agent import retrieve_books
from unittest.mock import MagicMock

def test_recommendation_parsing(mocker):
    mock_chat = mocker.patch("backend.agents.recommendation_agent.ChatOpenAI")
    mock_instance = mock_chat.return_value
    mock_instance.__or__ = lambda self, other: MagicMock(invoke=lambda x: MagicMock(content='[{"title": "Best Book", "book_id": "1"}]'))
    
    state = AgentState(
        query="teste",
        user_intent="recommendation",
        search_criteria={},
        retrieved_books=[{"title": "Best Book", "book_id": "1"}],
        recommendations=[],
        explanations=[],
        messages=[]
    )
    
    result = generate_recommendations(state)
    assert "recommendations" in result

def test_explanation_generation(mocker):
    mock_prompt = mocker.patch("backend.agents.explanation_agent.ChatPromptTemplate")
    mock_llm = mocker.patch("backend.agents.explanation_agent.ChatOpenAI")
    
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Explanation")
    mock_prompt.from_template.return_value.__or__ = lambda self, other: mock_chain
    
    state = AgentState(
        query="python",
        user_intent="recommendation",
        search_criteria={},
        retrieved_books=[],
        recommendations=[{"title": "Book 1"}],
        explanations=[],
        messages=[]
    )
    
    result = explain_recommendations(state)
    assert result["explanations"][0] == "Explanation"

def test_retriever_logic(mocker):
    mock_rag = mocker.patch("backend.agents.retriever_agent.RagPipeline")
    mock_rag.return_value.run_pipeline.return_value = [{"title": "RAG Book", "book_id": "1"}]
    
    mocker.patch("backend.agents.retriever_agent.QdrantHandler")
    mocker.patch("backend.agents.retriever_agent.GoogleBooksAPI.search_books", return_value=[])

    state = AgentState(
        query="test",
        user_intent="search",
        search_criteria={},
        retrieved_books=[],
        recommendations=[],
        explanations=[],
        messages=[]
    )
    
    result = retrieve_books(state)
    assert isinstance(result["retrieved_books"], list)
    assert len(result["retrieved_books"]) == 1