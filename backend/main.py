from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.agents.graph import graph
from backend.google_books import GoogleBooksAPI
from backend.qdrant_utils import QdrantHandler
from observability.langsmith_config import setup_langsmith

setup_langsmith()

app = FastAPI(title="Book Recommender API")
qdrant = QdrantHandler()

class SearchRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    recommendations: list
    explanations: list

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_books(request: SearchRequest):
    """
    Main endpoint that triggers the LangGraph Agent Workflow
    """
    inputs = {"query": request.query}
    
    # Run the graph
    result = graph.invoke(inputs)
    
    return {
        "recommendations": result.get("recommendations", []),
        "explanations": result.get("explanations", [])
    }

@app.get("/search_google_books")
def search_books_endpoint(query: str):
    """Direct Google Books Search"""
    return GoogleBooksAPI.search_books(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)