from backend.google_books import GoogleBooksAPI

def retrieve_books(state):
    print("--- Retriever Agent ---")
    
    query = state["query"]
    criteria = state.get("search_criteria", {})
    keywords = " ".join(criteria.get("keywords", [query]))
    
    new_books = GoogleBooksAPI.search_books(keywords)
    
    if new_books:
        try:
            from backend.rag_pipeline import RagPipeline
            rag_pipeline = RagPipeline()
            from backend.qdrant_utils import QdrantHandler
            qdrant_handler = QdrantHandler()
            qdrant_handler.store_books(new_books)
            final_books = rag_pipeline.run_pipeline(keywords, top_k=5)
            if final_books:
                return {"retrieved_books": final_books}
        except Exception as e:
            print(f"RAG pipeline error (using Google Books directly): {e}")
    
    return {"retrieved_books": new_books[:5] if new_books else []}