from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    return ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

def explain_recommendations(state):
    print("--- Explanation Agent ---")
    recs = state.get("recommendations", [])
    query = state.get("query", "")
    
    if not recs:
        return {"explanations": ["Nenhuma recomendação disponível para explicar."]}
    
    explanations = []
    
    prompt = ChatPromptTemplate.from_template(
        """Explain succinctly (1 sentence in Portuguese) why this book: '{book_title}' 
        is a good recommendation for a user who asked: '{query}'.
        Focus on the content match.
        """
    )
    
    llm = get_llm()
    chain = prompt | llm
    
    for book in recs:
        try:
            res = chain.invoke({"book_title": book.get('title', 'Livro'), "query": query})
            explanations.append(res.content)
        except Exception as e:
            explanations.append(f"Não foi possível gerar explicação: {e}")
        
    return {"explanations": explanations}