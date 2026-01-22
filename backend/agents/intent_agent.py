import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_llm():
    # Inicializa apenas quando chamado, permitindo mocks e evitando erro de import
    if not os.getenv("OPENAI_API_KEY"):
         # Retorna um dummy silentemente em testes se a chave nao existir, 
         # ou deixa estourar apenas na execução real
         pass
    return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

def analyze_intent(state):
    print("--- Intent Agent ---")
    query = state["query"]
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze the user query about books.
        Return a JSON with:
        - "intent": "search" or "recommendation"
        - "keywords": list of string keywords
        - "filters": dictionary with potential filters (year, author, category) if explicitly mentioned.
        
        Query: {query}
        """
    )
    
    llm = get_llm()
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"query": query})
    
    search_criteria = result
    
    return {"user_intent": result["intent"], "search_criteria": search_criteria}