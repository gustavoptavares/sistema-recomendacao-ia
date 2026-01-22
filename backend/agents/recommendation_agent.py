import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    return ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

def generate_recommendations(state):
    print("--- Recommendation Agent ---")
    books = state["retrieved_books"]
    user_intent = state["user_intent"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Book Expert. Based on the user's intent: {intent}
        and these retrieved books: {books}
        
        Select the top 3 most relevant books.
        Return ONLY a legitimate JSON list of the selected books (keep original structure).
        """
    )
    
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({"intent": user_intent, "books": json.dumps(books, default=str)})
    
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    try:
        recommendations = json.loads(content.strip())
    except:
        recommendations = books[:3] 
        
    return {"recommendations": recommendations}