from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.intent_agent import analyze_intent
from backend.agents.retriever_agent import retrieve_books
from backend.agents.recommendation_agent import generate_recommendations
from backend.agents.explanation_agent import explain_recommendations

# Initialize Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intent_analyzer", analyze_intent)
workflow.add_node("retriever", retrieve_books)
workflow.add_node("recommender", generate_recommendations)
workflow.add_node("explainer", explain_recommendations)

# Define Edges
workflow.set_entry_point("intent_analyzer")
workflow.add_edge("intent_analyzer", "retriever")
workflow.add_edge("retriever", "recommender")
workflow.add_edge("recommender", "explainer")
workflow.add_edge("explainer", END)

# Compile
graph = workflow.compile()