from typing import List, Annotated
from typing_extensions import TypedDict
import operator

class AgentState(TypedDict):
    query: str
    user_intent: str
    search_criteria: dict
    retrieved_books: List[dict]
    recommendations: List[dict]
    explanations: List[str]
    messages: Annotated[List[dict], operator.add]