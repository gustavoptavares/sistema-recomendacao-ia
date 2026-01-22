import os
from dotenv import load_dotenv

def setup_langsmith():
    load_dotenv()
    # Ensure variables are set, though usually handled by environment integration
    if os.getenv("LANGCHAIN_TRACING_V2") != "true":
        print("Warning: LangSmith tracing is not enabled in .env")