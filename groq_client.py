# groq_client.py
"""
Groq/LangChain client initialization.
Sits separate from pipeline so other modules can import `llm`.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # load .env if present

# You must have these packages installed: langchain-openai, langchain-core
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
except Exception as e:
    raise RuntimeError("Missing LangChain/Groq libs. Install langchain-openai and langchain-core.") from e

# Read API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Put it in .env or set environment variable.")

# Initialize ChatOpenAI adapter pointed at Groq API
# model_name can be changed to the model your account supports
llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model_name="llama3-70b-8192",
    temperature=0.0,
)

# Export llm for other modules
__all__ = ["llm", "ChatPromptTemplate", "JsonOutputParser", "StrOutputParser"]
