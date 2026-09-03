import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Create the configured Gemini chat model.

    LLM_MODEL can be changed without editing source code when a model is
    unavailable or its quota has been exhausted.
    """
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        # temperature=0,
        timeout=120,
        max_retries=2,
    )
