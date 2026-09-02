from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.6-flash",
        temperature = 0,
        timeout = 120,
        max_retries = 1,
        

    )

    return llm

