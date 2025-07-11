import os
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

load_dotenv()
def client_tool():
    return OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))