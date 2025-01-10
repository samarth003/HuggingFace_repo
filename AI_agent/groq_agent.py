from phi.agent import Agent 
from phi.model.groq import Groq 
from dotenv import load_dotenv

import os 

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("Error: API Key not found in the .env file")
else:
    print(f"Loaded API key: {API_KEY}")

ai_agent = Agent(model=Groq(id="llama-3.3-70b-versatile"))

ai_agent.print_response("Explain the importance of fast language models?")
