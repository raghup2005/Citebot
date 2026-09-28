from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

candidates = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]

for name in candidates:
    try:
        llm = ChatGoogleGenerativeAI(model=name, timeout=30, max_retries=0)
        reply = llm.invoke("Say OK")
        print(f"WORKS   {name}")
    except Exception as e:
        print(f"FAILED  {name}: {str(e)[:90]}")