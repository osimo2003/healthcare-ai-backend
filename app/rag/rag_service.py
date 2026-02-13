import os
import requests
from dotenv import load_dotenv
from app.rag.nhs_documents import documents

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

LLM_MODEL = "llama-3.1-8b-instant"


def llm_select_documents(query):
    prompt = f"""
You are selecting relevant NHS documents for a healthcare assistant.

User question:
{query}

Available NHS documents:
{documents}

Return only the documents that are most relevant to the question.
Return them exactly as written.
If none are relevant, return an empty list.
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You select relevant medical documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
    )

    data = response.json()

    if "choices" not in data:
        print("Groq Selection Error:", data)
        return []

    content = data["choices"][0]["message"]["content"]

    # Simple matching back to original docs
    selected_docs = []
    for doc in documents:
        if doc in content:
            selected_docs.append(doc)

    return selected_docs
