from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

from app.database.db import engine, SessionLocal
from app.database import models
from app.database.models import User, Appointment
from app.auth import hash_password, verify_password, create_access_token, verify_token
from app.rag.rag_service import llm_select_documents

load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://healthcare-ai-frontend-8wy7.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

LLM_MODEL = "llama-3.1-8b-instant"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str


class AppointmentRequest(BaseModel):
    title: str
    appointment_time: str
    recurring: str = "none"


@app.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(
        username=request.username,
        hashed_password=hash_password(request.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat")
async def chat(request: ChatRequest, username: str = Depends(verify_token)):

    user_message = request.message

    context_docs = llm_select_documents(user_message)
    context_text = "\n\n".join(context_docs)

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": f"""
You are a responsible healthcare AI assistant using NHS-based guidance.

Use the following NHS context to answer:

{context_text}

Provide structured educational information only.
If symptoms suggest emergency, clearly advise urgent care.
"""
                },
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3
        }
    )

    result = response.json()

    if "choices" not in result:
        raise HTTPException(status_code=500, detail="LLM response error")

    reply = result["choices"][0]["message"]["content"]

    emergency_keywords = ["chest pain", "stroke", "heart attack", "can't breathe", "severe bleeding"]
    is_emergency = any(word in user_message.lower() for word in emergency_keywords)

    confidence = "High" if len(context_docs) > 0 else "Medium"

    formatted_sources = [
        {"title": "NHS Guidance", "content": doc}
        for doc in context_docs
    ]

    return {
        "response": reply,
        "sources": formatted_sources,
        "confidence": confidence,
        "emergency": is_emergency
    }


@app.post("/appointments")
def create_appointment(
    request: AppointmentRequest,
    username: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    appointment = Appointment(
        title=request.title,
        appointment_time=datetime.fromisoformat(request.appointment_time),
        recurring=request.recurring,
        user_id=user.id
    )

    db.add(appointment)
    db.commit()

    return {"message": "Appointment saved successfully"}


@app.get("/appointments")
def get_appointments(
    username: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    appointments = db.query(Appointment).filter(Appointment.user_id == user.id).all()

    return [
        {
            "id": appt.id,
            "title": appt.title,
            "appointment_time": appt.appointment_time.isoformat(),
            "recurring": appt.recurring
        }
        for appt in appointments
    ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
