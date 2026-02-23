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


# =============================
# DATABASE
# =============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================
# MODELS
# =============================

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


# =============================
# AUTH
# =============================

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


# =============================
# CHAT (HEALTHCARE RESTRICTED)
# =============================

@app.post("/chat")
async def chat(request: ChatRequest, username: str = Depends(verify_token)):

    user_message = request.message.strip()


    if not user_message:
        return {
            "response": "Please enter a healthcare-related question.",
            "sources": [],
            "confidence": "Not Applicable",
            "emergency": False
        }

    healthcare_keywords = [
        "health", "medical", "doctor", "hospital", "symptom",
        "pain", "disease", "condition", "treatment", "medicine",
        "appointment", "nhs", "mental", "therapy",
        "blood", "pressure", "diabetes", "asthma",
        "infection", "injury", "emergency", "fever"
    ]

    # ✅ Use LLM to classify whether question is healthcare-related
    classification_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only with YES or NO. Is this question related to healthcare, medicine, symptoms, treatment, or NHS services?"
                },
                {"role": "user", "content": user_message}
            ],
            "temperature": 0
        }
    )

    classification_result = classification_response.json()

    if "choices" not in classification_result:
        raise HTTPException(status_code=500, detail="Classification error")

    is_healthcare = classification_result["choices"][0]["message"]["content"].strip().upper()

    if is_healthcare != "YES":
        return {
            "response": (
                "I am a healthcare assistant and can only respond to medical or healthcare-related questions.\n\n"
                "Please use a general assistant for non-health topics."
            ),
            "sources": [],
            "confidence": "Not Applicable",
            "emergency": False
        }

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
You are a responsible NHS-based healthcare AI assistant.

RULES:
- Only answer healthcare-related questions.
- Use brief, clear bullet points.
- Keep answers short unless user asks for detailed explanation.
- Do not diagnose or prescribe.
- Escalate serious cases to NHS 111 or 999.

NHS Context:
{context_text}
"""
                },
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3
        }
    )

    result = response.json()

    if "choices" not in result:
        raise HTTPException(status_code=500, detail="LLM error")

    reply = result["choices"][0]["message"]["content"]

    high_risk_keywords = [
        "chest pain", "stroke", "heart attack",
        "unconscious", "severe bleeding",
        "can't breathe", "suicidal", "overdose"
    ]

    is_emergency = any(word in user_message.lower() for word in high_risk_keywords)

    if is_emergency:
        reply += (
            "\n\n⚠️ URGENT: Call 999 immediately if life-threatening.\n"
            "For urgent but non-life-threatening medical help, contact NHS 111."
        )

    confidence = (
        "High (Emergency Identified)"
        if is_emergency
        else "High" if context_docs else "Medium"
    )

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


# =============================
# APPOINTMENTS
# =============================

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


@app.delete("/appointments/{appointment_id}")
def delete_appointment(
    appointment_id: int,
    username: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    appointment = db.query(Appointment).filter(
        Appointment.id == appointment_id,
        Appointment.user_id == user.id
    ).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    db.delete(appointment)
    db.commit()

    return {"message": "Appointment deleted successfully"}


@app.get("/")
def root():
    return {"message": "Healthcare AI Backend Running"}
