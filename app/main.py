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

# ✅ CORS
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
# DATABASE DEPENDENCY
# =============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================
# REQUEST MODELS
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
# AUTH ENDPOINTS
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
# CHAT (HEALTHCARE ONLY)
# =============================

@app.post("/chat")
async def chat(request: ChatRequest, username: str = Depends(verify_token)):

    user_message = request.message

    # ✅ Strict Healthcare Restriction
    healthcare_keywords = [
        "health", "medical", "doctor", "hospital", "symptom",
        "pain", "disease", "condition", "treatment", "medicine",
        "appointment", "nhs", "mental health", "therapy",
        "blood", "pressure", "diabetes", "asthma",
        "infection", "injury", "emergency", "fever",
        "headache", "breathing", "heart", "stroke",
        "anxiety", "depression", "injury"
    ]

    if not any(keyword in user_message.lower() for keyword in healthcare_keywords):
        return {
            "response": (
                "I am a healthcare assistant and can only respond to medical or healthcare-related questions.\n\n"
                "For non-health-related inquiries, please use a general-purpose assistant or search engine."
            ),
            "sources": [],
            "confidence": "Not Applicable",
            "emergency": False
        }

    # ✅ Retrieve NHS Context
    context_docs = llm_select_documents(user_message)
    context_text = "\n\n".join(context_docs)

    # ✅ Detect if user wants detailed explanation
    detailed_request = any(
        phrase in user_message.lower()
        for phrase in ["explain in detail", "more detail", "full explanation", "elaborate"]
    )

    # ✅ LLM Call
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

STRICT RULES:
- Only answer healthcare-related questions.
- If a question is not healthcare-related, politely refuse.
- Provide educational information only.
- Do not diagnose.
- Do not prescribe medication.
- For serious symptoms, advise contacting NHS 111 or emergency services.

RESPONSE STYLE:
- Always give clear, brief, simple answers.
- Use clean bullet points.
- Avoid long paragraphs.
- Only give detailed explanations if the user explicitly asks for more detail.

Use the following NHS context if relevant:

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
        raise HTTPException(status_code=500, detail="LLM response error")

    reply = result["choices"][0]["message"]["content"]

    # ✅ Advanced Emergency Detection
    high_risk_keywords = [
        "chest pain", "stroke", "heart attack",
        "unconscious", "severe bleeding",
        "can't breathe", "not breathing",
        "suicidal", "overdose", "seizure",
        "collapse", "paralysis"
    ]

    is_emergency = any(word in user_message.lower() for word in high_risk_keywords)

    if is_emergency:
        reply += (
            "\n\n⚠️ URGENT: If this is life-threatening, call 999 immediately.\n"
            "For urgent but non-life-threatening medical concerns, contact NHS 111 for advice."
        )

    # ✅ Confidence Logic
    if is_emergency:
        confidence = "High (Emergency Identified)"
    elif len(context_docs) > 0:
        confidence = "High"
    else:
        confidence = "Medium"

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
# APPOINTMENT SYSTEM
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


# =============================
# ROOT ROUTE
# =============================

@app.get("/")
def root():
    return {"message": "Healthcare AI Backend is running successfully."}
