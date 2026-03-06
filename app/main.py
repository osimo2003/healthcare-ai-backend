from app.database.models import PushSubscription
from fastapi import FastAPI, Depends, HTTPException
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
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

# =============================
# RATE LIMITER SETUP
# =============================
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://healthlink-access-enterprise-frontend.onrender.com",
        "https://healthcare-ai-frontend-8wy7.onrender.com"
    ],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

LLM_MODEL = "llama-3.1-8b-instant"

# Timeout for all Groq API calls (seconds)
GROQ_TIMEOUT = 15


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
# HELPER: Resolve user from token or raise 404
# =============================

def get_user_or_404(username: str, db: Session) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


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



class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str

class PushSubscriptionRequest(BaseModel):
    endpoint: str
    keys: PushSubscriptionKeys


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
@limiter.limit("20/minute")  
async def chat(
    request_obj: Request,
    request: ChatRequest,
    username: str = Depends(verify_token)
):
    user_message = request.message.strip()

    if not user_message:
        return {
            "response": "Please enter a healthcare-related question.",
            "sources": [],
            "confidence": "Not Applicable",
            "emergency": False
        }


    #LLM classification
    try:
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
            },
            timeout=GROQ_TIMEOUT 
        )
        classification_result = classification_response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Classification service timed out. Please try again.")
    except Exception:
        raise HTTPException(status_code=500, detail="Classification service unavailable.")

    if "choices" not in classification_result:
        raise HTTPException(status_code=500, detail="Classification error")

    
    is_healthcare_raw = classification_result["choices"][0]["message"]["content"].strip().upper()
    is_healthcare = is_healthcare_raw.startswith("YES")

    if not is_healthcare:
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

    #Main LLM response call
    try:
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
- If the situation sounds like an emergency, include the word EMERGENCY in your response.

NHS Context:
{context_text}
"""
                    },
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.3
            },
            timeout=GROQ_TIMEOUT  
        )
        result = response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Response service timed out. Please try again.")
    except Exception:
        raise HTTPException(status_code=500, detail="Response service unavailable.")

    if "choices" not in result:
        raise HTTPException(status_code=500, detail="LLM error")

    reply = result["choices"][0]["message"]["content"]

    high_risk_keywords = [
        "chest pain", "stroke", "heart attack",
        "unconscious", "severe bleeding",
        "can't breathe", "suicidal", "overdose"
    ]

    combined_check = user_message.lower() + " " + reply.lower()
    is_emergency = any(word in combined_check for word in high_risk_keywords) or "emergency" in reply.lower()

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
   
    user = get_user_or_404(username, db)

    
    try:
        parsed_time = datetime.fromisoformat(request.appointment_time)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid appointment_time format. Use ISO 8601 e.g. 2025-06-01T10:00:00"
        )

    appointment = Appointment(
        title=request.title,
        appointment_time=parsed_time,
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
    
    user = get_user_or_404(username, db)

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
   
    user = get_user_or_404(username, db)

    appointment = db.query(Appointment).filter(
        Appointment.id == appointment_id,
        Appointment.user_id == user.id
    ).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    db.delete(appointment)
    db.commit()

    return {"message": "Appointment deleted successfully"}


# =============================
# PUSH SUBSCRIPTION
# =============================

@app.post("/subscribe")
def subscribe(
    subscription: PushSubscriptionRequest, 
    username: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    
    user = get_user_or_404(username, db)

    existing = db.query(PushSubscription).filter(
        PushSubscription.endpoint == subscription.endpoint
    ).first()

    if existing:
        return {"message": "Subscription already exists"}

    new_subscription = PushSubscription(
        endpoint=subscription.endpoint,
        p256dh=subscription.keys.p256dh,
        auth=subscription.keys.auth,
        user_id=user.id
    )

    db.add(new_subscription)
    db.commit()

    return {"message": "Subscription saved successfully"}


# =============================
# ROOT
# =============================

@app.get("/")
def root():
    return {"message": "Healthcare AI Backend Running"}
