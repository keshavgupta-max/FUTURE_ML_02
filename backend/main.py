from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.pipeline import clean_ticket_text, load_ml_pipeline
from backend.database import init_db, save_ticket_to_db

# Initialize application layers
app = FastAPI(
    title="Enterprise AI Ticket Routing Engine",
    description="Production-grade API for automated multi-tenant ticket classification",
    version="2.0.0"
)

# Generate database file instantly out-of-the-box
init_db()

# Load models
try:
    vectorizer, cat_model, prio_model = load_ml_pipeline()
    print("Core machine learning artifacts successfully loaded into memory.")
except Exception as e:
    print(f"Critical initialization error: {str(e)}")
    vectorizer, cat_model, prio_model = None, None, None

class TicketInput(BaseModel):
    subject: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    tenant_id: str = Field("default_tenant")

class TicketPredictionResponse(BaseModel):
    id: int
    tenant_id: str
    cleaned_features: str
    assigned_department: str
    urgency_priority: str

@app.post("/predict", response_model=TicketPredictionResponse)
def predict_ticket(ticket: TicketInput):
    if not vectorizer or not cat_model or not prio_model:
        raise HTTPException(status_code=500, detail="Machine Learning models are unavailable.")
    
    try:
        raw_text = f"{ticket.subject} {ticket.description}"
        processed_text = clean_ticket_text(raw_text)
        
        if not processed_text.strip():
            processed_text = "default empty ticket text"
            
        features_vectorized = vectorizer.transform([processed_text])
        predicted_department = cat_model.predict(features_vectorized)[0]
        predicted_urgency = prio_model.predict(features_vectorized)[0]
        
        # 💾 SAVE TO DATABASE: Native execution block without SQLAlchemy dependency
        record_id = save_ticket_to_db(
            tenant_id=ticket.tenant_id,
            subject=ticket.subject,
            description=ticket.description,
            cleaned_text=processed_text,
            predicted_dept=predicted_department,
            predicted_prio=predicted_urgency
        )
        
        return TicketPredictionResponse(
            id=record_id,
            tenant_id=ticket.tenant_id,
            cleaned_features=processed_text,
            assigned_department=predicted_department,
            urgency_priority=predicted_urgency
        )
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(error)}")