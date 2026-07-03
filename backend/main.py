from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.pipeline import clean_ticket_text, load_ml_pipeline

# Initialize the core FastAPI engine application
app = FastAPI(
    title="Enterprise AI Ticket Routing Engine",
    description="Production-grade API for automated multi-tenant ticket classification and prioritization",
    version="2.0.0"
)

# Load the freshly trained models into memory at startup
try:
    vectorizer, cat_model, prio_model = load_ml_pipeline()
    print("Core machine learning artifacts successfully loaded into memory.")
except Exception as e:
    print(f"Critical initialization error: Failed to load models: {str(e)}")
    vectorizer, cat_model, prio_model = None, None, None

# Define strict input schema for incoming customer data requests
class TicketInput(BaseModel):
    subject: str = Field(..., min_length=1, description="The subject line of the customer ticket")
    description: str = Field(..., min_length=1, description="The detailed body text of the ticket issue")
    tenant_id: str = Field("default_tenant", description="Identifier to isolate data between organizations")

# Define strict output schema structure for predictions
class TicketPredictionResponse(BaseModel):
    tenant_id: str
    cleaned_features: str
    assigned_department: str
    urgency_priority: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Enterprise Support Triage API Engine is active and listening."
    }

@app.post("/predict", response_model=TicketPredictionResponse)
def predict_ticket(ticket: TicketInput):
    # Safety check to ensure models loaded properly at startup
    if not vectorizer or not cat_model or not prio_model:
        raise HTTPException(
            status_code=500, 
            detail="Machine Learning models are unavailable or corrupted on the server backend."
        )
    
    try:
        # 1. Combine inputs and run text preprocessing steps through the pipeline
        raw_text = f"{ticket.subject} {ticket.description}"
        processed_text = clean_ticket_text(raw_text)
        
        # Guardrail check in case cleaning completely empties out a junk input string
        if not processed_text.strip():
            processed_text = "default empty ticket text"
            
        # 2. Extract feature matrix values using the fitted TF-IDF parameters
        features_vectorized = vectorizer.transform([processed_text])
        
        # 3. Generate inference conclusions from both specialized models
        predicted_department = cat_model.predict(features_vectorized)[0]
        predicted_urgency = prio_model.predict(features_vectorized)[0]
        
        # 4. Construct and return structured response payload matching the schema
        return TicketPredictionResponse(
            tenant_id=ticket.tenant_id,
            cleaned_features=processed_text,
            assigned_department=predicted_department,
            urgency_priority=predicted_urgency
        )
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(error)}")