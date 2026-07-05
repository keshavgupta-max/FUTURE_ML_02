# Enterprise AI Ticket Routing Engine

An end-to-end, production-grade machine learning microservice built to automate multi-tenant customer support ticket triage. The system preprocesses unstructured text payloads, applies vectorized feature extraction, and concurrently infers both the target organizational department (including high-stakes sectors like *Security & Fraud*) and business urgency tiers. 

---

## Key Architectural Highlights

* **Production-Grade FastAPI Microservice:** Designed with a decoupled architecture pattern, separating core processing logic from presentation layers, complete with strict data schema validation via Pydantic.
* **Dual-Task Machine Learning Pipeline:** Built an internal NLP pipeline leveraging optimized text preprocessing (tokenization, stopword filtering) and serialized Scikit-Learn estimators for parallel multi-class categorization and priority routing.
* **Pure-Python Native Storage Engine:** Built a thread-safe SQLite relational data-access layer utilizing raw, zero-dependency connection drivers to completely eliminate platform-specific binary compilation overhead.
* **Operational Logging & Audit Trails:** Every prediction cycle dynamically generates an audit sequence—capturing multi-tenant identifiers, raw inputs, preprocessed text features, model inference indicators, and human-override hooks.

---
Visuals of giving input in the framework we created through FastAPI :
<img width="1135" height="518" alt="Screenshot 2026-07-05 120625" src="https://github.com/user-attachments/assets/d9c803fe-15df-4943-9ba6-fea7e003d7d3" />
Visuals of getting output in the framework we created through FastAPI:
<img width="1522" height="742" alt="Screenshot 2026-07-05 120356" src="https://github.com/user-attachments/assets/eabcd8b2-a0ad-4dac-8e9b-1f530b06f74a" />

## Directory Layout

```text
FUTURE_ML_02/
├── backend/
│   ├── __init__.py
│   ├── database.py       # Native SQLite interface & transaction engines
│   ├── main.py           # Core FastAPI application routing & schema logic
│   └── pipeline.py       # Tokenization, text-cleaning, & ML pipeline loaders
├── notebooks/
│   └── eda_and_preprocessing.ipynb  # Model training, validation, & serialization
├── saved_models/         # Serialized production pipeline binaries (.pkl)
├── tickets_history.db    # Dynamic local relational database instance
├── requirements.txt      # Project environment dependencies
└── README.md             # Project documentation


