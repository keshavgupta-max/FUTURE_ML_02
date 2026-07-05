# Enterprise AI Ticket Routing Engine

An end-to-end, production-grade machine learning microservice built to automate multi-tenant customer support ticket triage. The system preprocesses unstructured text payloads, applies vectorized feature extraction, and concurrently infers both the target organizational department (including high-stakes sectors like *Security & Fraud*) and business urgency tiers. 

---

## Key Architectural Highlights

* **Production-Grade FastAPI Microservice:** Designed with a decoupled architecture pattern, separating core processing logic from presentation layers, complete with strict data schema validation via Pydantic.
* **Dual-Task Machine Learning Pipeline:** Built an internal NLP pipeline leveraging optimized text preprocessing (tokenization, stopword filtering) and serialized Scikit-Learn estimators for parallel multi-class categorization and priority routing.
* **Pure-Python Native Storage Engine:** Built a thread-safe SQLite relational data-access layer utilizing raw, zero-dependency connection drivers to completely eliminate platform-specific binary compilation overhead.
* **Operational Logging & Audit Trails:** Every prediction cycle dynamically generates an audit sequence—capturing multi-tenant identifiers, raw inputs, preprocessed text features, model inference indicators, and human-override hooks.

---

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

Here is the visual mapping of the microservice structure:
![Project Architecture Framework](assets/Screenshot%202026-07-05%20120356.png)

The following screenshot shows successful machine learning inference endpoints running live:
![API Implementation Interface](assets/Screenshot%202026-07-05%20120625.png)