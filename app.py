import os
import string
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_ticket_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in string.punctuation and word not in stop_words and word.isalnum()
    ]
    return " ".join(cleaned_tokens)


@st.cache_resource # Keeps models in server RAM memory so predictions are instant
def load_ml_pipeline():
    models_dir = os.path.join(os.getcwd(), 'saved_models')
    v_path = os.path.join(models_dir, 'vectorizer.pkl')
    cat_path = os.path.join(models_dir, 'category_model.pkl')
    prio_path = os.path.join(models_dir, 'priority_model.pkl')
    
    vectorizer = joblib.load(v_path)
    cat_model = joblib.load(cat_path)
    prio_model = joblib.load(prio_path)
    return vectorizer, cat_model, prio_model

try:
    vectorizer, cat_model, prio_model = load_ml_pipeline()
    model_status = " AI Core Engine Online"
except Exception as e:
    model_status = f" Error Loading Core Models: {str(e)}"


st.set_page_config(page_title="AI Support Routing Hub", page_icon="🚨", layout="centered")

st.title(" Smart Support Ticket Routing System")
st.markdown("### Production-Ready Triage Prediction Engine")
st.caption(model_status)
st.write("---")

# User Input Controls
subject_input = st.text_input(" Ticket Subject Line", placeholder="e.g., Account locked out after multiple attempts")
description_input = st.text_area("Detailed Issue Description", placeholder="Paste the user complaint email body here...")

st.write("")

if st.button(" Process & Routing Ticket", use_container_width=True):
    if not subject_input.strip() or not description_input.strip():
        st.warning(" Please provide both a subject line and description.")
    else:
        # 1. Pipeline Text Transformation
        raw_combined = f"{subject_input} {description_input}"
        clean_features = clean_ticket_text(raw_combined)
        
        # 2. Vectorization & Inference
        vectorized_features = vectorizer.transform([clean_features])
        pred_cat = cat_model.predict(vectorized_features)[0]
        pred_prio = prio_model.predict(vectorized_features)[0]
        
        # 3. Present Results
        st.write("###  Automated Triage Conclusion Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"🏷️ **Assigned Department:**\n\n### {pred_cat}")
            
        with col2:
            if pred_prio.lower() in ['high', 'critical']:
                st.error(f" **Urgency Priority:**\n\n### {pred_prio.upper()}")
            elif pred_prio.lower() == 'medium':
                st.warning(f" **Urgency Priority:**\n\n### {pred_prio.upper()}")
            else:
                st.success(f" **Urgency Priority:**\n\n### {pred_prio.upper()}")
                
        st.balloons()