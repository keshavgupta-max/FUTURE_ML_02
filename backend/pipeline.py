import os
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Quietly download necessary asset models inside runtime environments
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

def load_ml_pipeline():
    # Looks up one level to find the saved_models folder relative to backend directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'saved_models')
    
    vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
    cat_model = joblib.load(os.path.join(models_dir, 'category_model.pkl'))
    prio_model = joblib.load(os.path.join(models_dir, 'priority_model.pkl'))
    return vectorizer, cat_model, prio_model