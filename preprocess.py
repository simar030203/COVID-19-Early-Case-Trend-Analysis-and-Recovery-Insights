import pandas as pd
import joblib
import streamlit as st
import os
import numpy as np

# =====================================================
# Load Dataset Safely
# =====================================================
@st.cache_data
def load_dataset(path="patient.csv"):
    """Loads and auto-cleans dataset files for runtime operations."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data tracking layout asset: {path}")
        
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    cleaned = [l for l in lines if not (l.startswith("<<<") or l.startswith("===") or l.startswith(">>>"))]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(cleaned)
        
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Clean whitespace metrics safely
    return df

# =====================================================
# Load Unified Production Pipeline
# =====================================================
@st.cache_resource
def load_production_pipeline():
    """Loads the compiled pipeline containing preprocessing and the classifier."""
    return joblib.load("covid_prediction_model.pkl")

# =====================================================
# Prediction Execution
# =====================================================
def predict_patient(country, confirmed_year, confirmed_month, confirmed_day):
    """
    Directly passes raw UI attributes into the pipeline.
    Eliminates manual label encoding, scaling, and leakage steps.
    """
    pipeline = load_production_pipeline()
    
    # Construct a dataframe matching the features the model was trained on
    input_data = pd.DataFrame({
        "sex": ["unknown"], # Handled safely via pipeline's most_frequent imputer
        "birth_year": [np.nan], # Handled safely via pipeline's median imputer
        "country": [country],
        "region": ["unknown"],
        "group": [np.nan],
        "infection_reason": ["unknown"],
        "contact_number": [np.nan],
        "confirmed_month": [int(confirmed_month)],
        "confirmed_day": [int(confirmed_day)]
    })
    
    # Run prediction
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    
    # Map back to state labels
    label = "Released" if prediction == 1 else "Isolated / Deceased"
    confidence = probabilities[prediction] * 100
    
    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "probability": probabilities
    }