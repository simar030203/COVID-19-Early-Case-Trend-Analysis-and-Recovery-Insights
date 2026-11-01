import streamlit as st
import pandas as pd
from utils.preprocess import load_dataset
# CHANGE THIS LINE:
from utils.helper import load_css, page_header, sidebar, footer
# -----------------------------
# Configuration & Layout
# -----------------------------
st.set_page_config(
    page_title="COVID-19 AI Analytics System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load layout utilities
load_css()
sidebar()

# Page Routing
page = st.sidebar.radio(
    "Go To Workspace Tab", 
    ["🏠 Home Workstation", "📊 Analytics Dashboard", "🤖 Predictive Module", "🧠 AI Insights Engine"]
)

# Load shared cached dataset
try:
    df = load_dataset()
except Exception:
    st.error("⚠️ 'patient.csv' dataset file not found in directory!")
    st.stop()

# -----------------------------
# Workspace Rendering
# -----------------------------
if page == "🏠 Home Workstation":
    page_header("🦠 COVID-19 AI Analytics Dashboard", "HealthGuard Analytics Pvt. Ltd.")
    
    st.header("📌 Project Overview")
    st.write("""
    This platform processes early-stage patient documentation using Machine Learning and GenAI pipelines 
    to empower local public health administrators and hospital coordination cells.
    """)
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📊 Workstation Dashboard\nExplore structural data metrics, country-wise patterns, and outcome timelines.")
    with col2:
        st.success("### 🤖 AI Prognosis Engine\nEvaluate recovery risk matrices safely with zero data leakage and dynamically draft strategy reports.")

elif page == "📊 Analytics Dashboard":
    from pages_runtime.dashboard import render_dashboard
    render_dashboard(df)

elif page == "🤖 Predictive Module":
    from pages_runtime.prediction import render_prediction
    render_prediction()

elif page == "🧠 AI Insights Engine":
    from pages_runtime.ai_insights import render_insights
    render_insights(df)

footer()