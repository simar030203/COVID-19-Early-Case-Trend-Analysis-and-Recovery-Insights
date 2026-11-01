import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from utils.helper import page_header  # Safely imports the page header from helper.py

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Insights Workspace",
    page_icon="🧠",
    layout="wide"
)

page_header("🧠 AI Powered COVID-19 Insights", "Strategic public health report generation workstation.")

# -------------------------------------------------
# Gemini API Setup
# -------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ Gemini API Key not found! Please check your `.streamlit/secrets.toml` file.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# FIX: Initializing with the exact verified model name string
model = genai.GenerativeModel("gemini-3.5-flash")

# -------------------------------------------------
# Safe Load Dataset (Sanitizes Git markers & whitespace)
# -------------------------------------------------
@st.cache_data
def load_data():
    path = "patient.csv"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Clean any accidental Git markers if they reappear
        cleaned = [l for l in lines if not (l.startswith("<<<") or l.startswith("===") or l.startswith(">>>"))]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(cleaned)
    return pd.read_csv(path)

df = load_data()
df.columns = df.columns.str.strip()  # Strip hidden whitespace characters

# -------------------------------------------------
# Dataset Overview
# -------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# -------------------------------------------------
# Dataset Summary Cards
# -------------------------------------------------
st.subheader("Dataset Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])

with col3:
    st.metric("Missing Data Points", int(df.isnull().sum().sum()))

with col4:
    if "country" in df.columns:
        st.metric("Unique Countries", df["country"].nunique())
    else:
        st.metric("Unique Countries", "N/A")

st.markdown("---")

# -------------------------------------------------
# Context Generation for Prompt Engineering
# -------------------------------------------------
state_summary = df['state'].value_counts().to_string() if 'state' in df.columns else 'N/A'
country_summary = df['country'].value_counts().head(20).to_string() if 'country' in df.columns else 'N/A'

summary_context = f"""
Dataset Summary Context:
- Rows: {df.shape[0]} | Columns: {df.shape[1]}
- Features List: {list(df.columns)}
- Missing Values Log:\n{df.isnull().sum().to_string()}
- Patient Outcomes Summary:\n{state_summary}
- Country-level Breakdown:\n{country_summary}
"""

# -------------------------------------------------
# Report Generator Execution
# -------------------------------------------------
st.subheader("📋 Executive Strategic Planner")
if st.button("🚀 Draft Automated Public Health Strategy Report", use_container_width=True):
    with st.spinner("Analyzing operational trends and compiling executive brief..."):
        prompt = f"""
        You are a Senior Healthcare Data Scientist analyzing early-stage infectious disease case data.
        Generate a comprehensive markdown report with the following exact sections:
        1. Executive Summary
        2. Key Transmission Findings
        3. Regional Risk Assessment
        4. Operational Insights & Actionable Recommendations
        5. Conclusion
        
        Dataset Context:
        {summary_context}
        """
        try:
            response = model.generate_content(prompt)
            st.success("Strategic Document Compiled Successfully!")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"AI Generation Failed: {e}")

st.markdown("---")

# -------------------------------------------------
# Iterative Chat Workstation
# -------------------------------------------------
st.header("💬 Interactive AI Analyst Desk")
question = st.text_input("Ask a professional question about patterns, demographics, or sources:")

if st.button("Ask AI Workspace", key="chat_btn"):
    if question.strip() != "":
        with st.spinner("Querying contextual data arrays..."):
            prompt = f"Contextual Dataset Information:\n{summary_context}\n\nUser Question: {question}\nAnswer professionally as an epidemiologist."
            try:
                response = model.generate_content(prompt)
                st.info("Response Analysis:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

st.success("AI Workflow Completed Successfully")