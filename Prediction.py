import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Safe framework utility imports
from utils.preprocess import predict_patient, load_production_pipeline
from utils.helper import page_header

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="COVID-19 Prediction",
    page_icon="🦠",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>
.main{
    background:#f5f7fa;
}
.block-container{
    padding-top:2rem;
}
.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}
.metric-card{
    border-radius:15px;
    padding:20px;
    background:white;
    box-shadow:0px 0px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD RESOURCES (Unified Pipeline Engine)
# ==========================================================
@st.cache_resource
def load_resources():
    # The updated model incorporates preprocessing, imputing, and standard scaling internally
    pipeline = load_production_pipeline()
    return pipeline

try:
    pipeline = load_resources()
except Exception as e:
    st.error("⚠️ Prediction assets missing! Please make sure to run 'train_pipeline.py' first.")
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
        width=120
    )
    st.title("COVID-19 AI")
    st.info("""
Prediction Module
• Machine Learning
• AI Analytics
• Risk Assessment
• Confidence Score
""")
    st.divider()
    st.write("Production Model Pipeline Status:")
    st.success("Active / Ready")

# ==========================================================
# HEADER
# ==========================================================
page_header("🦠 COVID-19 Patient State Prediction", "Predict the patient's probable recovery trajectory using safe, optimized ML model pipelines.")

# ==========================================================
# USER INPUT
# ==========================================================
st.subheader("Patient Information Profile")

left, right = st.columns(2)

with left:
    # Extracted unique variations based on patient records
    country = st.selectbox(
        "Patient Location Country",
        ["Korea", "China", "Other"]
    )

    confirmed_year = st.number_input(
        "Confirmed Year",
        min_value=2020,
        max_value=2030,
        value=2020
    )

with right:
    confirmed_month = st.selectbox(
        "Confirmed Month",
        list(range(1, 13)),
        index=2 # Defaults to March
    )

    confirmed_day = st.selectbox(
        "Confirmed Day",
        list(range(1, 32)),
        index=14 # Defaults to 15th
    )

st.divider()

# ==========================================================
# SHOW INPUT SUMMARY
# ==========================================================
summary = pd.DataFrame({
    "Clinical Attribute Feature": [
        "Patient Country Location",
        "Case Confirmed Year",
        "Case Confirmed Month",
        "Case Confirmed Day"
    ],
    "Input Parameter Value": [
        country,
        confirmed_year,
        confirmed_month,
        confirmed_day
    ]
})

st.subheader("Configured Attribute Input Summary")
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# PREDICT BUTTON EXECUTION
# ==========================================================
predict_btn = st.button(
    "🚀 Evaluate Target Recovery Odds",
    use_container_width=True
)

if predict_btn:
    with st.spinner("Processing clinical data attributes through model pipeline arrays..."):
        try:
            # Execute predictions safely with no leaky data components
            result = predict_patient(
                country=country,
                confirmed_year=confirmed_year,
                confirmed_month=confirmed_month,
                confirmed_day=confirmed_day
            )

            prediction = result["prediction"]
            confidence = result["confidence"]
            probability = result["probability"]

            st.success("Risk Analysis Calculations Completed Successfully!")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Predicted Recovery Track Resolution", prediction)
            with c2:
                st.metric("Model Certainty Score Index", f"{confidence:.2f}%")

            st.divider()

            # ==========================================================
            # PROBABILITY CHART
            # ==========================================================
            probability_df = pd.DataFrame({
                "Resolution Path": ["Isolated / Deceased", "Released (Recovered)"],
                "Odds Probability (%)": probability * 100
            })

            fig = px.bar(
                probability_df,
                x="Resolution Path",
                y="Odds Probability (%)",
                color="Odds Probability (%)",
                text_auto=".2f",
                title="Prognosis Resolution Track Probability Distribution Map",
                color_continuous_scale="Blues"
            )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # ==========================================================
            # CONFIDENCE GAUGE
            # ==========================================================
            st.subheader("🎯 Pipeline Prediction Certainty Gauge")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Summary Meter (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0066cc"},
                    'steps': [
                        {'range': [0, 40], 'color': '#ffcccc'},
                        {'range': [40, 70], 'color': '#ffe6cc'},
                        {'range': [70, 100], 'color': '#ccffcc'}
                    ]
                }
            ))
            gauge.update_layout(height=350)
            st.plotly_chart(gauge, use_container_width=True)

            st.divider()

            # ==========================================================
            # RISK ASSESSMENT & RECOMMENDATIONS
            # ==========================================================
            st.subheader("🩺 Dynamic Healthcare Strategy Risk Assessment")

            if prediction.lower() == "released":
                st.success("""
### 🟢 STATUS: LOW CLINICAL RISKS DETECTED
The patient exhibits highly favorable parameters linked directly with systematic recovery timelines.

**Operational Guidelines:**
* ✔ Authorize transition protocols to standard outpatient observation.
* ✔ Advise routine nutritional tracking and standard hydration goals.
* ✔ Arrange standard 14-day tele-health recovery check sequence.
""")
            else:
                st.error("""
### 🔴 STATUS: HIGH RECOVERY RISKS DETECTED
The configuration matrix indicates a prolonged isolation trajectory or immediate risk vectors.

**Operational Guidelines:**
* 🚨 Trigger critical evaluation parameters for hospital bed allocation routing cells.
* 🚨 Enforce immediate and absolute local infection containment protocols.
* 🚨 Implement persistent continuous diagnostic tracking (SPO2 metrics, auxiliary updates).
""")

            st.divider()

            # ==========================================================
            # LOG PREDICTION HISTORY
            # ==========================================================
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Target Country": country,
                "Model Resolution Output": prediction,
                "Certainty Metric (%)": round(confidence, 2)
            })

            st.subheader("🕒 Session Prognosis Logging History")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            st.divider()

            # ==========================================================
            # DATA EXPORT ENGINE
            # ==========================================================
            report_df = pd.DataFrame({
                "Feature Parameters": ["Country", "Year", "Month", "Day", "Prognosis Outcome", "Certainty Metrics"],
                "Configured Value Log": [country, confirmed_year, confirmed_month, confirmed_day, prediction, f"{confidence:.2f}%"]
            })
            
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                "⬇️ Export Local Diagnostics Summary (CSV File)",
                csv_report,
                "Patient_Risk_Assessment.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error("Prognosis Execution Run Failed")
            st.exception(e)