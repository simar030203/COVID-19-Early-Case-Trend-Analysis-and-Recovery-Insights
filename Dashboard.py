import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="COVID-19 Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 COVID-19 Analytics Dashboard")
st.caption("Exploratory visualizations and clinical pattern distribution desks.")
st.divider()

# -----------------------------------
# Safe Load Dataset (Sanitizes Git conflict headers)
# -----------------------------------
@st.cache_data
def load_data():
    path = "patient.csv"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Clean out version tracking text block remnants dynamically
        cleaned = [l for l in lines if not (l.startswith("<<<") or l.startswith("===") or l.startswith(">>>"))]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(cleaned)
    return pd.read_csv(path)

df = load_data()
df.columns = df.columns.str.strip()  # Strip trailing or leading spaces globally

st.success("✅ Dataset Loaded and Normalized Successfully")

# -----------------------------------
# Dataset Preview
# -----------------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.divider()

# -----------------------------------
# Dataset Quick Metrics
# -----------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows Logged", df.shape[0])
with col2:
    st.metric("Total Data Features", df.shape[1])
with col3:
    st.metric("Total Missing Cells", int(df.isnull().sum().sum()))
st.divider()

# -----------------------------------
# Visual Distributions Space
# -----------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🗺️ Country-wise Patient Distribution")
    if "country" in df.columns:
        fig1 = px.histogram(df, x="country", color="country", title="Case Counts by Country")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Feature column 'country' missing from source records.")

with right_col:
    st.subheader("🎯 Patient Outcome Breakdown")
    if "state" in df.columns:
        fig2 = px.pie(df, names="state", title="Patient Resolution States")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Feature column 'state' missing from source records.")
st.divider()

# -----------------------------------
# Time Series Extraction
# -----------------------------------
st.subheader("📉 Line Analysis: Cases Logged Across Month Timelines")

# Parse months automatically from standard timestamp strings if column isn't pre-computed
if "confirmed_month" not in df.columns and "confirmed_date" in df.columns:
    df["confirmed_date"] = pd.to_datetime(df["confirmed_date"], errors="coerce")
    df["confirmed_month"] = df["confirmed_date"].dt.month

if "confirmed_month" in df.columns and not df["confirmed_month"].isna().all():
    monthly_data = df.groupby("confirmed_month").size().reset_index(name="Confirmed Cases")
    fig3 = px.line(
        monthly_data, x="confirmed_month", y="Confirmed Cases", 
        markers=True, title="Timeline Case Trends By Month Number"
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("ℹ️ Time-series mapping is unavailable—insufficient data found in date logs.")
st.divider()

# -----------------------------------
# Correlation Matrix Safeguard
# -----------------------------------
st.subheader("📊 Feature Matrix Correlations")

# Select explicitly valid float/integer representations
numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna(axis=1, how="all")

if not numeric_df.empty and numeric_df.shape[1] > 1:
    fig_heat, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_heat)
else:
    st.info("ℹ️ Not enough numerical dimensions are available to construct a valid covariance matrix map.")
st.divider()

# -----------------------------------
# Variable Density Distributions
# -----------------------------------
st.subheader("🔍 Interactive Variable Distribution")

numeric_columns = numeric_df.columns.tolist()
if numeric_columns:
    selected_feature = st.selectbox("Select Target Attribute to Profile:", numeric_columns)
    fig4 = px.histogram(df, x=selected_feature, nbins=30, title=f"Density Breakdown: {selected_feature}")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("No continuous numeric features found for histograms.")
st.divider()

# -----------------------------------
# File Export Engine
# -----------------------------------
csv_data = df.to_csv(index=False)
st.download_button(
    label="📥 Export Cleaned Dataset (CSV)",
    data=csv_data,
    file_name="sanitized_patient_dataset.csv",
    mime="text/csv",
    use_container_width=True
)