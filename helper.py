import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================================
# CUSTOM CSS
# ==========================================================

def load_css():

    st.markdown("""

    <style>

    .main{
        background-color:#f7f7f7;
    }

    .stButton>button{
        width:100%;
        border-radius:10px;
        height:50px;
        font-size:18px;
        font-weight:bold;
        background:#0066cc;
        color:white;
    }

    .stButton>button:hover{
        background:#004d99;
        color:white;
    }

    .metric-card{
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:2px 2px 10px rgba(0,0,0,0.2);
    }

    footer{
        visibility:hidden;
    }

    </style>

    """,unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================

def page_header(title,subtitle):

    st.title(title)

    st.caption(subtitle)

    st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

def sidebar():

    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
        width=120
    )

    st.sidebar.title("COVID-19 AI Analytics")

    st.sidebar.success("""
Navigation

🏠 Home

📊 Dashboard

🤖 Prediction

🧠 AI Insights
""")


# ==========================================================
# FOOTER
# ==========================================================

def footer():

    st.divider()

    st.markdown("""

Developed by **Simar Babbar**

MBA Business Analytics | Data Analytics | Data Science


© 2026

""")


# ==========================================================
# METRIC CARDS
# ==========================================================

def show_metrics(df):

    col1,col2,col3,col4=st.columns(4)

    with col1:
        st.metric("Rows",df.shape[0])

    with col2:
        st.metric("Columns",df.shape[1])

    with col3:
        st.metric("Missing",int(df.isnull().sum().sum()))

    with col4:
        st.metric("Duplicates",int(df.duplicated().sum()))


# ==========================================================
# MISSING VALUE TABLE
# ==========================================================

def missing_table(df):

    missing=pd.DataFrame({

        "Column":df.columns,

        "Missing Values":df.isnull().sum(),

        "Percentage":round(
            df.isnull().mean()*100,
            2
        )

    })

    return missing.sort_values(
        by="Percentage",
        ascending=False
    )


# ==========================================================
# PIE CHART
# ==========================================================

def pie_chart(df,column,title):

    fig=px.pie(

        df,

        names=column,

        title=title

    )

    st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# BAR CHART
# ==========================================================

def bar_chart(df,column,title):

    fig=px.histogram(

        df,

        x=column,

        color=column,

        title=title

    )

    st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# LINE CHART
# ==========================================================

def line_chart(data,x,y,title):

    fig=px.line(

        data,

        x=x,

        y=y,

        markers=True,

        title=title

    )

    st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def heatmap(df):

    numeric=df.select_dtypes(
        include=["int64","float64"]
    )

    fig,ax=plt.subplots(figsize=(8,6))

    sns.heatmap(

        numeric.corr(),

        annot=True,

        cmap="coolwarm",

        ax=ax

    )

    st.pyplot(fig)


# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

def download_dataset(df):

    csv=df.to_csv(index=False)

    st.download_button(

        "⬇ Download Dataset",

        csv,

        "patient_dataset.csv",

        "text/csv"

    )


# ==========================================================
# SUCCESS BOX
# ==========================================================

def success_box(message):

    st.success(message)


# ==========================================================
# WARNING BOX
# ==========================================================

def warning_box(message):

    st.warning(message)


# ==========================================================
# ERROR BOX
# ==========================================================

def error_box(message):

    st.error(message)


# ==========================================================
# INFO BOX
# ==========================================================

def info_box(message):

    st.info(message)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance(model,columns):

    if hasattr(model,"feature_importances_"):

        importance=pd.DataFrame({

            "Feature":columns,

            "Importance":model.feature_importances_

        })

        importance=importance.sort_values(

            by="Importance",

            ascending=False

        )

        fig=px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            title="Feature Importance"

        )

        st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# PREDICTION PROBABILITY
# ==========================================================

def probability_chart(classes,probability):

    df=pd.DataFrame({

        "State":classes,

        "Probability":probability*100

    })

    fig=px.bar(

        df,

        x="State",

        y="Probability",

        color="Probability",

        text="Probability",

        title="Prediction Probability"

    )

    st.plotly_chart(fig,use_container_width=True)