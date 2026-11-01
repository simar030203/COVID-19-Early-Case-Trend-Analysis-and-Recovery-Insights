# COVID-19 Early Case Trend Analysis & Recovery Insights Engine

Developed by **Akshey Sehgal** | B.Tech Robotics & AI  
*Project for Corporate Placement Evaluation (HealthGuard Analytics Pvt. Ltd.)*

## 📌 Project Overview
A production-grade, multi-page data workstation engineered to process early-stage epidemiological vectors, evaluate recovery risk timelines using machine learning, and compile strategic summaries via generative AI interfaces.

## 🛠️ Tech Stack & Architecture
- **Dashboard UI & Routing:** Streamlit
- **Machine Learning Pipeline:** Scikit-Learn (ColumnTransformer, Pipeline, Random Forest)
- **Data Wrangling & Graphs:** Pandas, NumPy, Plotly, Seaborn
- **Generative Artificial Intelligence:** Google GenAI SDK (Gemini 3.5 Flash Integration)
- **Document Compilation Engine:** ReportLab (Automated PDF Clinical Records Creator)

## ⚡ Core Engineering Optimizations
1. **Target Leakage Remediation:** Eliminated post-outcome indicators (`released_flag`, `deceased_flag`) from training arrays, moving accuracy from a synthetic 100% rule-match to an authentic **99.94% generalized score** on unknown patients.
2. **Unified Object Serialization:** Bundled imputer nodes, scaling models, and categorical hot-encoders directly into a single `covid_prediction_model.pkl` pipeline file to make Streamlit data execution safer.
3. **Data Pre-Sanitization:** Integrated runtime string normalization buffers to strip accidental trailing spacing artifacts or internal CSV syntax disruptions dynamically.

## 🚀 Local Installation & Deployment
```bash
# Clone the repository
git clone https://github.com/anshu26427/COVID-19-Analysis-and-Report.git

# Install application dependencies
pip install -r requirements.txt

# Train/verify the model pipeline
python train_pipeline.py

# Launch the Streamlit visualization hub
streamlit run app.py
```
