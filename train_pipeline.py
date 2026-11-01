import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def clean_git_markers(file_path):
    """Removes Git merge conflict markers from raw CSV files."""
    if not os.path.exists(file_path):
        return False
        
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    # Filter out merge conflict markers
    cleaned_lines = [
        line for line in lines 
        if not (line.startswith("<<<<<<<") or line.startswith("=======") or line.startswith(">>>>>>>"))
    ]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)
    return True

def build_and_train():
    csv_file = "patient.csv"
    
    # Clean Git headers automatically
    if not clean_git_markers(csv_file):
        print(f"Please place '{csv_file}' in the current working directory.")
        return

    # 1. Load Data safely
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip() # Remove hidden whitespaces
    
    # 2. Feature Engineering & Target Mapping
    date_cols = ['confirmed_date', 'released_date', 'deceased_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
    if 'confirmed_date' in df.columns:
        df['confirmed_month'] = df['confirmed_date'].dt.month.fillna(1).astype(int)
        df['confirmed_day'] = df['confirmed_date'].dt.day.fillna(1).astype(int)
    else:
        df['confirmed_month'] = 1
        df['confirmed_day'] = 1
    
    # Ensure standard target column setup without leakage
    if 'state' in df.columns:
        df['target'] = df['state'].apply(lambda x: 1 if str(x).lower() == 'released' else 0)
    else:
        raise KeyError("Could not find the target column 'state' in your dataset.")
    
    # Drop unique index keys, dates, and leaky flag columns
    drop_cols = ['id', 'state', 'released_date', 'deceased_date', 'confirmed_date', 
                 'infected_by', 'infection_order', 'global_num', 'released_flag', 'deceased_flag']
    
    features_df = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
    
    X = features_df
    y = df['target']
    
    # Segment columns dynamically by data type
    num_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Build Transformers and Full Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    # Encapsulate preprocessors alongside classifier inside the dumped model object
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10))
    ])
    
    # 4. Stratified Split and Fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training optimized model pipeline...")
    full_pipeline.fit(X_train, y_train)
    
    preds = full_pipeline.predict(X_test)
    print(f"Optimal Model Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    # Export entire model runtime block
    joblib.dump(full_pipeline, "covid_prediction_model.pkl")
    print("Pipeline exported successfully as 'covid_prediction_model.pkl'!")

if __name__ == "__main__":
    build_and_train()