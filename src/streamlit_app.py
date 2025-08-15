import json
import os
from typing import Dict

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DIAGNOSIS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical Diagnosis", page_icon="🩺", layout="centered")

st.title("🩺 Medical Diagnosis Predictor")

st.caption("Enter patient data and get a prediction. This UI sends requests to the FastAPI service.")

# Fetch schema for dynamic form
@st.cache_data(show_spinner=False)
def get_schema() -> Dict:
    try:
        r = requests.get(f"{API_URL}/schema", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

schema = get_schema()

if not schema:
    st.warning("Could not load schema from API. Start the API server first.")
    st.stop()

num_cols = schema.get("numeric_features", [])
cat_cols = schema.get("categorical_features", [])
text_cols = schema.get("text_features", [])
num_ranges = schema.get("numeric_ranges", {})
cat_vocab = schema.get("categorical_vocab", {})

with st.sidebar:
    st.markdown("### Settings")
    st.text_input("API URL", value=API_URL, key="api_url")

st.subheader("Patient Features")

inputs: Dict[str, object] = {}
cols = st.columns(2)
for i, col in enumerate(num_cols):
    with cols[i % 2]:
        if col in num_ranges:
            r = num_ranges[col]
            inputs[col] = st.slider(col, min_value=float(r["min"]), max_value=float(r["max"]), value=float(r["min"]))
        else:
            inputs[col] = st.number_input(col, value=0.0)

for i, col in enumerate(cat_cols):
    with cols[i % 2]:
        if col in cat_vocab and cat_vocab[col]:
            inputs[col] = st.selectbox(col, options=[""] + cat_vocab[col], index=0)
        else:
            inputs[col] = st.text_input(col, value="")

if text_cols:
    st.subheader("Doctor Notes")
    for col in text_cols:
        inputs[col] = st.text_area(col, height=150)

if st.button("Predict"):
    try:
        base = st.session_state.get("api_url", API_URL)
        r = requests.post(f"{base}/predict", json={"features": inputs}, timeout=10)
        r.raise_for_status()
        data = r.json()
        prob = data.get("probability", 0.0)
        pred = data.get("prediction", 0)
        st.success(f"Prediction: {'Positive' if pred==1 else 'Negative'} (probability: {prob:.2f})")
        explanation = data.get("explanation", [])
        if explanation:
            st.subheader("Top factors")
            df = pd.DataFrame(explanation)
            st.bar_chart(df.set_index("feature")['contribution'])
    except Exception as e:
        st.error(f"Request failed: {e}")
