# Medical Diagnosis ML System
<img width="401" height="506" alt="Screenshot 2025-08-15 at 12 45 07 PM" src="https://github.com/user-attachments/assets/0dbe88ed-f583-4e0e-acac-a88acc61daef" />

An end-to-end Python machine learning application to predict a binary disease outcome (e.g., diabetes/heart disease) with:
- Data loading/cleaning, missing value handling, categorical encoding, and scaling
- Multiple models: Logistic Regression, Random Forest, SVM, MLP (simple neural net)
- Evaluation: accuracy, precision, recall, F1, ROC-AUC
- Visualizations: confusion matrix, feature importance
- Best model saved as `.pkl`
- FastAPI REST API for predictions with explanations (SHAP when available, LIME fallback, or model importances)
- Streamlit UI for interactive predictions
 - Optional Dockerized deployment (API + UI)

## Quick start

### 1) Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Generate a synthetic dataset (optional)
This creates a demo dataset at `data/synthetic_disease.csv` with mixed numeric and categorical features and a binary target `disease`.
```bash
python scripts/generate_synthetic_data.py --rows 1000 --out data/synthetic_disease.csv
```

### 4) Train models and produce artifacts
```bash
python src/train.py --data data/synthetic_disease.csv --target disease --out-dir artifacts
```
Artifacts produced:
- `artifacts/best_model.pkl` – the best pipeline (preprocessing + model)
- `artifacts/metadata.json` – feature schema and settings for UI/API
- `artifacts/metrics.json` – metrics per model
- `artifacts/confusion_matrix.png` – confusion matrix (best model)
- `artifacts/feature_importance.png` – top features (best model)
- `artifacts/transformed_feature_names.json` – names after preprocessing
- `artifacts/error_report.json` – flagged potential false positives/negatives from the test set
- `artifacts/shap_background.csv` – background sample for explanations (if available)
- `artifacts/class_map.json` – mapping between class labels and 0/1 used for training

### 5) Run the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Open docs at http://localhost:8000/docs

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"age": 50, "bmi": 31.2, "blood_pressure": 135, "cholesterol": 220, "glucose": 160, "gender": "Female", "smoker": "No", "family_history": "Yes", "physical_activity": "Low", "diet": "Poor"}}'
```

Health check and metrics:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### 6) Run the Streamlit UI
```bash
streamlit run src/streamlit_app.py
```
By default the UI calls the API at `http://localhost:8000`. You can set another URL:
```bash
export DIAGNOSIS_API_URL="http://localhost:8000"
```

### 7) Run with Docker Compose (optional)
```bash
docker compose up --build
```
- API: http://localhost:8000
- UI: http://localhost:8501

## Use your own dataset
Provide a CSV with a binary target column (e.g., `disease`). The script will:
- Infer numeric vs categorical features
- Impute missing values (median for numeric, most frequent for categorical)
- One-hot encode categorical features
- Standard-scale numeric features

Train:
```bash
python src/train.py --data path/to/your.csv --target your_target_column
```

## Notes
- NLP: Doctor notes are ingested via TF-IDF by default; optionally enable sentence-transformer embeddings by setting `USE_EMBEDDINGS=1` before training.
- Explanations: Uses SHAP when installed; otherwise falls back to LIME if installed; else uses model importances/coefficients.
- All inputs are validated server-side. Unknown categorical values are handled by the encoder.
 - API clamps numeric inputs to training min/max and logs unknown categorical values to `logs/api.log`.
 - Training logs are written to `logs/training.log`.

### Optional packages
- To enable LIME explanations, install `lime` in your environment.
- To enable SHAP explanations, install `shap` (may require extras on Apple Silicon). The app works without it.

## Project structure
```
.
├── artifacts/                 # created after training
├── data/                      # datasets (ignored if you provide your own)
├── scripts/
│   └── generate_synthetic_data.py
├── src/
│   ├── api.py                 # FastAPI app
│   ├── data_utils.py          # preprocessing + utilities
│   ├── models.py              # model zoo
│   ├── streamlit_app.py       # Streamlit UI
│   └── train.py               # training & evaluation
├── requirements.txt
└── README.md
```

## Troubleshooting
- If SHAP warnings appear for certain models, the API will fall back to a simpler explanation method.
- Ensure the same Python environment is used for training and serving.
