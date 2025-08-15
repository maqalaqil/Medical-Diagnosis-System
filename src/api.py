import json
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    HAS_LIME = True
except Exception:
    HAS_LIME = False
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys as _sys
try:
    # Ensure unpickling works when training was run as a script using top-level 'data_utils'
    from . import data_utils as _du  # type: ignore
    _sys.modules.setdefault("data_utils", _du)
except Exception:
    pass

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "transformed_feature_names.json")
CLASS_MAP_PATH = os.path.join(ARTIFACTS_DIR, "class_map.json")
BACKGROUND_CSV = os.path.join(ARTIFACTS_DIR, "shap_background.csv")
INPUT_FEATURE_ORDER = os.path.join(ARTIFACTS_DIR, "input_feature_order.json")

app = FastAPI(title="Medical Diagnosis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# logging
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(os.path.join(log_dir, "api.log"), maxBytes=2_000_000, backupCount=2)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class PredictRequest(BaseModel):
    features: Dict[str, object]


class ExplanationItem(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    explanation: List[ExplanationItem]
    model: str


# Load artifacts on startup
_model = None
_metadata = None
_feature_names = None
_class_map = None
_explainer = None
_input_feature_order = None
_lime_explainer = None


def _load_artifacts():
    global _model, _metadata, _feature_names, _class_map, _explainer, _input_feature_order, _lime_explainer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run training first.")
    _model = joblib.load(MODEL_PATH)

    with open(META_PATH) as f:
        _metadata = json.load(f)
    with open(FEATURE_NAMES_PATH) as f:
        _feature_names = json.load(f)
    with open(CLASS_MAP_PATH) as f:
        _class_map = json.load(f)

    # Load input feature order
    if os.path.exists(INPUT_FEATURE_ORDER):
        with open(INPUT_FEATURE_ORDER) as f:
            _input_feature_order = json.load(f)
    else:
        _input_feature_order = None

    # Build SHAP or LIME explainer if possible
    _explainer = None
    _lime_explainer = None
    if HAS_SHAP:
        try:
            if os.path.exists(BACKGROUND_CSV):
                background_df = pd.read_csv(BACKGROUND_CSV)
            else:
                background_df = None
            # Define prediction function on raw features; pipeline handles preprocessing
            def predict_proba_raw(Xraw: np.ndarray) -> np.ndarray:
                df = pd.DataFrame(Xraw, columns=_input_feature_order or _model.feature_names_in_)
                return _model.predict_proba(df)[:, 1]

            if background_df is not None and not background_df.empty:
                _explainer = shap.Explainer(predict_proba_raw, background_df)
            else:
                _explainer = shap.Explainer(predict_proba_raw)
        except Exception:
            _explainer = None

    # LIME fallback if SHAP unavailable or failed
    if _explainer is None and HAS_LIME:
        try:
            if os.path.exists(BACKGROUND_CSV):
                background_df = pd.read_csv(BACKGROUND_CSV)
            else:
                background_df = None
            if background_df is not None and not background_df.empty:
                feature_names = list(background_df.columns)
                categorical_features = []  # optional: could infer
                _lime_explainer = LimeTabularExplainer(
                    training_data=background_df.values,
                    feature_names=feature_names,
                    categorical_features=categorical_features,
                    verbose=False,
                    class_names=["negative", "positive"],
                    mode="classification",
                )
        except Exception:
            _lime_explainer = None


def _ensure_loaded():
    """Lazily load artifacts if the FastAPI startup hook didn't run (e.g., in tests)."""
    global _metadata
    if _metadata is None:
        _load_artifacts()

@app.on_event("startup")
async def startup_event():
    _load_artifacts()


@app.get("/health")
async def health():
    _ensure_loaded()
    return {"status": "ok", "model": _metadata.get("best_model")}


@app.get("/schema")
async def get_schema():
    _ensure_loaded()
    return _metadata.get("schema", {})


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    _ensure_loaded()
    # Validate features against schema
    schema = _metadata["schema"]
    text_cols = schema.get("text_features", [])
    cols = schema["numeric_features"] + schema["categorical_features"] + text_cols

    # assemble single-row DataFrame
    row = {}
    for c in cols:
        if c in text_cols:
            row[c] = req.features.get(c, "")
        else:
            row[c] = req.features.get(c, None)

    # numeric ranges validation: clamp to [min,max] if provided
    for c in schema.get("numeric_ranges", {}):
        if row.get(c) is not None:
            try:
                v = float(row[c])
                r = schema["numeric_ranges"][c]
                row[c] = min(max(v, r["min"]), r["max"])
            except Exception:
                pass

    # categorical vocab validation: if string not in vocab, leave as-is (encoder handles infrequent), but log
    for c, vocab in schema.get("categorical_vocab", {}).items():
        if row.get(c) is not None and str(row[c]) not in vocab:
            logger.info("unknown_categorical_value feature=%s value=%s", c, row[c])
    df = pd.DataFrame([row])

    # run pipeline
    try:
        proba = float(_model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # explanation
    explanation: List[Dict[str, float]] = []
    try:
        # Transform to model's feature space using preprocessor
        if _explainer is not None:
            # Explain on raw inputs following input feature order
            df_shap = df[_input_feature_order] if _input_feature_order else df
            shap_values = _explainer(df_shap)
            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
            else:
                vals = np.array(shap_values)[0]
            idx = np.argsort(np.abs(vals))[::-1][:10]
            explanation = [
                {"feature": (_input_feature_order[i] if _input_feature_order else _feature_names[i]), "contribution": float(vals[i])}
                for i in idx
            ]
        elif _lime_explainer is not None:
            x = (df[_input_feature_order] if _input_feature_order else df).iloc[0].values
            # predict_proba wrapper for LIME expects full probability vector
            def predict_fn(Xraw: np.ndarray) -> np.ndarray:
                d = pd.DataFrame(Xraw, columns=_input_feature_order or _model.feature_names_in_)
                probs = _model.predict_proba(d)
                return np.column_stack([1 - probs[:, 1], probs[:, 1]])

            exp = _lime_explainer.explain_instance(x, predict_fn, num_features=10)
            # LIME returns list of (feature, weight)
            explanation = [
                {"feature": str(name), "contribution": float(weight)} for name, weight in exp.as_list(label=1)
            ]
        else:
            # fallback: coefficients/feature_importances
            model = _model.named_steps.get("model")
            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
            elif hasattr(model, "coef_"):
                vals = np.abs(model.coef_).ravel()
            else:
                vals = np.zeros(len(_feature_names))
            idx = np.argsort(vals)[::-1][:10]
            explanation = [
                {"feature": _feature_names[i], "contribution": float(vals[i])}
                for i in idx
            ]
    except Exception:
        pass

    logger.info("prediction proba=%.4f pred=%d", proba, pred)
    return PredictResponse(
        prediction=pred,
        probability=proba,
        explanation=explanation,
        model=_metadata.get("best_model", "unknown"),
    )


@app.get("/metrics")
async def get_metrics():
    try:
        with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/error-report")
async def get_error_report():
    path = os.path.join(ARTIFACTS_DIR, "error_report.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []
