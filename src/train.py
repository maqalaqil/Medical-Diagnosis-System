import argparse
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

# Support running as a script or as a module
try:
    from .data_utils import (
        Schema,
        build_preprocessor,
        get_feature_names,
        infer_schema,
        load_csv,
        save_json,
        split_xy,
    )
    from .models import get_model_zoo
except ImportError:  # pragma: no cover
    from data_utils import (
        Schema,
        build_preprocessor,
        get_feature_names,
        infer_schema,
        load_csv,
        save_json,
        split_xy,
    )
    from models import get_model_zoo


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate_model(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title="Confusion Matrix")
    fig.write_image(out_path)


def plot_feature_importance(model, feature_names: List[str], out_path: str):
    values = None
    try:
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_).ravel()
        else:
            # use permutation importance as fallback
            values = None
    except Exception:
        values = None

    if values is None:
        # fallback image with note
        fig = go.Figure()
        fig.add_annotation(text="Feature importance not available for this model",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(width=800, height=400)
        fig.write_image(out_path)
        return

    idx = np.argsort(values)[::-1][:20]
    names = [feature_names[i] for i in idx]
    fig = go.Figure(go.Bar(x=values[idx][::-1], y=names[::-1], orientation="h"))
    fig.update_layout(title="Top Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
    fig.write_image(out_path)


def train_and_evaluate(df: pd.DataFrame, target: str, test_size: float, out_dir: str, random_state: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    # logging setup
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler(os.path.join(log_dir, "training.log"), maxBytes=2_000_000, backupCount=2)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    schema = infer_schema(df, target)
    X, y = split_xy(df, target)

    # class map for UI/API
    class_map = {"negative": 0, "positive": 1}
    save_json(os.path.join(out_dir, "class_map.json"), class_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(schema)

    models = get_model_zoo(random_state)
    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_auc = -np.inf
    best_pipeline = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for name, mdl in models.items():
        pipeline = Pipeline(steps=[("pre", preprocessor), ("model", mdl)])
        # cross-val predicted probabilities on train split for model selection
        y_cv_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        y_cv_pred = (y_cv_proba >= 0.5).astype(int)
        cv_metrics = evaluate_model(y_train, y_cv_pred, y_cv_proba)
        # fit final model on full train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        res = evaluate_model(y_test, y_pred, y_proba)
        res_cv = {f"cv_{k}": v for k, v in cv_metrics.items()}
        results[name] = {**res, **res_cv}
        logger.info("model=%s metrics=%s", name, results[name])
        # select best by CV AUC primarily, then test AUC as tiebreaker
        score = cv_metrics["roc_auc"]
        if score > best_auc:
            best_auc = score
            best_name = name
            best_pipeline = pipeline

    # save metrics
    save_json(os.path.join(out_dir, "metrics.json"), results)

    # feature names after fit
    pre = best_pipeline.named_steps["pre"]
    pre.fit(X_train, y_train)
    feature_names = get_feature_names(pre)
    save_json(os.path.join(out_dir, "transformed_feature_names.json"), feature_names)

    # confusion matrix and feature importance plots
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, os.path.join(out_dir, "confusion_matrix.png"))

    # plot feature importance of the model only (without preprocessor)
    try:
        model = best_pipeline.named_steps["model"]
        plot_feature_importance(model, feature_names, os.path.join(out_dir, "feature_importance.png"))
    except Exception:
        pass

    # Save input feature order for API/UI
    input_feature_order = list(X_train.columns)
    save_json(os.path.join(out_dir, "input_feature_order.json"), input_feature_order)

    # try to compute and save a small SHAP background for API explanations (original features)
    if HAS_SHAP:
        try:
            bg = shap.utils.sample(X_train, 100, random_state=random_state)
            bg.to_csv(os.path.join(out_dir, "shap_background.csv"), index=False)
        except Exception:
            pass

    # Build categorical vocab and numeric ranges from training data
    categorical_vocab = {}
    for c in schema.categorical_features:
        vals = X_train[c].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        categorical_vocab[c] = vals
    numeric_ranges = {}
    for c in schema.numeric_features:
        s = X_train[c]
        numeric_ranges[c] = {"min": float(np.nanmin(s.values)), "max": float(np.nanmax(s.values))}

    # save metadata for API/UI
    metadata = {
        "schema": {
            "numeric_features": schema.numeric_features,
            "categorical_features": schema.categorical_features,
            "text_features": schema.text_features,
            "target": target,
            "categorical_vocab": categorical_vocab,
            "numeric_ranges": numeric_ranges,
        },
        "best_model": best_name,
        "metrics": results[best_name],
    }
    save_json(os.path.join(out_dir, "metadata.json"), metadata)

    # persist best model pipeline
    joblib.dump(best_pipeline, os.path.join(out_dir, "best_model.pkl"))

    # Identify and save potential false positives/negatives from test set
    try:
        y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
        report_rows = []
        X_test_reset = X_test.reset_index(drop=True)
        for i, (yt, yp, p) in enumerate(zip(y_test.reset_index(drop=True), y_pred_best, y_proba_best)):
            if yt == 0 and yp == 1:
                kind = "false_positive"
            elif yt == 1 and yp == 0:
                kind = "false_negative"
            else:
                continue
            row = {"index": int(i), "actual": int(yt), "pred": int(yp), "probability": float(p), "type": kind}
            # include a few key fields for triage if present
            for col in ["age", "bmi", "blood_pressure", "cholesterol", "glucose"]:
                if col in X_test_reset.columns:
                    v = X_test_reset.iloc[i][col]
                    row[col] = float(v) if pd.api.types.is_numeric_dtype(type(v)) or isinstance(v, (int, float, np.floating)) else v
            if "notes" in X_test_reset.columns:
                row["notes"] = str(X_test_reset.iloc[i]["notes"])[:300]
            report_rows.append(row)
        with open(os.path.join(out_dir, "error_report.json"), "w") as f:
            json.dump(report_rows, f, indent=2)
    except Exception:
        pass

    logger.info("best_model=%s roc_auc=%.3f", best_name, best_auc)
    print(f"Best model: {best_name} with ROC-AUC={best_auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_csv(args.data)
    train_and_evaluate(df, target=args.target, test_size=args.test_size, out_dir=args.out_dir)
