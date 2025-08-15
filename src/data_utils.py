import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Schema:
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    text_features: List[str]


def infer_schema(df: pd.DataFrame, target: str, text_cols: Optional[List[str]] = None) -> Schema:
    features = [c for c in df.columns if c != target]
    text_features: List[str] = []
    if text_cols:
        text_features = [c for c in text_cols if c in df.columns]
    else:
        # auto-detect a typical notes column
        for candidate in ["notes", "doctor_notes", "clinical_notes"]:
            if candidate in df.columns:
                text_features = [candidate]
                break

    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and c not in text_features]
    categorical_features = [c for c in features if c not in numeric_features and c not in text_features]
    return Schema(numeric_features=numeric_features, categorical_features=categorical_features, target=target, text_features=text_features)


class TextColumnTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, max_features: int = 5000, ngram_range=(1, 2), min_df: int = 2):
        # store params as attributes for sklearn cloning
        self.column = column
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None

    def fit(self, X, y=None):
        s = pd.Series(X[self.column]).fillna("").astype(str)
        # initialize vectorizer at fit time with current params
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, min_df=self.min_df)
        self.vectorizer.fit(s)
        return self

    def transform(self, X):
        s = pd.Series(X[self.column]).fillna("").astype(str)
        if self.vectorizer is None:
            # in case transform called before fit (e.g., within pipelines), initialize empty fit
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, min_df=self.min_df)
            self.vectorizer.fit(pd.Series([]))
        return self.vectorizer.transform(s).toarray()

    def get_feature_names_out(self, input_features=None):
        if self.vectorizer is None:
            return np.array([])
        return np.array([f"{self.column}__{t}" for t in self.vectorizer.get_feature_names_out()])


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self.column = column
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dim = None

    def _load_model(self):
        if self._model is None:
            import importlib
            st_mod = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st_mod, "SentenceTransformer")
            self._model = SentenceTransformer(self.model_name)
            # prime to get dim
            emb = self._model.encode([""], show_progress_bar=False)
            self._dim = emb.shape[1]

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        self._load_model()
        s = pd.Series(X[self.column]).fillna("").astype(str).tolist()
        emb = self._model.encode(s, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(emb)

    def get_feature_names_out(self, input_features=None):
        dim = self._dim or 384
        return np.array([f"{self.column}__emb_{i}" for i in range(dim)])


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
        ]
    )
    transformers = [
        ("num", numeric_pipeline, schema.numeric_features),
        ("cat", categorical_pipeline, schema.categorical_features),
    ]
    if schema.text_features:
        # current support: first text column only
        text_col = schema.text_features[0]
        use_emb = os.environ.get("USE_EMBEDDINGS", "0") == "1"
        if use_emb:
            transformers.append(("text_emb", SentenceEmbeddingTransformer(text_col), [text_col]))
        else:
            transformers.append(("text", TextColumnTfidfVectorizer(text_col), [text_col]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    # Convert to 0/1 if not numeric
    if not pd.api.types.is_numeric_dtype(y):
        classes = sorted(y.unique())
        class_map = {cls: i for i, cls in enumerate(classes)}
        y = y.map(class_map).astype(int)
    return X, y


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic cleaning: drop completely empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    return df


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps") and "encoder" in trans.named_steps:
            enc = trans.named_steps["encoder"]
            try:
                enc_feature_names = enc.get_feature_names_out(cols)
            except Exception:
                enc_feature_names = enc.get_feature_names(cols)
            feature_names.extend(list(enc_feature_names))
        elif hasattr(trans, "get_feature_names_out"):
            feature_names.extend(list(trans.get_feature_names_out()))
        else:
            # numeric pipeline
            if isinstance(cols, list):
                feature_names.extend(cols)
            else:
                feature_names.append(cols)
    return feature_names
