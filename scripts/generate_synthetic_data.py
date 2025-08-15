import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


CATEGORICAL_FEATURES = {
    "gender": ["Male", "Female"],
    "smoker": ["Yes", "No"],
    "family_history": ["Yes", "No"],
    "physical_activity": ["Low", "Medium", "High"],
    "diet": ["Poor", "Average", "Good"],
}

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "blood_pressure",
    "cholesterol",
    "glucose",
]


def main(rows: int, out: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_num, y = make_classification(
        n_samples=rows,
        n_features=len(NUMERIC_FEATURES),
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        flip_y=0.02,
        class_sep=1.0,
        random_state=seed,
    )

    df = pd.DataFrame(X_num, columns=NUMERIC_FEATURES)

    # transform numeric columns into realistic ranges
    df["age"] = (df["age"].clip(-2, 2) * 15 + 50).round(0).clip(18, 90)
    df["bmi"] = (df["bmi"].clip(-2, 2) * 4 + 27).round(1).clip(15, 55)
    df["blood_pressure"] = (df["blood_pressure"].clip(-2, 2) * 15 + 130).round(0).clip(80, 220)
    df["cholesterol"] = (df["cholesterol"].clip(-2, 2) * 30 + 200).round(0).clip(100, 400)
    df["glucose"] = (df["glucose"].clip(-2, 2) * 25 + 110).round(0).clip(60, 300)

    # add categorical features with some correlation to target
    df["gender"] = rng.choice(CATEGORICAL_FEATURES["gender"], size=rows)
    df["smoker"] = rng.choice(CATEGORICAL_FEATURES["smoker"], size=rows, p=[0.3, 0.7])
    df["family_history"] = rng.choice(CATEGORICAL_FEATURES["family_history"], size=rows, p=[0.4, 0.6])
    df["physical_activity"] = rng.choice(CATEGORICAL_FEATURES["physical_activity"], size=rows, p=[0.3, 0.5, 0.2])
    df["diet"] = rng.choice(CATEGORICAL_FEATURES["diet"], size=rows, p=[0.25, 0.5, 0.25])

    # inject missingness
    for col in NUMERIC_FEATURES:
        mask = rng.random(rows) < 0.05
        df.loc[mask, col] = np.nan
    for col in CATEGORICAL_FEATURES.keys():
        mask = rng.random(rows) < 0.05
        df.loc[mask, col] = np.nan

    df["disease"] = y.astype(int)

    # generate simple doctor notes correlated with features/label for NLP
    def synth_note(row):
        parts = []
        if row["smoker"] == "Yes":
            parts.append("patient smokes")
        if row["family_history"] == "Yes":
            parts.append("family history present")
        if row["physical_activity"] == "Low":
            parts.append("sedentary lifestyle")
        if row["diet"] == "Poor":
            parts.append("poor diet")
        if row["blood_pressure"] > 140:
            parts.append("high blood pressure")
        if row["cholesterol"] > 220:
            parts.append("elevated cholesterol")
        if row["glucose"] > 140:
            parts.append("high glucose")
        if row["disease"] == 1 and not parts:
            parts.append("shows concerning symptoms")
        if not parts:
            parts.append("no significant issues noted")
        return ", ".join(parts)

    df["notes"] = df.apply(synth_note, axis=1)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": {k: v for k, v in CATEGORICAL_FEATURES.items()},
        "text_features": ["notes"],
        "target": "disease",
        "class_map": {"negative": 0, "positive": 1},
    }
    with open(os.path.join(os.path.dirname(out), "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Saved dataset to {out} with shape {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/synthetic_disease.csv")
    args = parser.parse_args()
    main(args.rows, args.out)
