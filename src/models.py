from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_model_zoo(random_state: int = 42) -> Dict[str, object]:
    models = {
    "logreg": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced"),
        "svm": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=random_state),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-4, max_iter=300, random_state=random_state),
    }
    return models
