from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from ml.features import extract_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def build_and_train(
    texts: list,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    model_type: str = "logistic",
    structured_df: Optional[Any] = None,
    random_state: int = 42
) -> Tuple[Any, Any, Dict[str, float]]:
    """
    Role 3 Contract:
    Fits feature vectorizer and classifier on resume texts.
    Accepts optional sample `weights` for fairness mitigation (reweighing).
    
    Returns:
        (fitted_vectorizer, fitted_classifier, metrics_dict)
    """
    X, vectorizer = extract_features(texts, fit=True, structured_df=structured_df)
    
    if model_type == "logistic":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="liblinear"
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    clf.fit(X, y, sample_weight=weights)
    
    # Calculate training / evaluation metrics
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else preds
    
    metrics = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, probs)), 4)
    }
    
    return vectorizer, clf, metrics

def predict(
    vectorizer: Any,
    clf: Any,
    texts: list,
    structured_df: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Role 3 Contract:
    Given vectorizer and classifier, produces binary predictions and probabilities.
    """
    X, _ = extract_features(texts, vectorizer=vectorizer, fit=False, structured_df=structured_df)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else preds.astype(float)
    return preds, probs

def save_model_artifacts(vectorizer: Any, clf: Any, name_prefix: str = "baseline") -> None:
    """Saves model and vectorizer to models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / f"{name_prefix}_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / f"{name_prefix}_vectorizer.pkl")

def load_model_artifacts(name_prefix: str = "baseline") -> Tuple[Any, Any]:
    """Loads saved model and vectorizer from models/ directory."""
    clf_path = MODELS_DIR / f"{name_prefix}_model.pkl"
    vec_path = MODELS_DIR / f"{name_prefix}_vectorizer.pkl"
    
    if not clf_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Model artifacts not found for prefix '{name_prefix}' in {MODELS_DIR}")
        
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)
    return vec, clf
