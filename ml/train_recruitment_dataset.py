import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.audit import audit
from ml.model import save_model_artifacts
from ml.redact import redact, load_names
from ml.mitigate import compute_reweighing_weights

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kaggle_recruitment" / "dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def clean_first_name(raw_name: Any) -> str:
    """Extracts first name while properly stripping professional titles / honorifics."""
    if not isinstance(raw_name, str):
        return ""
    clean = re.sub(r"^(mr\.|mrs\.|ms\.|dr\.|miss)\s+", "", raw_name.strip(), flags=re.IGNORECASE)
    parts = clean.split()
    return parts[0].capitalize() if parts else ""

def load_and_preprocess_dataset() -> pd.DataFrame:
    """
    Loads and cleans the Kaggle AI Recruitment Pipeline Dataset (10,174 candidates).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Kaggle recruitment dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # 1. Clean Target Hiring Decision
    df["y_hired"] = (df["decision"].astype(str).str.strip().str.lower() == "select").astype(int)
    
    df["first_name"] = df["Name"].apply(clean_first_name)
    
    # 2. Extract Gender column directly from dataset (or fallback to first-name matching)
    if "Gender" in df.columns:
        df["inferred_gender"] = df["Gender"].astype(str)
    else:
        males, females = load_names()
        male_set = set(males)
        female_set = set(females)
        
        def infer_gender(first: str) -> str:
            if first in female_set:
                return "Female"
            if first in male_set:
                return "Male"
            return "Other"

        df["inferred_gender"] = df["first_name"].apply(infer_gender)
    
    # Clean text column
    df["clean_resume"] = df["Resume"].fillna("").astype(str)
    
    return df

def compute_role_breakdown(df: pd.DataFrame, min_candidates: int = 100) -> List[Dict[str, Any]]:
    """Computes raw historical selection rates for male vs female candidates across top roles."""
    role_stats = []
    top_roles = df["Role"].value_counts()
    
    for role, count in top_roles.items():
        if count < min_candidates:
            continue
        sub = df[df["Role"] == role]
        f_sub = sub[sub["inferred_gender"] == "Female"]
        m_sub = sub[sub["inferred_gender"] == "Male"]
        
        f_n = len(f_sub)
        m_n = len(m_sub)
        
        if f_n >= 15 and m_n >= 15:
            f_rate = float(f_sub["y_hired"].mean())
            m_rate = float(m_sub["y_hired"].mean())
            role_stats.append({
                "role": role,
                "total_candidates": int(count),
                "female_candidates": f_n,
                "male_candidates": m_n,
                "female_selection_rate": round(f_rate, 4),
                "male_selection_rate": round(m_rate, 4),
                "disparity_delta": round(m_rate - f_rate, 4),
                "disparate_impact_ratio": round(f_rate / m_rate, 4) if m_rate > 0 else 1.0
            })
            
    role_stats.sort(key=lambda x: x["disparity_delta"], reverse=True)
    return role_stats

def evaluate_counterfactual_swaps(
    df_test: pd.DataFrame,
    clf_base: Any,
    vec_base: Any,
    clf_red: Any,
    vec_red: Any,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Evaluates score stability when a candidate's name is counterfactually swapped
    (e.g., Male -> Female or Female -> Male) on authentic test resumes.
    """
    males, females = load_names()
    known_df = df_test[df_test["inferred_gender"].isin(["Male", "Female"])].head(n_samples)
    
    shifts_base = []
    shifts_red = []
    
    for _, row in known_df.iterrows():
        orig_text = row["clean_resume"]
        orig_name = row["first_name"]
        is_male = row["inferred_gender"] == "Male"
        swap_name = females[len(shifts_base) % len(females)] if is_male else males[len(shifts_base) % len(males)]
        
        # Swap name in text
        swapped_text = re.sub(rf"\b{re.escape(orig_name)}\b", swap_name, orig_text, flags=re.IGNORECASE)
        
        # Baseline model inference
        p_base_orig = float(clf_base.predict_proba(vec_base.transform([orig_text]))[0, 1])
        p_base_swap = float(clf_base.predict_proba(vec_base.transform([swapped_text]))[0, 1])
        shifts_base.append(abs(p_base_swap - p_base_orig))
        
        # Redacted model inference
        red_orig = redact(orig_text, names=[orig_name])
        red_swap = redact(swapped_text, names=[swap_name])
        p_red_orig = float(clf_red.predict_proba(vec_red.transform([red_orig]))[0, 1])
        p_red_swap = float(clf_red.predict_proba(vec_red.transform([red_swap]))[0, 1])
        shifts_red.append(abs(p_red_swap - p_red_orig))
        
    return {
        "samples_evaluated": len(shifts_base),
        "baseline_mean_shift": round(float(np.mean(shifts_base)) * 100, 3),
        "baseline_max_shift": round(float(np.max(shifts_base)) * 100, 3),
        "redacted_mean_shift": round(float(np.mean(shifts_red)) * 100, 3),
        "redacted_max_shift": round(float(np.max(shifts_red)) * 100, 3),
        "shift_elimination_rate": round(
            float((np.mean(shifts_base) - np.mean(shifts_red)) / max(np.mean(shifts_base), 1e-6)) * 100, 2
        )
    }

def train_recruitment_models() -> Dict[str, Any]:
    """
    Trains Baseline, Redacted, and Mitigated models strictly on the 10,174 authentic Kaggle resumes.
    """
    print("1. Loading and preparing Kaggle AI Recruitment Pipeline Dataset (10,174 rows)...")
    df = load_and_preprocess_dataset()
    
    texts = np.asarray(df["clean_resume"].tolist(), dtype=object)
    y = np.asarray(df["y_hired"].values, dtype=int)
    genders = np.asarray(df["inferred_gender"].tolist(), dtype=object)
    names = np.asarray(df["first_name"].tolist(), dtype=object)
    
    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test, g_train, g_test, n_train, n_test = train_test_split(
        texts, y, genders, names, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train_raw = list(X_train_raw)
    X_test_raw = list(X_test_raw)
    
    # 2. Vectorizer tuned for real resume vocabulary
    vectorizer_raw = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z0-9_\-\[\]]+\b"
    )
    
    print("2. Training Baseline Model on raw resume text (containing real names)...")
    X_train_vec = vectorizer_raw.fit_transform(X_train_raw)
    X_test_vec = vectorizer_raw.transform(X_test_raw)
    
    clf_base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_base.fit(X_train_vec, y_train)
    
    pred_base = clf_base.predict(X_test_vec)
    prob_base = clf_base.predict_proba(X_test_vec)[:, 1]
    
    metrics_base = {
        "accuracy": round(float(accuracy_score(y_test, pred_base)), 4),
        "precision": round(float(precision_score(y_test, pred_base, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_base, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, prob_base)), 4)
    }
    
    known_mask_test = np.isin(g_test, ["Male", "Female"])
    audit_base = audit(
        y_true=y_test[known_mask_test],
        pred=pred_base[known_mask_test],
        sensitive_group=g_test[known_mask_test],
        unprivileged_value="Female",
        privileged_value="Male"
    )
    save_model_artifacts(vectorizer_raw, clf_base, name_prefix="recruitment_baseline")
    print(f"   Baseline Disparate Impact: {audit_base['disparate_impact_ratio']}")
    
    # 3. Train Redacted Model (identity signals stripped)
    print("3. Scrubbing identity signals and training Redacted Model...")
    X_train_red = [redact(t, names=[n] if n else None) for t, n in zip(X_train_raw, n_train)]
    X_test_red = [redact(t, names=[n] if n else None) for t, n in zip(X_test_raw, n_test)]
    
    vectorizer_red = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z0-9_\-\[\]]+\b"
    )
    X_train_red_vec = vectorizer_red.fit_transform(X_train_red)
    X_test_red_vec = vectorizer_red.transform(X_test_red)
    
    clf_red = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_red.fit(X_train_red_vec, y_train)
    
    pred_red = clf_red.predict(X_test_red_vec)
    prob_red = clf_red.predict_proba(X_test_red_vec)[:, 1]
    
    metrics_red = {
        "accuracy": round(float(accuracy_score(y_test, pred_red)), 4),
        "precision": round(float(precision_score(y_test, pred_red, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_red, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, prob_red)), 4)
    }
    
    audit_red = audit(
        y_true=y_test[known_mask_test],
        pred=pred_red[known_mask_test],
        sensitive_group=g_test[known_mask_test],
        unprivileged_value="Female",
        privileged_value="Male"
    )
    save_model_artifacts(vectorizer_red, clf_red, name_prefix="recruitment_redacted")
    print(f"   Redacted Disparate Impact: {audit_red['disparate_impact_ratio']}")
    
    # 4. Train Mitigated Model with AIF360 Sample Reweighing
    print("4. Applying AIF360 Sample Reweighing and training Mitigated Model...")
    known_mask_train = np.isin(g_train, ["Male", "Female"])
    weights_train = compute_reweighing_weights(
        y=y_train,
        sensitive_group=g_train,
        unprivileged_value="Female",
        privileged_value="Male"
    )
    
    clf_mit = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_mit.fit(X_train_red_vec, y_train, sample_weight=weights_train)
    
    pred_mit = clf_mit.predict(X_test_red_vec)
    prob_mit = clf_mit.predict_proba(X_test_red_vec)[:, 1]
    
    metrics_mit = {
        "accuracy": round(float(accuracy_score(y_test, pred_mit)), 4),
        "precision": round(float(precision_score(y_test, pred_mit, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_mit, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, prob_mit)), 4)
    }
    
    audit_mit = audit(
        y_true=y_test[known_mask_test],
        pred=pred_mit[known_mask_test],
        sensitive_group=g_test[known_mask_test],
        unprivileged_value="Female",
        privileged_value="Male"
    )
    save_model_artifacts(vectorizer_red, clf_mit, name_prefix="recruitment_mitigated")
    print(f"   Mitigated Disparate Impact: {audit_mit['disparate_impact_ratio']}")
    
    # 5. Compute Role-by-Role Real Disparity Breakdown
    print("5. Computing authentic role-by-role hiring rates...")
    role_breakdown = compute_role_breakdown(df)
    
    # 6. Evaluate Counterfactual Name Swap Sensitivity
    print("6. Evaluating counterfactual name swap sensitivity on real resumes...")
    df_test = pd.DataFrame({
        "clean_resume": X_test_raw,
        "first_name": n_test,
        "inferred_gender": g_test,
        "y_hired": y_test
    })
    swap_evaluation = evaluate_counterfactual_swaps(
        df_test=df_test,
        clf_base=clf_base,
        vec_base=vectorizer_raw,
        clf_red=clf_red,
        vec_red=vectorizer_red,
        n_samples=100
    )
    print(f"   Baseline Mean Name Shift: {swap_evaluation['baseline_mean_shift']}% (Max: {swap_evaluation['baseline_max_shift']}%)")
    print(f"   Redacted Mean Name Shift: {swap_evaluation['redacted_mean_shift']}% (Max: {swap_evaluation['redacted_max_shift']}%)")
    
    # 7. Save Comprehensive Benchmark Summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_name": "Kaggle AI Recruitment Pipeline Dataset (yaswanthkumary)",
        "total_rows": len(df),
        "identified_demographic_rows": int(np.sum(df["inferred_gender"].isin(["Male", "Female"]))),
        "overall_raw_rates": {
            "female_selection_rate": round(float(df[df["inferred_gender"] == "Female"]["y_hired"].mean()), 4),
            "male_selection_rate": round(float(df[df["inferred_gender"] == "Male"]["y_hired"].mean()), 4),
            "disparate_impact": round(
                float(df[df["inferred_gender"] == "Female"]["y_hired"].mean()) /
                float(df[df["inferred_gender"] == "Male"]["y_hired"].mean()), 4
            )
        },
        "role_breakdown": role_breakdown,
        "counterfactual_swap": swap_evaluation,
        "baseline": {
            "metrics": metrics_base,
            "audit": audit_base
        },
        "redacted": {
            "metrics": metrics_red,
            "audit": audit_red
        },
        "mitigated": {
            "metrics": metrics_mit,
            "audit": audit_mit
        }
    }
    
    with open(RESULTS_DIR / "recruitment_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("Successfully trained on authentic Kaggle dataset and serialized models to models/!")
    return summary

if __name__ == "__main__":
    train_recruitment_models()

