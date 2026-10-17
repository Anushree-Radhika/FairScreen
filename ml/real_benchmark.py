import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.audit import audit
from ml.features import extract_features
from ml.mitigate import compute_reweighing_weights
from ml.model import build_and_train
from ml.redact import redact

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
REAL_RESUMES_URL = "https://raw.githubusercontent.com/Sum02it/Resume-Screening/master/UpdatedResumeDataSet.csv"

def fetch_real_resumes() -> pd.DataFrame:
    """
    Downloads and caches the Kaggle Real Resumes Dataset (962 authentic resumes across 25 job domains).
    """
    cache_path = DATA_DIR / "real_resumes.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(REAL_RESUMES_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        content = response.read()
        df = pd.read_csv(io.BytesIO(content))
        df.to_csv(cache_path, index=False)
        return df

def run_real_resume_pipeline(target_category: str = "Data Science") -> Dict[str, Any]:
    """
    Trains and audits the FairScreen pipeline on authentic resume texts:
    1. Ingests 962 authentic resumes across 25 professional categories.
    2. Defines target screening role (e.g. Data Science / Technical roles).
    3. Evaluates raw text vs redacted text screening performance.
    """
    df = fetch_real_resumes()
    
    # Define binary qualification target: Matches target role or related technical roles
    tech_categories = {"Data Science", "Python Developer", "DevOps Engineer", "Java Developer", "Database"}
    if target_category == "Tech":
        df["target"] = df["Category"].isin(tech_categories).astype(int)
    else:
        df["target"] = (df["Category"] == target_category).astype(int)

    texts = df["Resume"].tolist()
    y = df["target"].values
    
    # 1. Train on Raw Authentic Text
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )
    
    vec_raw, clf_raw, raw_metrics = build_and_train(X_train_raw, y_train, model_type="logistic")
    X_test_vec = vec_raw.transform(X_test_raw)
    test_preds_raw = clf_raw.predict(X_test_vec)
    test_probs_raw = clf_raw.predict_proba(X_test_vec)[:, 1]
    
    raw_eval = {
        "accuracy": round(float(accuracy_score(y_test, test_preds_raw)), 4),
        "precision": round(float(precision_score(y_test, test_preds_raw, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds_raw, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, test_probs_raw)), 4)
    }

    # 2. Train on Redacted Text (Identity & demographic signals stripped from real resumes)
    redacted_texts = [redact(t) for t in texts]
    X_train_red, X_test_red, _, _ = train_test_split(
        redacted_texts, y, test_size=0.25, random_state=42, stratify=y
    )
    
    vec_red, clf_red, red_metrics = build_and_train(X_train_red, y_train, model_type="logistic")
    X_test_red_vec = vec_red.transform(X_test_red)
    test_preds_red = clf_red.predict(X_test_red_vec)
    test_probs_red = clf_red.predict_proba(X_test_red_vec)[:, 1]
    
    red_eval = {
        "accuracy": round(float(accuracy_score(y_test, test_preds_red)), 4),
        "precision": round(float(precision_score(y_test, test_preds_red, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds_red, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, test_probs_red)), 4)
    }

    return {
        "dataset_name": "Kaggle Real Resumes (UpdatedResumeDataSet)",
        "sample_size": len(df),
        "categories_count": df["Category"].nunique(),
        "target_screened_category": target_category,
        "raw_text_performance": raw_eval,
        "redacted_text_performance": red_eval,
        "utility_preservation": round(red_eval["accuracy"] / raw_eval["accuracy"], 4)
    }

def run_uci_adult_benchmark(sample_n: int = 12000) -> Dict[str, Any]:
    """
    Evaluates fairness and mitigation on the canonical real-world benchmark:
    UCI Adult Census Income (48,842 real humans, real demographics: sex, race, age).
    """
    adult_frame = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame.dropna()
    if sample_n and sample_n < len(adult_frame):
        df = adult_frame.sample(n=sample_n, random_state=42)
    else:
        df = adult_frame

    sensitive_sex = df["sex"].values
    y = (df["class"] == ">50K").astype(int).values

    num_cols = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X = preprocessor.fit_transform(df[num_cols + cat_cols])
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_sex, test_size=0.3, random_state=42, stratify=y
    )

    # 1. Baseline Model (Learns real-world societal wage & hiring disparities)
    clf_base = LogisticRegression(max_iter=1000, random_state=42)
    clf_base.fit(X_train, y_train)
    pred_base = clf_base.predict(X_test)
    prob_base = clf_base.predict_proba(X_test)[:, 1]
    
    audit_base = audit(y_test, pred_base, s_test, unprivileged_value="Female", privileged_value="Male")
    acc_base = float(accuracy_score(y_test, pred_base))
    auc_base = float(roc_auc_score(y_test, prob_base))

    # 2. Mitigated Model (AIF360 Sample Reweighing on real demographic distributions)
    weights_train = compute_reweighing_weights(y_train, s_train, unprivileged_value="Female", privileged_value="Male")
    clf_mit = LogisticRegression(max_iter=1000, random_state=42)
    clf_mit.fit(X_train, y_train, sample_weight=weights_train)
    pred_mit = clf_mit.predict(X_test)
    prob_mit = clf_mit.predict_proba(X_test)[:, 1]
    
    audit_mit = audit(y_test, pred_mit, s_test, unprivileged_value="Female", privileged_value="Male")
    acc_mit = float(accuracy_score(y_test, pred_mit))
    auc_mit = float(roc_auc_score(y_test, prob_mit))

    return {
        "dataset_name": "UCI Adult (Real-World Benchmark)",
        "sample_size": len(df),
        "unprivileged_group": "Female",
        "privileged_group": "Male",
        "baseline": {
            "accuracy": round(acc_base, 4),
            "auc": round(auc_base, 4),
            "disparate_impact": audit_base["disparate_impact_ratio"],
            "statistical_parity_diff": audit_base["statistical_parity_difference"],
            "equal_opportunity_diff": audit_base["equal_opportunity_difference"],
            "unprivileged_selection_rate": audit_base["selection_rate_unprivileged"],
            "privileged_selection_rate": audit_base["selection_rate_privileged"],
            "four_fifths_passed": audit_base["four_fifths_rule_passed"]
        },
        "mitigated": {
            "accuracy": round(acc_mit, 4),
            "auc": round(auc_mit, 4),
            "disparate_impact": audit_mit["disparate_impact_ratio"],
            "statistical_parity_diff": audit_mit["statistical_parity_difference"],
            "equal_opportunity_diff": audit_mit["equal_opportunity_difference"],
            "unprivileged_selection_rate": audit_mit["selection_rate_unprivileged"],
            "privileged_selection_rate": audit_mit["selection_rate_privileged"],
            "four_fifths_passed": audit_mit["four_fifths_rule_passed"]
        },
        "disparate_impact_improvement": round(audit_mit["disparate_impact_ratio"] - audit_base["disparate_impact_ratio"], 4),
        "accuracy_retained": round((acc_mit / acc_base) * 100, 2)
    }

def run_all_real_benchmarks():
    """Executes both real datasets and writes summary to data/results/real_benchmarks.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Running evaluation on Kaggle Real Resumes (962 authentic documents)...")
    resumes_result = run_real_resume_pipeline(target_category="Tech")
    print(f"   Real Resume Raw Text Accuracy: {resumes_result['raw_text_performance']['accuracy']}")
    print(f"   Real Resume Redacted Accuracy: {resumes_result['redacted_text_performance']['accuracy']}")
    print(f"   Utility Retention: {resumes_result['utility_preservation']*100}%")

    print("\n2. Running fairness audit on UCI Adult (Real Demographics & Economics)...")
    adult_result = run_uci_adult_benchmark(sample_n=12000)
    print(f"   Baseline Disparate Impact: {adult_result['baseline']['disparate_impact']}")
    print(f"   Mitigated Disparate Impact: {adult_result['mitigated']['disparate_impact']}")
    print(f"   Accuracy Retained: {adult_result['accuracy_retained']}%")

    all_results = {
        "real_resumes": resumes_result,
        "uci_adult": adult_result
    }

    out_file = RESULTS_DIR / "real_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll real-world benchmarks complete! Saved to {out_file}")
    return all_results

if __name__ == "__main__":
    run_all_real_benchmarks()
