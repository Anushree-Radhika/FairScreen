import json
from pathlib import Path
from ml.synth import make_candidates
from ml.redact import redact
from ml.model import build_and_train, save_model_artifacts
from ml.audit import audit
from ml.mitigate import train_mitigated_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

def run_training_pipeline():
    print("1. Generating synthetic biased candidate cohort (N=1200, bias=Female, strength=0.40)...")
    df = make_candidates(n=1200, bias_group="Female", bias_strength=0.40, seed=42)
    
    raw_texts = df["resume_text"].tolist()
    y_hired = df["hired"].values
    y_qualified = df["true_qualified"].values
    sensitive = df["gender"].values
    
    # 2. Train Baseline Model (Trained on raw text with historical biased outcomes)
    print("2. Training Baseline Screener Model...")
    vec_raw, clf_raw, raw_metrics = build_and_train(raw_texts, y_hired, model_type="logistic")
    save_model_artifacts(vec_raw, clf_raw, name_prefix="baseline")
    
    # Audit baseline model
    pred_raw = clf_raw.predict(vec_raw.transform(raw_texts))
    baseline_audit = audit(y_qualified, pred_raw, sensitive)
    print(f"   Baseline Disparate Impact: {baseline_audit['disparate_impact_ratio']} (Biased: {baseline_audit['is_biased']})")
    
    # 3. Train Redacted Model (Trained on text stripped of identity signals)
    print("3. Generating Redacted Resumes and Training Redacted Model...")
    all_names = df["first_name"].unique().tolist()
    redacted_texts = [redact(t, names=all_names) for t in raw_texts]
    
    vec_red, clf_red, red_metrics = build_and_train(redacted_texts, y_hired, model_type="logistic")
    save_model_artifacts(vec_red, clf_red, name_prefix="redacted")
    
    pred_red = clf_red.predict(vec_red.transform(redacted_texts))
    redacted_audit = audit(y_qualified, pred_red, sensitive)
    print(f"   Redacted Disparate Impact: {redacted_audit['disparate_impact_ratio']} (Biased: {redacted_audit['is_biased']})")
    
    # 4. Train Mitigated Model (Trained on redacted text + sample reweighing)
    print("4. Training Mitigated Model with AIF360 Sample Reweighing...")
    vec_mit, clf_mit, mit_metrics = train_mitigated_model(
        texts=redacted_texts,
        y=y_hired,
        sensitive_group=sensitive,
        unprivileged_value="Female",
        privileged_value="Male"
    )
    save_model_artifacts(vec_mit, clf_mit, name_prefix="mitigated")
    
    pred_mit = clf_mit.predict(vec_mit.transform(redacted_texts))
    mitigated_audit = audit(y_qualified, pred_mit, sensitive)
    print(f"   Mitigated Disparate Impact: {mitigated_audit['disparate_impact_ratio']} (Biased: {mitigated_audit['is_biased']})")
    
    # 5. Save summary benchmark results for dashboard UI
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_results = {
        "baseline": {
            "metrics": raw_metrics,
            "audit": baseline_audit
        },
        "redacted": {
            "metrics": red_metrics,
            "audit": redacted_audit
        },
        "mitigated": {
            "metrics": mit_metrics,
            "audit": mitigated_audit
        }
    }
    
    with open(RESULTS_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
        
    print("All models trained and serialized to models/, results saved to data/results/audit_summary.json!")

if __name__ == "__main__":
    run_training_pipeline()
