import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.ingest import extract_text_from_pdf, extract_sections, clean_text
from ml.redact import redact
from ml.model import load_model_artifacts
from ml.explain import explain

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
KAGGLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kaggle_recruitment" / "dataset.csv"

class FairScreenPipeline:
    """
    Backend Orchestrator for the FairScreen End-to-End Pipeline:
    PDF Ingestion -> Redaction -> Screening with Kaggle AI Recruitment Models -> Explanation.
    """
    def __init__(self):
        self._models_loaded = False
        self.baseline_vec = None
        self.baseline_clf = None
        self.redacted_vec = None
        self.redacted_clf = None
        self._kaggle_df = None
        self._ensure_models_loaded()
        self._load_kaggle_dataset()
        
    def _ensure_models_loaded(self):
        try:
            self.baseline_vec, self.baseline_clf = load_model_artifacts("recruitment_baseline")
            self.redacted_vec, self.redacted_clf = load_model_artifacts("recruitment_redacted")
            self._models_loaded = True
        except Exception:
            self._models_loaded = False

    def _load_kaggle_dataset(self):
        if KAGGLE_DATA_PATH.exists():
            try:
                self._kaggle_df = pd.read_csv(KAGGLE_DATA_PATH)
            except Exception:
                self._kaggle_df = None

    def get_available_roles(self) -> List[str]:
        """Returns all distinct job roles present in the Kaggle dataset."""
        if self._kaggle_df is not None and "Role" in self._kaggle_df.columns:
            roles = sorted(self._kaggle_df["Role"].dropna().unique().tolist())
            return roles
        return [
            "Data Scientist", "Software Engineer", "Machine Learning Engineer",
            "DevOps Engineer", "Cloud Architect", "Product Manager"
        ]

    def get_role_decision_insights(self, role: str) -> Dict[str, Any]:
        """Fetches real HR selection vs rejection reasons and sample candidates from the Kaggle dataset."""
        if self._kaggle_df is None or "Role" not in self._kaggle_df.columns:
            return {"selection_reasons": [], "rejection_reasons": [], "samples": []}

        sub = self._kaggle_df[self._kaggle_df["Role"].str.lower() == role.lower()]
        if len(sub) == 0:
            sub = self._kaggle_df

        sel = sub[sub["decision"].str.lower() == "select"]
        rej = sub[sub["decision"].str.lower() == "reject"]

        top_sel_reasons = sel["Reason_for_decision"].value_counts().head(4).to_dict()
        top_rej_reasons = rej["Reason_for_decision"].value_counts().head(4).to_dict()

        samples = []
        if len(sel) > 0:
            s_row = sel.iloc[0]
            samples.append({
                "type": "Selected Candidate (Real Kaggle Benchmark)",
                "name": s_row["Name"],
                "decision": "select",
                "reason": s_row["Reason_for_decision"],
                "resume_snippet": s_row["Resume"][:450]
            })
        if len(rej) > 0:
            r_row = rej.iloc[0]
            samples.append({
                "type": "Rejected Candidate (Real Kaggle Benchmark)",
                "name": r_row["Name"],
                "decision": "reject",
                "reason": r_row["Reason_for_decision"],
                "resume_snippet": r_row["Resume"][:450]
            })

        return {
            "role": role,
            "total_applicants_in_role": len(sub),
            "selection_rate": round(len(sel) / max(len(sub), 1), 3),
            "selection_reasons": top_sel_reasons,
            "rejection_reasons": top_rej_reasons,
            "benchmark_samples": samples
        }

    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Ingests and cleans an uploaded PDF file stream."""
        return extract_text_from_pdf(pdf_file)

    def screen_candidate(
        self,
        raw_text: str,
        candidate_first_name: Optional[str] = None,
        target_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation of candidate resume:
        1. Redaction of sensitive demographic features
        2. Scoring across Baseline and Redacted models trained on Kaggle data
        3. Explainability attribution & real Kaggle benchmark comparison
        """
        if not self._models_loaded:
            self._ensure_models_loaded()

        # 1. Clean & Redact
        clean = clean_text(raw_text)
        sections = extract_sections(clean)
        redacted_text = redact(clean)
        
        # 2. Score with models
        results = {
            "clean_text": clean,
            "redacted_text": redacted_text,
            "sections": sections,
            "models_available": self._models_loaded,
            "model_source": "Kaggle AI Recruitment Pipeline Dataset (10,174 rows)"
        }
        
        if self._models_loaded:
            # Baseline score (evaluating raw text with historical biases)
            X_base = self.baseline_vec.transform([clean])
            prob_base = float(self.baseline_clf.predict_proba(X_base)[0, 1])
            pred_base = int(prob_base >= 0.50)
            
            # Redacted score (evaluating identity-neutral text)
            X_red = self.redacted_vec.transform([redacted_text])
            prob_red = float(self.redacted_clf.predict_proba(X_red)[0, 1])
            pred_red = int(prob_red >= 0.50)
            
            # Explanation via identity-blind fair model
            explanation_data = explain(self.redacted_vec, self.redacted_clf, redacted_text)
            
            # Role insights from Kaggle
            role_to_check = target_role if target_role else "Data Scientist"
            insights = self.get_role_decision_insights(role_to_check)
            
            results.update({
                "scores": {
                    "baseline": {"probability": round(prob_base, 3), "shortlisted": bool(pred_base)},
                    "redacted": {"probability": round(prob_red, 3), "shortlisted": bool(pred_red)}
                },
                "explanation": explanation_data,
                "kaggle_role_insights": insights
            })
            
        return results

    def get_audit_summary(self) -> Dict[str, Any]:
        """Loads cached benchmark audit metrics from Kaggle recruitment dataset."""
        rec_summary_path = RESULTS_DIR / "recruitment_audit_summary.json"
        if rec_summary_path.exists():
            with open(rec_summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
