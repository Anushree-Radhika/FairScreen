import json
from pathlib import Path
from typing import Any, Dict, Optional
from backend.ingest import extract_text_from_pdf, extract_sections, clean_text
from ml.redact import redact
from ml.model import load_model_artifacts
from ml.audit import swap_gender
from ml.explain import explain
from ml.synth import load_names

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

class FairScreenPipeline:
    """
    Backend Orchestrator for the FairScreen End-to-End Pipeline:
    PDF Ingestion -> Redaction -> Screening (Baseline vs Mitigated) -> Explanation -> Swap Test.
    """
    def __init__(self):
        self._models_loaded = False
        self.baseline_vec = None
        self.baseline_clf = None
        self.redacted_vec = None
        self.redacted_clf = None
        self.mitigated_vec = None
        self.mitigated_clf = None
        self._ensure_models_loaded()
        
    def _ensure_models_loaded(self):
        try:
            self.baseline_vec, self.baseline_clf = load_model_artifacts("baseline")
            self.redacted_vec, self.redacted_clf = load_model_artifacts("redacted")
            self.mitigated_vec, self.mitigated_clf = load_model_artifacts("mitigated")
            self._models_loaded = True
        except Exception:
            self._models_loaded = False

    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Ingests and cleans an uploaded PDF file stream."""
        return extract_text_from_pdf(pdf_file)

    def screen_candidate(self, raw_text: str, candidate_first_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation of candidate resume:
        1. Redaction of sensitive demographic features
        2. Scoring across Baseline, Redacted, and Mitigated models
        3. Explainability attribution
        4. Counterfactual gender swap test
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
            "models_available": self._models_loaded
        }
        
        if self._models_loaded:
            # Baseline score (evaluating raw text)
            X_base = self.baseline_vec.transform([clean])
            prob_base = float(self.baseline_clf.predict_proba(X_base)[0, 1])
            pred_base = int(prob_base >= 0.50)
            
            # Redacted score (evaluating identity-neutral text)
            X_red = self.redacted_vec.transform([redacted_text])
            prob_red = float(self.redacted_clf.predict_proba(X_red)[0, 1])
            pred_red = int(prob_red >= 0.50)
            
            # Mitigated score (evaluating debiased reweighed model)
            X_mit = self.mitigated_vec.transform([redacted_text])
            prob_mit = float(self.mitigated_clf.predict_proba(X_mit)[0, 1])
            pred_mit = int(prob_mit >= 0.50)
            
            # Explanation via mitigated model
            explanation_data = explain(self.mitigated_vec, self.mitigated_clf, redacted_text)
            
            # Counterfactual Gender Swap Test
            males, females = load_names()
            detected_name = candidate_first_name
            if not detected_name:
                first_word = clean.split()[0] if clean else "Candidate"
                detected_name = first_word if (first_word in males or first_word in females) else "Sarah"
                
            swapped_text, swapped_name = swap_gender(clean, detected_name)
            X_swap_base = self.baseline_vec.transform([swapped_text])
            prob_swap_base = float(self.baseline_clf.predict_proba(X_swap_base)[0, 1])
            
            X_swap_mit = self.mitigated_vec.transform([redact(swapped_text)])
            prob_swap_mit = float(self.mitigated_clf.predict_proba(X_swap_mit)[0, 1])
            
            results.update({
                "scores": {
                    "baseline": {"probability": round(prob_base, 3), "shortlisted": bool(pred_base)},
                    "redacted": {"probability": round(prob_red, 3), "shortlisted": bool(pred_red)},
                    "mitigated": {"probability": round(prob_mit, 3), "shortlisted": bool(pred_mit)}
                },
                "explanation": explanation_data,
                "counterfactual_swap": {
                    "original_name": detected_name,
                    "swapped_name": swapped_name,
                    "baseline_score_before": round(prob_base, 3),
                    "baseline_score_after": round(prob_swap_base, 3),
                    "baseline_score_delta": round(prob_swap_base - prob_base, 3),
                    "mitigated_score_before": round(prob_mit, 3),
                    "mitigated_score_after": round(prob_swap_mit, 3),
                    "mitigated_score_delta": round(prob_swap_mit - prob_mit, 3)
                }
            })
            
        return results

    def get_audit_summary(self) -> Dict[str, Any]:
        """Loads cached benchmark audit metrics."""
        summary_path = RESULTS_DIR / "audit_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
