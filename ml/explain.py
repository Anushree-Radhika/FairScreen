from typing import Any, Dict, List, Optional
import numpy as np

def explain(
    vectorizer: Any,
    classifier: Any,
    text: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Role 5 Contract:
    Provides transparent, interpretable, merit-based explanations for candidate scoring.
    Uses linear coefficients and token presence to identify key positive drivers (matched skills)
    and negative drivers (missing requirements).
    """
    if not hasattr(classifier, "coef_"):
        return {
            "summary": "Model decision based on non-linear ensemble weighting.",
            "positive_drivers": [],
            "negative_drivers": []
        }
        
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]
    
    # Vectorize input text
    X_vec = vectorizer.transform([text])
    non_zero_indices = X_vec.indices
    
    if len(non_zero_indices) == 0:
        return {
            "summary": "No domain skills or qualifying terms matched the evaluation vocabulary.",
            "positive_drivers": [],
            "negative_drivers": []
        }
        
    token_weights = []
    for idx in non_zero_indices:
        token = feature_names[idx]
        weight = coefs[idx] * X_vec[0, idx]
        token_weights.append((token, float(weight)))
        
    # Sort by impact
    token_weights.sort(key=lambda x: x[1], reverse=True)
    
    positive_drivers = [w for w in token_weights if w[1] > 0][:top_k]
    negative_drivers = [w for w in token_weights if w[1] < 0][-top_k:]
    
    # Generate human-readable summary
    top_pos_tokens = ", ".join([f"'{t[0]}'" for t in positive_drivers[:3]])
    
    if positive_drivers and positive_drivers[0][1] > 0.3:
        summary = (
            f"Candidate score is strongly propelled by key competencies: {top_pos_tokens}. "
            f"The profile demonstrates alignment with core technical role requirements."
        )
    else:
        summary = (
            "Candidate score reflects limited overlap with the target role's core technical requirements. "
            "Additional technical competency signals would strengthen the profile."
        )
        
    return {
        "summary": summary,
        "positive_drivers": [{"token": t[0], "weight": round(t[1], 3)} for t in positive_drivers],
        "negative_drivers": [{"token": t[0], "weight": round(t[1], 3)} for t in negative_drivers]
    }
