from typing import Any, Dict
import numpy as np
from sklearn.metrics import recall_score

def audit(
    y_true: np.ndarray,
    pred: np.ndarray,
    sensitive_group: np.ndarray,
    unprivileged_value: str = "Female",
    privileged_value: str = "Male"
) -> Dict[str, Any]:
    """
    Role 4 Contract:
    Calculates comprehensive algorithmic fairness and parity metrics.
    
    Metrics:
    - Disparate Impact Ratio (Selection rate unprivileged / privileged, threshold 0.80)
    - Statistical Parity Difference (Selection rate difference)
    - Equal Opportunity Difference (TPR_unprivileged - TPR_privileged)
    - Group-wise selection rates, precision, and recall
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    sensitive_group = np.asarray(sensitive_group)
    
    mask_unpriv = (sensitive_group == unprivileged_value)
    mask_priv = (sensitive_group == privileged_value)
    
    # Selection Rates (Positive Prediction Rate)
    sr_unpriv = float(np.mean(pred[mask_unpriv])) if np.sum(mask_unpriv) > 0 else 0.0
    sr_priv = float(np.mean(pred[mask_priv])) if np.sum(mask_priv) > 0 else 0.0
    
    # Disparate Impact (Four-Fifths Rule threshold: 0.80)
    disparate_impact = (sr_unpriv / sr_priv) if sr_priv > 0 else 1.0
    statistical_parity_diff = sr_unpriv - sr_priv
    
    # Equal Opportunity (Difference in True Positive Rates)
    tpr_unpriv = float(recall_score(y_true[mask_unpriv], pred[mask_unpriv], zero_division=0)) if np.sum(mask_unpriv) > 0 else 0.0
    tpr_priv = float(recall_score(y_true[mask_priv], pred[mask_priv], zero_division=0)) if np.sum(mask_priv) > 0 else 0.0
    equal_opportunity_diff = tpr_unpriv - tpr_priv
    
    is_biased = disparate_impact < 0.80
    
    return {
        "disparate_impact_ratio": round(disparate_impact, 4),
        "statistical_parity_difference": round(statistical_parity_diff, 4),
        "equal_opportunity_difference": round(equal_opportunity_diff, 4),
        "selection_rate_unprivileged": round(sr_unpriv, 4),
        "selection_rate_privileged": round(sr_priv, 4),
        "tpr_unprivileged": round(tpr_unpriv, 4),
        "tpr_privileged": round(tpr_priv, 4),
        "four_fifths_rule_passed": not is_biased,
        "is_biased": is_biased,
        "unprivileged_group": unprivileged_value,
        "privileged_group": privileged_value
    }
